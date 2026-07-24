#ifndef HPP_GUARD_ARGMIN_QP_SPARSE_ADMM_QP_H
#define HPP_GUARD_ARGMIN_QP_SPARSE_ADMM_QP_H

// Sparse operator-splitting QP solver (OSQP-class ADMM) over CSC data.
//
// Solves: min 0.5 * x^T P x + q^T x  s.t.  l <= A x <= u
//
// P and A are column-major Eigen sparse matrices; P is expected FULL symmetric
// (no triangular convention is imposed on the caller). The solver copies and
// compresses its own storage, so the caller's matrices are neither mutated nor
// aliased and need not be compressed on entry.
//
// Each ADMM iteration solves the full quasi-definite KKT system
//   [[ P + sigma*I,       A^T      ]] [x_tilde]   [ sigma*x - q      ]
//   [[ A,           -diag(rho)^-1  ]] [  nu   ] = [ z - rho^-1 * y   ]
// through a no-pivot simplicial LDL^T factorization with AMD ordering. Because
// the KKT is solved in this indirect-free form, z_tilde is recovered from the
// dual block as z + rho^-1 * (nu - y) rather than by a second product with A.
// The sparsity pattern is analyzed once, at pose; an accepted adaptive-rho
// update is a values-only numeric refactorization on that same pattern.
//
// Problem data is Ruiz-equilibrated (diagonal D, E and a cost scalar c) at pose
// time; those factors are frozen and reused verbatim by resolve(), the
// vectors-only (q, l, u) control-step path. A sparsity-pattern change is a
// re-pose, never a resolve: resolve() accepts no matrix argument at all.
// Convergence and the primal/dual infeasibility certificates are evaluated on
// UNSCALED residuals. The adaptive-rho schedule triggers on a fixed iteration
// interval, never on a measured duration, so the solver reads no clock. A
// delta-regularized reduced-KKT polish sized to the active set (active set from
// dual signs that clear a relative threshold, iterative refinement against the
// unregularized system, accepted only when both residuals improve and the
// objective does not worsen) lifts the accepted iterate.
//
// Contract: the factorization is computed once at pose, and pose allocates its
// storage -- a one-time real-time setup cost. resolve() performs no
// factorization and no pattern work, and, given a warm pose and polish off, its
// vectors-only iteration performs no heap allocation; that property is held by a
// labeled allocation gate. Polish re-analyzes and re-allocates its reduced KKT
// whenever the active-set pattern, the delta or the pose changes -- which a
// real-time tracking workload does on nearly every step -- so a real-time
// deployment runs with polish off. The allocation-free property is the resolve
// iteration only -- never the whole solver, and never pose or polish -- so this
// is a host-tier solver.
//
// Reference: Stellato, Banjac, Goulart, Bemporad, Boyd (2020), "OSQP: An
//            operator splitting solver for quadratic programs," Math. Prog.
//            Comp. 12:637-672 -- Algorithm 1 (Section 3), Section 5.1 (Ruiz
//            equilibration), Section 5.2 (adaptive rho), Section 5.3 (polish);
//            Vanderbei (1995), "Symmetric quasi-definite matrices," SIAM J.
//            Optim. 5(1):100-113, for the strong factorizability that makes the
//            no-pivot LDL^T legitimate on this indefinite matrix. The
//            equality-row rho boost 1e3 and the clamp range [1e-6, 1e6] follow
//            Section 5.2 and the reference implementation; the
//            solved-inaccurate report uses a relaxed tolerance at factor
//            K = 10, OSQP's convention on the iteration cap.

#include "argmin/qp/detail/sparse_kkt.h"
#include "argmin/qp/detail/polish_accept.h"
#include "argmin/options/sparse_qp_options.h"
#include "argmin/qp/qp_types.h"
#include "argmin/expected.h"
#include "argmin/types.h"

#include <Eigen/SparseCore>

#include <cmath>
#include <vector>
#include <cstddef>
#include <optional>
#include <algorithm>

namespace argmin
{

template <typename Scalar = double>
class sparse_admm_qp_solver
{
public:
    using sparse_type = Eigen::SparseMatrix<Scalar, Eigen::ColMajor>;
    using index_type = typename sparse_type::StorageIndex;
    using result_type = qp_result<Scalar>;

    expected<result_type, qp_error>
    solve(const sparse_type& P, const vector<Scalar>& q, const sparse_type& A,
          const vector<Scalar>& l, const vector<Scalar>& u, const sparse_qp_options& opts = {})
    {
        result_type out;
        if(auto err = solve_into(P, q, A, l, u, out, opts))
            return argmin::unexpected<qp_error>{*err};
        return out;
    }

    expected<result_type, qp_error>
    resolve(const vector<Scalar>& q, const vector<Scalar>& l, const vector<Scalar>& u,
            const sparse_qp_options& opts = {})
    {
        result_type out;
        if(auto err = resolve_into(q, l, u, out, opts))
            return argmin::unexpected<qp_error>{*err};
        return out;
    }

    // Fill-into sibling of solve(): poses the problem (copy + compress + Ruiz
    // scaling + symbolic analysis + factorization) and runs warm-started ADMM
    // into a caller-owned result. Returns an error only on an argument or
    // precondition violation; solver outcomes travel in out.status.
    std::optional<qp_error>
    solve_into(const sparse_type& P, const vector<Scalar>& q, const sparse_type& A,
               const vector<Scalar>& l, const vector<Scalar>& u, result_type& out,
               const sparse_qp_options& opts = {})
    {
        if(auto err = validate_full(P, q, A, l, u))
            return err;
        if(!pose(P, q, A, l, u, opts))
            return qp_error::invalid_problem;
        run_admm(opts);
        finalize(opts, out);
        return std::nullopt;
    }

    // Vectors-only path: updates (q, l, u), reuses the frozen Ruiz factors and
    // the existing factorization, warm-starts from the retained iterate. A
    // pattern change cannot arrive here, because no matrix crosses this
    // boundary; it requires a fresh solve_into().
    std::optional<qp_error>
    resolve_into(const vector<Scalar>& q, const vector<Scalar>& l, const vector<Scalar>& u,
                 result_type& out, const sparse_qp_options& opts = {})
    {
        if(!posed_)
            return qp_error::invalid_problem;
        if(static_cast<int>(q.size()) != n_ || static_cast<int>(l.size()) != m_
           || static_cast<int>(u.size()) != m_)
            return qp_error::dimension_mismatch;
        if(!q.allFinite())
            return qp_error::non_finite_input;
        for(int i = 0; i < m_; ++i)
        {
            if(std::isnan(l[i]) || std::isnan(u[i]))
                return qp_error::non_finite_input;
            if(l[i] > u[i])
                return qp_error::invalid_bounds;
        }
        q_ = q;
        if(m_ > 0)
        {
            l_ = l;
            u_ = u;
        }
        rescale_vectors();
        seed_iterates(opts);
        run_admm(opts);
        finalize(opts, out);
        return std::nullopt;
    }

    void warm_start(const vector<Scalar>& x, const vector<Scalar>& y)
    {
        ws_x_ = x;
        ws_y_ = y;
        has_warm_ = true;
    }

    void reset()
    {
        has_warm_ = false;
        x_.setZero();
        y_.setZero();
        z_.setZero();
    }

    // Read-only view of the Ruiz equilibration factors frozen at pose time. A
    // vectors-only resolve() reuses them verbatim (it never re-equilibrates),
    // so a caller can assert they are unchanged across a resolve.
    Scalar scaling_cost() const { return c_; }
    const vector<Scalar>& scaling_primal() const { return D_; }
    const vector<Scalar>& scaling_dual() const { return E_; }

    // Lifetime counts of the linear-algebra work actually performed, so the
    // factor-once-at-pose contract is observable rather than merely documented:
    // across a resolve() the iteration counts must not move except for the
    // numeric refactorization an accepted adaptive-rho update performs. The
    // polish is counted separately because it reuses its reduced-KKT analysis
    // and numeric factor whenever the active-set pattern, the delta and the pose
    // are unchanged, and re-analyzes only when one of those changes or the
    // factorization fails, so its counters increment once per analysis actually
    // performed rather than once per call.
    std::size_t iteration_analyses() const { return kkt_.symbolic_analyses(); }
    std::size_t iteration_factorizations() const { return kkt_.numeric_factorizations(); }
    std::size_t polish_analyses() const { return polish_kkt_.symbolic_analyses(); }
    std::size_t polish_factorizations() const { return polish_kkt_.numeric_factorizations(); }

    // Measurement and test seam, not a solver option: toggles the pattern-gated
    // reuse of the polish reduced-KKT analysis and factor so an in-process A/B
    // and the bit-identity correctness test can compare the reusing path against
    // the always-re-analyzing path. Enabled by default; disabling forces a
    // re-analysis on every polish call, reproducing the pre-reuse behavior.
    void set_polish_reuse(bool enabled) { polish_reuse_enabled_ = enabled; }
    bool polish_reuse() const { return polish_reuse_enabled_; }

private:
    static constexpr double rho_min_ = 1e-6;
    static constexpr double rho_max_ = 1e6;
    static constexpr double eq_boost_ = 1e3;
    static constexpr double eq_tol_ = 1e-9;
    static constexpr int inaccurate_factor_ = 10;
    static constexpr double polish_active_tol_ = 1e-12;

    // The single site at which symbolic analysis is performed. Pose reaches it
    // for the iteration KKT and the polish reaches it for its own reduced KKT;
    // nothing on the resolve path does.
    static bool factor_fresh(detail::sparse_kkt<Scalar>& kkt, const sparse_type& P,
                             const sparse_type& A, Scalar sigma,
                             const std::vector<Scalar>& rho_inv)
    {
        return kkt.analyze(P, A, sigma, rho_inv) && kkt.refactorize(rho_inv);
    }

    // Walks the stored entries rather than the value array, because the caller
    // is not required to hand over a compressed matrix and coeffs() asserts on
    // an uncompressed one.
    static bool all_finite(const sparse_type& M)
    {
        for(int j = 0; j < static_cast<int>(M.cols()); ++j)
            for(typename sparse_type::InnerIterator it(M, j); it; ++it)
                if(!std::isfinite(it.value()))
                    return false;
        return true;
    }

    std::optional<qp_error>
    validate_full(const sparse_type& P, const vector<Scalar>& q, const sparse_type& A,
                  const vector<Scalar>& l, const vector<Scalar>& u) const
    {
        const int n = static_cast<int>(P.rows());
        const int m = static_cast<int>(A.rows());
        if(static_cast<int>(P.cols()) != n || static_cast<int>(A.cols()) != n
           || static_cast<int>(q.size()) != n || static_cast<int>(l.size()) != m
           || static_cast<int>(u.size()) != m)
            return qp_error::dimension_mismatch;
        if(n <= 0)
            return qp_error::dimension_mismatch;
        if(!all_finite(P) || !all_finite(A) || !q.allFinite())
            return qp_error::non_finite_input;
        for(int i = 0; i < m; ++i)
        {
            if(std::isnan(l[i]) || std::isnan(u[i]))
                return qp_error::non_finite_input;
            if(l[i] > u[i])
                return qp_error::invalid_bounds;
        }
        return std::nullopt;
    }

    bool pose(const sparse_type& P, const vector<Scalar>& q, const sparse_type& A,
              const vector<Scalar>& l, const vector<Scalar>& u, const sparse_qp_options& opts)
    {
        const int n = static_cast<int>(P.rows());
        const int m = static_cast<int>(A.rows());
        // A retained iterate from a differently shaped problem carries no
        // meaning, so a shape change drops it rather than truncating it.
        if(n != n_ || m != m_)
            has_warm_ = false;
        posed_ = false;
        n_ = n;
        m_ = m;
        sigma_ = static_cast<Scalar>(opts.sigma);

        P_ = P;
        A_ = A;
        P_.makeCompressed();
        A_.makeCompressed();
        q_ = q;
        l_ = l;
        u_ = u;

        D_.resize(n_);
        scan_n_.resize(n_);
        q_scaled_.resize(n_);
        E_.resize(m_);
        Einv_.resize(m_);
        scan_m_.resize(m_);
        l_scaled_.resize(m_);
        u_scaled_.resize(m_);
        rho_vec_.resize(m_);
        rho_inv_vec_.resize(m_);
        base_mult_.resize(m_);
        rho_inv_.assign(static_cast<std::size_t>(m_), Scalar(1));

        rhs_kkt_.resize(n_ + m_);
        sol_kkt_.resize(n_ + m_);
        x_.resize(n_);
        x_tilde_.resize(n_);
        x_iter_prev_.resize(n_);
        x_unscaled_.resize(n_);
        Px_buf_.resize(n_);
        Aty_buf_.resize(n_);
        rd_.resize(n_);
        ntmp_.resize(n_);
        ntmp2_.resize(n_);
        px_x_.resize(n_);
        z_.resize(m_);
        z_tilde_.resize(m_);
        z_prev_.resize(m_);
        y_.resize(m_);
        y_iter_prev_.resize(m_);
        y_unscaled_.resize(m_);
        z_unscaled_.resize(m_);
        Ax_buf_.resize(m_);
        rp_.resize(m_);
        mtmp_.resize(m_);
        mtmp2_.resize(m_);
        px_y_.resize(m_);
        act_row_.assign(static_cast<std::size_t>(m_), -1);
        active_lower_.clear();
        active_upper_.clear();
        active_lower_.reserve(static_cast<std::size_t>(m_));
        active_upper_.reserve(static_cast<std::size_t>(m_));
        // A re-pose can change P, A or the shape, so any cached polish factor is
        // no longer bit-identical to a freshly analyzed one: drop it.
        polish_cache_valid_ = false;

        ruiz(opts);
        if(m_ > 0)
            Einv_ = E_.cwiseInverse();
        init_rho(opts);
        if(!factor_fresh(kkt_, P_scaled_, A_scaled_, sigma_, rho_inv_))
            return false;
        seed_iterates(opts);
        posed_ = true;
        return true;
    }

    // Modified Ruiz equilibration on the stacked KKT form [P A^T; A 0] plus a
    // cost scaling c. Reference: Stellato et al. 2020, Section 5.1.
    void ruiz(const sparse_qp_options& opts)
    {
        P_scaled_ = P_;
        A_scaled_ = A_;
        q_scaled_ = q_;
        D_.setOnes();
        if(m_ > 0)
            E_.setOnes();
        c_ = Scalar(1);

        for(int it = 0; it < static_cast<int>(opts.scaling); ++it)
        {
            for(int j = 0; j < n_; ++j)
            {
                Scalar d = Scalar(0);
                for(typename sparse_type::InnerIterator pit(P_scaled_, j); pit; ++pit)
                    d = std::max(d, std::abs(pit.value()));
                for(typename sparse_type::InnerIterator ait(A_scaled_, j); ait; ++ait)
                    d = std::max(d, std::abs(ait.value()));
                scan_n_[j] = d > Scalar(0) ? Scalar(1) / std::sqrt(d) : Scalar(1);
            }
            if(m_ > 0)
            {
                scan_m_.setZero();
                for(int j = 0; j < n_; ++j)
                    for(typename sparse_type::InnerIterator ait(A_scaled_, j); ait; ++ait)
                    {
                        const int i = static_cast<int>(ait.row());
                        scan_m_[i] = std::max(scan_m_[i], std::abs(ait.value()));
                    }
                for(int i = 0; i < m_; ++i)
                    scan_m_[i] =
                        scan_m_[i] > Scalar(0) ? Scalar(1) / std::sqrt(scan_m_[i]) : Scalar(1);
            }

            for(int j = 0; j < n_; ++j)
                for(typename sparse_type::InnerIterator pit(P_scaled_, j); pit; ++pit)
                    pit.valueRef() *= scan_n_[static_cast<int>(pit.row())] * scan_n_[j];
            for(int j = 0; j < n_; ++j)
                for(typename sparse_type::InnerIterator ait(A_scaled_, j); ait; ++ait)
                    ait.valueRef() *= scan_m_[static_cast<int>(ait.row())] * scan_n_[j];
            q_scaled_.array() *= scan_n_.array();
            D_.array() *= scan_n_.array();
            if(m_ > 0)
                E_.array() *= scan_m_.array();

            Scalar mean_pcol = Scalar(0);
            for(int j = 0; j < n_; ++j)
            {
                Scalar cmax = Scalar(0);
                for(typename sparse_type::InnerIterator pit(P_scaled_, j); pit; ++pit)
                    cmax = std::max(cmax, std::abs(pit.value()));
                mean_pcol += cmax;
            }
            mean_pcol /= static_cast<Scalar>(n_);
            const Scalar qn = q_scaled_.cwiseAbs().maxCoeff();
            const Scalar denom = std::max(mean_pcol, qn);
            const Scalar gamma = denom > Scalar(0) ? Scalar(1) / denom : Scalar(1);
            P_scaled_.coeffs() *= gamma;
            q_scaled_ *= gamma;
            c_ *= gamma;
        }

        if(m_ > 0)
        {
            l_scaled_ = E_.cwiseProduct(l_);
            u_scaled_ = E_.cwiseProduct(u_);
        }
    }

    void rescale_vectors()
    {
        q_scaled_ = c_ * D_.cwiseProduct(q_);
        if(m_ > 0)
        {
            l_scaled_ = E_.cwiseProduct(l_);
            u_scaled_ = E_.cwiseProduct(u_);
        }
    }

    void init_rho(const sparse_qp_options& opts)
    {
        current_rho_ = static_cast<Scalar>(opts.rho);
        for(int i = 0; i < m_; ++i)
        {
            const bool eq = std::isfinite(l_[i]) && std::isfinite(u_[i])
                            && (u_[i] - l_[i]) <= Scalar(eq_tol_);
            base_mult_[i] = eq ? Scalar(eq_boost_) : Scalar(1);
        }
        rebuild_rho_vec();
    }

    void rebuild_rho_vec()
    {
        for(int i = 0; i < m_; ++i)
            rho_vec_[i] =
                std::clamp(current_rho_ * base_mult_[i], Scalar(rho_min_), Scalar(rho_max_));
        if(m_ > 0)
        {
            rho_inv_vec_ = rho_vec_.cwiseInverse();
            for(int i = 0; i < m_; ++i)
                rho_inv_[static_cast<std::size_t>(i)] = rho_inv_vec_[i];
        }
    }

    void seed_iterates(const sparse_qp_options& opts)
    {
        if(opts.warm_start && has_warm_ && static_cast<int>(ws_x_.size()) == n_)
        {
            x_ = ws_x_.cwiseQuotient(D_);
            if(m_ > 0 && static_cast<int>(ws_y_.size()) == m_)
            {
                y_ = c_ * ws_y_.cwiseQuotient(E_);
                mtmp_.noalias() = A_scaled_ * x_;
                z_ = mtmp_.cwiseMax(l_scaled_).cwiseMin(u_scaled_);
                return;
            }
            if(m_ > 0)
            {
                y_.setZero();
                z_.setZero();
            }
            return;
        }
        x_.setZero();
        if(m_ > 0)
        {
            y_.setZero();
            z_.setZero();
        }
    }

    // Algorithm 1, with x_tilde and nu read out of one full-KKT solve and
    // z_tilde recovered from the dual block.
    void admm_step(const sparse_qp_options& opts)
    {
        const Scalar alpha = static_cast<Scalar>(opts.alpha);
        x_iter_prev_ = x_;
        if(m_ > 0)
            y_iter_prev_ = y_;

        rhs_kkt_.head(n_) = sigma_ * x_ - q_scaled_;
        if(m_ > 0)
            rhs_kkt_.tail(m_) = z_ - rho_inv_vec_.cwiseProduct(y_);
        kkt_.solve_into(rhs_kkt_, sol_kkt_);
        x_tilde_ = sol_kkt_.head(n_);
        x_ = alpha * x_tilde_ + (Scalar(1) - alpha) * x_;
        if(m_ > 0)
        {
            z_tilde_ = z_ + rho_inv_vec_.cwiseProduct(sol_kkt_.tail(m_) - y_);
            z_prev_ = z_;
            z_ = (alpha * z_tilde_ + (Scalar(1) - alpha) * z_prev_
                  + rho_inv_vec_.cwiseProduct(y_))
                     .cwiseMax(l_scaled_)
                     .cwiseMin(u_scaled_);
            y_ += rho_vec_.cwiseProduct(alpha * z_tilde_ + (Scalar(1) - alpha) * z_prev_ - z_);
        }
    }

    void run_admm(const sparse_qp_options& opts)
    {
        status_ = qp_solve_status::max_iterations;
        iters_ = 0;
        const int maxit = static_cast<int>(opts.max_iterations);
        const int check = std::max(1, static_cast<int>(opts.check_termination));
        const int rho_iv = static_cast<int>(opts.adaptive_rho_interval);
        for(int k = 0; k < maxit; ++k)
        {
            admm_step(opts);
            iters_ = k + 1;
            bool have_resid = false;
            if((k + 1) % check == 0)
            {
                refresh_residuals();
                have_resid = true;
                if(converged(opts))
                {
                    status_ = qp_solve_status::solved;
                    break;
                }
                if(check_primal_infeasible(opts))
                {
                    status_ = qp_solve_status::primal_infeasible;
                    break;
                }
                if(check_dual_infeasible(opts))
                {
                    status_ = qp_solve_status::dual_infeasible;
                    break;
                }
            }
            if(opts.adaptive_rho && rho_iv > 0 && (k + 1) % rho_iv == 0)
            {
                if(!have_resid)
                    refresh_residuals();
                maybe_update_rho(opts);
            }
        }
        if(status_ == qp_solve_status::max_iterations)
        {
            refresh_residuals();
            if(relaxed_converged(opts))
                status_ = qp_solve_status::solved_inaccurate;
        }
    }

    void refresh_residuals()
    {
        compute_unscaled_from_scaled();
        residual_norms(x_unscaled_, y_unscaled_, z_unscaled_, rp_norm_, rd_norm_, prim_scale_,
                       dual_scale_);
    }

    void compute_unscaled_from_scaled()
    {
        x_unscaled_ = D_.cwiseProduct(x_);
        if(m_ > 0)
        {
            y_unscaled_ = E_.cwiseProduct(y_) / c_;
            z_unscaled_ = Einv_.cwiseProduct(z_);
        }
    }

    void residual_norms(const vector<Scalar>& xu, const vector<Scalar>& yu,
                        const vector<Scalar>& zu, Scalar& rpn, Scalar& rdn, Scalar& ps,
                        Scalar& ds)
    {
        Px_buf_.noalias() = P_ * xu;
        rd_ = Px_buf_ + q_;
        Scalar ax_inf = Scalar(0);
        Scalar z_inf = Scalar(0);
        Scalar aty_inf = Scalar(0);
        rpn = Scalar(0);
        if(m_ > 0)
        {
            Ax_buf_.noalias() = A_ * xu;
            Aty_buf_.noalias() = A_.transpose() * yu;
            rd_ += Aty_buf_;
            rp_ = Ax_buf_ - zu;
            rpn = rp_.cwiseAbs().maxCoeff();
            ax_inf = Ax_buf_.cwiseAbs().maxCoeff();
            z_inf = zu.cwiseAbs().maxCoeff();
            aty_inf = Aty_buf_.cwiseAbs().maxCoeff();
        }
        rdn = rd_.cwiseAbs().maxCoeff();
        const Scalar q_inf = q_.cwiseAbs().maxCoeff();
        const Scalar px_inf = Px_buf_.cwiseAbs().maxCoeff();
        ps = std::max(ax_inf, z_inf);
        ds = std::max(px_inf, std::max(aty_inf, q_inf));
    }

    bool converged(const sparse_qp_options& o) const
    {
        const Scalar ep = static_cast<Scalar>(o.eps_abs);
        const Scalar er = static_cast<Scalar>(o.eps_rel);
        const bool p = m_ == 0 || rp_norm_ <= ep + er * prim_scale_;
        const bool d = rd_norm_ <= ep + er * dual_scale_;
        return p && d;
    }

    bool relaxed_converged(const sparse_qp_options& o) const
    {
        const Scalar ep = Scalar(inaccurate_factor_) * static_cast<Scalar>(o.eps_abs);
        const Scalar er = Scalar(inaccurate_factor_) * static_cast<Scalar>(o.eps_rel);
        const bool p = m_ == 0 || rp_norm_ <= ep + er * prim_scale_;
        const bool d = rd_norm_ <= ep + er * dual_scale_;
        return p && d;
    }

    // Certificates from the successive-iterate deltas dy, dx (Banjac et al.
    // 2019; Stellato et al. 2020, Section 3.4). Deltas are unscaled so the
    // bound tests use the caller's l, u.
    bool check_primal_infeasible(const sparse_qp_options& o)
    {
        if(m_ == 0)
            return false;
        mtmp_ = E_.cwiseProduct(y_ - y_iter_prev_) / c_;
        const Scalar dy_inf = mtmp_.cwiseAbs().maxCoeff();
        if(dy_inf <= Scalar(0))
            return false;
        ntmp_.noalias() = A_.transpose() * mtmp_;
        if(ntmp_.cwiseAbs().maxCoeff() > static_cast<Scalar>(o.eps_prim_inf) * dy_inf)
            return false;
        Scalar sup = Scalar(0);
        for(int i = 0; i < m_; ++i)
        {
            const Scalar d = mtmp_[i];
            if(d > Scalar(0))
                sup += u_[i] * d;
            else if(d < Scalar(0))
                sup += l_[i] * d;
        }
        // Scaled rather than a bare sign test: when dy lands in the null space
        // of A^T -- which redundant rows produce -- sup is a round-off quantity
        // whose sign is arbitrary, and a bare test would certify a feasible
        // problem as infeasible.
        return sup < -static_cast<Scalar>(o.eps_prim_inf) * dy_inf;
    }

    bool check_dual_infeasible(const sparse_qp_options& o)
    {
        ntmp_ = D_.cwiseProduct(x_ - x_iter_prev_);
        const Scalar dx_inf = ntmp_.cwiseAbs().maxCoeff();
        if(dx_inf <= Scalar(0))
            return false;
        ntmp2_.noalias() = P_ * ntmp_;
        if(ntmp2_.cwiseAbs().maxCoeff() > static_cast<Scalar>(o.eps_dual_inf) * dx_inf)
            return false;
        if(q_.dot(ntmp_) > -static_cast<Scalar>(o.eps_dual_inf) * dx_inf)
            return false;
        if(m_ > 0)
        {
            mtmp_.noalias() = A_ * ntmp_;
            const Scalar tol = static_cast<Scalar>(o.eps_dual_inf) * dx_inf;
            for(int i = 0; i < m_; ++i)
            {
                const Scalar adx = mtmp_[i];
                if(std::isfinite(u_[i]) && adx > tol)
                    return false;
                if(std::isfinite(l_[i]) && adx < -tol)
                    return false;
            }
        }
        return true;
    }

    void maybe_update_rho(const sparse_qp_options& o)
    {
        if(m_ == 0)
            return;
        const Scalar num = prim_scale_ > Scalar(0) ? rp_norm_ / prim_scale_ : rp_norm_;
        const Scalar den = dual_scale_ > Scalar(0) ? rd_norm_ / dual_scale_ : rd_norm_;
        if(num <= Scalar(0) || den <= Scalar(0))
            return;
        const Scalar ratio = std::sqrt(num / den);
        const Scalar tol = static_cast<Scalar>(o.adaptive_rho_tolerance);
        if(ratio > tol || ratio < Scalar(1) / tol)
        {
            current_rho_ = std::clamp(current_rho_ * ratio, Scalar(rho_min_), Scalar(rho_max_));
            rebuild_rho_vec();
            kkt_.refactorize(rho_inv_);
        }
    }

    void finalize(const sparse_qp_options& o, result_type& out)
    {
        // The ADMM iterate's primal residual is Ax - z against the operator-
        // splitting auxiliary z (Stellato et al. 2020, Section 3.4), not the
        // feasibility violation: z and Ax generally differ, so this residual is
        // the honest metric the polish must beat.
        compute_unscaled_from_scaled();
        Scalar rpn = Scalar(0);
        Scalar rdn = Scalar(0);
        Scalar ps = Scalar(0);
        Scalar ds = Scalar(0);
        residual_norms(x_unscaled_, y_unscaled_, z_unscaled_, rpn, rdn, ps, ds);

        ntmp_.noalias() = P_ * x_unscaled_;
        const Scalar iterate_objective =
            Scalar(0.5) * x_unscaled_.dot(ntmp_) + q_.dot(x_unscaled_);

        bool polished = false;
        if(o.polish
           && (status_ == qp_solve_status::solved || status_ == qp_solve_status::solved_inaccurate))
        {
            if(try_polish(o, rpn, rdn, iterate_objective))
            {
                polished = true;
                x_unscaled_ = px_x_;
                if(m_ > 0)
                {
                    y_unscaled_ = px_y_;
                    Ax_buf_.noalias() = A_ * x_unscaled_;
                    z_unscaled_ = Ax_buf_.cwiseMax(l_).cwiseMin(u_);
                }
                residual_norms(x_unscaled_, y_unscaled_, z_unscaled_, rpn, rdn, ps, ds);
            }
        }

        out.x = x_unscaled_;
        if(m_ > 0)
            out.y = y_unscaled_;
        else
            out.y.resize(0);
        out.status = status_;
        out.iterations = iters_;
        out.polished = polished;
        out.primal_residual = m_ > 0 ? rpn : Scalar(0);
        out.dual_residual = rdn;
        ntmp_.noalias() = P_ * x_unscaled_;
        out.objective_value = Scalar(0.5) * x_unscaled_.dot(ntmp_) + q_.dot(x_unscaled_);

        if(o.warm_start)
        {
            ws_x_ = x_unscaled_;
            if(m_ > 0)
                ws_y_ = y_unscaled_;
            has_warm_ = true;
        }
    }

    // Delta-regularized reduced-KKT polish (Stellato et al. 2020, Section 5.3).
    // The active set comes from the unscaled dual signs and the reduced system
    // is sized to it. Because P, A and delta are frozen across a resolve, an
    // unchanged active-set pattern yields a bit-identical reduced KKT, so its
    // symbolic analysis and numeric factor are reused: the pattern, the delta
    // and the pose are compared per call and the analysis is redone only when
    // one of them changes or the factorization fails. The comparison is exact
    // element-wise index-vector equality, never a hash, so a changed pattern can
    // never reuse a stale factor. A factorization failure is a rejection of the
    // polish, not an error.
    //
    // Two departures from the reference implementation's accept rule, both
    // forced by the same observed failure: an inactive row whose multiplier is
    // pure round-off (|y_i| ~ 1e-17 against a dual norm of order one) has an
    // arbitrary sign, and reading that sign pins the row to a bound it is
    // nowhere near. The resulting point is still primal feasible and still
    // solves its own reduced KKT system exactly, so BOTH residuals improve and
    // a residual-only accept test takes it -- returning a grossly suboptimal
    // answer labeled solved. So a multiplier must clear a relative threshold
    // before it is read as a bound assignment, and the polished pair is
    // additionally required not to worsen the objective.
    bool try_polish(const sparse_qp_options& o, Scalar cur_rp, Scalar cur_rd, Scalar cur_obj)
    {
        active_lower_.clear();
        active_upper_.clear();
        const Scalar y_inf = m_ > 0 ? y_unscaled_.cwiseAbs().maxCoeff() : Scalar(0);
        const Scalar active_tol =
            Scalar(polish_active_tol_) * std::max(Scalar(1), y_inf);
        for(int i = 0; i < m_; ++i)
        {
            if(y_unscaled_[i] < -active_tol)
                active_lower_.push_back(i);
            else if(y_unscaled_[i] > active_tol)
                active_upper_.push_back(i);
        }
        const int nl = static_cast<int>(active_lower_.size());
        const int nu = static_cast<int>(active_upper_.size());
        const int mact = nl + nu;
        const int dim = n_ + mact;
        const Scalar delta = static_cast<Scalar>(o.delta);

        // Conservative pattern-gated reuse: an unchanged active-set pattern, an
        // unchanged delta and an intact cache (no re-pose, no prior factor
        // failure) mean the reduced KKT is bit-identical to the one already
        // analyzed and factored, so both the symbolic analysis and the numeric
        // factor are reused verbatim. The index vectors are compared by exact
        // element-wise content -- size first, then element by element,
        // short-circuiting on the first mismatch -- never by a hash or compact
        // signature, whose collisions could serve a stale factor for a changed
        // pattern and return a wrong polished answer.
        const bool reuse = polish_reuse_enabled_ && polish_cache_valid_
                           && polish_cache_delta_ == delta
                           && polish_cache_lower_ == active_lower_
                           && polish_cache_upper_ == active_upper_;

        if(!reuse)
        {
            std::fill(act_row_.begin(), act_row_.end(), -1);
            for(int k = 0; k < nl; ++k)
                act_row_[static_cast<std::size_t>(active_lower_[static_cast<std::size_t>(k)])] = k;
            for(int k = 0; k < nu; ++k)
                act_row_[static_cast<std::size_t>(active_upper_[static_cast<std::size_t>(k)])] =
                    nl + k;

            A_act_ = sparse_type(mact, n_);
            triplets_.clear();
            for(int j = 0; j < n_; ++j)
                for(typename sparse_type::InnerIterator it(A_, j); it; ++it)
                {
                    const int r = act_row_[static_cast<std::size_t>(it.row())];
                    if(r >= 0)
                        triplets_.emplace_back(static_cast<index_type>(r),
                                               static_cast<index_type>(j), it.value());
                }
            A_act_.setFromTriplets(triplets_.begin(), triplets_.end());
            A_act_.makeCompressed();

            polish_delta_.assign(static_cast<std::size_t>(mact), delta);
            if(!factor_fresh(polish_kkt_, P_, A_act_, delta, polish_delta_))
            {
                // A failed analysis leaves no reusable factor: drop the cache so
                // the next call cannot mistake the stale member factor for a hit.
                polish_cache_valid_ = false;
                return false;
            }

            K_unreg_ = sparse_type(dim, dim);
            triplets_.clear();
            for(int j = 0; j < n_; ++j)
                for(typename sparse_type::InnerIterator it(P_, j); it; ++it)
                    triplets_.emplace_back(static_cast<index_type>(it.row()),
                                           static_cast<index_type>(j), it.value());
            for(int j = 0; j < n_; ++j)
                for(typename sparse_type::InnerIterator it(A_act_, j); it; ++it)
                {
                    const index_type r = static_cast<index_type>(n_ + it.row());
                    const index_type cj = static_cast<index_type>(j);
                    triplets_.emplace_back(r, cj, it.value());
                    triplets_.emplace_back(cj, r, it.value());
                }
            K_unreg_.setFromTriplets(triplets_.begin(), triplets_.end());
            K_unreg_.makeCompressed();

            // The member factor and unregularized reduced KKT now match this
            // pattern and delta; record them so an unchanged next call reuses
            // them. Set only past a successful factor_fresh.
            polish_cache_lower_ = active_lower_;
            polish_cache_upper_ = active_upper_;
            polish_cache_delta_ = delta;
            polish_cache_valid_ = true;
        }

        rhs_red_.resize(dim);
        rhs_red_.head(n_) = -q_;
        for(int k = 0; k < nl; ++k)
            rhs_red_[n_ + k] = l_[active_lower_[static_cast<std::size_t>(k)]];
        for(int k = 0; k < nu; ++k)
            rhs_red_[n_ + nl + k] = u_[active_upper_[static_cast<std::size_t>(k)]];

        polish_kkt_.solve_into(rhs_red_, sol_red_);
        for(int r = 0; r < static_cast<int>(o.polish_refine_iter); ++r)
        {
            refine_res_.noalias() = K_unreg_ * sol_red_;
            refine_res_ = rhs_red_ - refine_res_;
            polish_kkt_.solve_into(refine_res_, refine_corr_);
            sol_red_ += refine_corr_;
        }
        if(!sol_red_.allFinite())
            return false;

        px_x_ = sol_red_.head(n_);
        if(m_ > 0)
        {
            px_y_.setZero();
            for(int k = 0; k < nl; ++k)
                px_y_[active_lower_[static_cast<std::size_t>(k)]] = sol_red_[n_ + k];
            for(int k = 0; k < nu; ++k)
                px_y_[active_upper_[static_cast<std::size_t>(k)]] = sol_red_[n_ + nl + k];
            Ax_buf_.noalias() = A_ * px_x_;
            mtmp2_ = Ax_buf_.cwiseMax(l_).cwiseMin(u_);
        }
        Scalar rpn = Scalar(0);
        Scalar rdn = Scalar(0);
        Scalar ps = Scalar(0);
        Scalar ds = Scalar(0);
        residual_norms(px_x_, px_y_, mtmp2_, rpn, rdn, ps, ds);
        ntmp_.noalias() = P_ * px_x_;
        const Scalar polished_objective = Scalar(0.5) * px_x_.dot(ntmp_) + q_.dot(px_x_);
        // The iterate is accepted while still slightly infeasible, and an
        // infeasible point can undercut the optimal objective. The multipliers
        // price that violation, so the amount of objective an accepted iterate
        // can hide is of the order of the convergence tolerance scaled by the
        // dual magnitude -- anything beyond that is a broken polish, not
        // rounding.
        const Scalar tol = std::max(static_cast<Scalar>(o.eps_abs),
                                    static_cast<Scalar>(o.eps_rel) * std::abs(cur_obj));
        const Scalar slack = tol * (Scalar(1) + y_inf);
        return detail::polish_is_accepted(rpn, rdn, cur_rp, cur_rd,
                                          polished_objective, cur_obj, slack, m_ > 0);
    }

    int n_{0};
    int m_{0};
    Scalar sigma_{Scalar(1e-6)};
    Scalar c_{Scalar(1)};
    Scalar current_rho_{Scalar(0.1)};
    bool posed_{false};
    bool has_warm_{false};

    sparse_type P_;
    sparse_type A_;
    vector<Scalar> q_;
    vector<Scalar> l_;
    vector<Scalar> u_;

    sparse_type P_scaled_;
    sparse_type A_scaled_;
    vector<Scalar> q_scaled_;
    vector<Scalar> l_scaled_;
    vector<Scalar> u_scaled_;

    vector<Scalar> D_;
    vector<Scalar> E_;
    vector<Scalar> Einv_;
    vector<Scalar> scan_n_;
    vector<Scalar> scan_m_;

    vector<Scalar> rho_vec_;
    vector<Scalar> rho_inv_vec_;
    vector<Scalar> base_mult_;
    std::vector<Scalar> rho_inv_;

    detail::sparse_kkt<Scalar> kkt_;
    vector<Scalar> rhs_kkt_;
    vector<Scalar> sol_kkt_;

    vector<Scalar> x_;
    vector<Scalar> x_tilde_;
    vector<Scalar> z_;
    vector<Scalar> z_tilde_;
    vector<Scalar> z_prev_;
    vector<Scalar> y_;
    vector<Scalar> x_iter_prev_;
    vector<Scalar> y_iter_prev_;

    vector<Scalar> x_unscaled_;
    vector<Scalar> Px_buf_;
    vector<Scalar> Aty_buf_;
    vector<Scalar> rd_;
    vector<Scalar> y_unscaled_;
    vector<Scalar> z_unscaled_;
    vector<Scalar> Ax_buf_;
    vector<Scalar> rp_;

    vector<Scalar> ntmp_;
    vector<Scalar> ntmp2_;
    vector<Scalar> mtmp_;
    vector<Scalar> mtmp2_;

    vector<Scalar> ws_x_;
    vector<Scalar> ws_y_;

    Scalar rp_norm_{0};
    Scalar rd_norm_{0};
    Scalar prim_scale_{0};
    Scalar dual_scale_{0};

    qp_solve_status status_{qp_solve_status::solved};
    int iters_{0};

    detail::sparse_kkt<Scalar> polish_kkt_;
    sparse_type A_act_;
    sparse_type K_unreg_;
    std::vector<Eigen::Triplet<Scalar, index_type>> triplets_;
    std::vector<Scalar> polish_delta_;
    std::vector<int> act_row_;
    std::vector<int> active_lower_;
    std::vector<int> active_upper_;
    // Pattern-gated polish-reuse cache: the previous call's active-set index
    // vectors and delta, plus a validity flag reset by a re-pose or a factor
    // failure. A hit reuses polish_kkt_ and K_unreg_ without re-analyzing.
    std::vector<int> polish_cache_lower_;
    std::vector<int> polish_cache_upper_;
    Scalar polish_cache_delta_{Scalar(0)};
    bool polish_cache_valid_{false};
    bool polish_reuse_enabled_{true};
    vector<Scalar> rhs_red_;
    vector<Scalar> sol_red_;
    vector<Scalar> refine_res_;
    vector<Scalar> refine_corr_;
    vector<Scalar> px_x_;
    vector<Scalar> px_y_;
};

}

#endif
