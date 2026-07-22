#ifndef HPP_GUARD_ARGMIN_QP_DENSE_ADMM_QP_H
#define HPP_GUARD_ARGMIN_QP_DENSE_ADMM_QP_H

// Dense operator-splitting QP solver (OSQP-class ADMM), templated over N.
//
// Solves: min 0.5 * x^T P x + q^T x  s.t.  l <= A x <= u
//
// Each ADMM iteration solves the condensed SPD "indirect method" system
//   K x_tilde = sigma*x - q + A^T (diag(rho) z - y),
//   K = P + sigma*I + A^T diag(rho) A,
// factored once with Eigen::LLT on a constant n x n shape; K is refactored
// only when an accepted adaptive-rho update changes rho. The problem data is
// Ruiz-equilibrated (diagonal D, E and a cost scalar c) at pose time; those
// factors are frozen and reused by resolve(), which is the vectors-only
// (q, l, u) control-step hot path. Convergence and the primal/dual
// infeasibility certificates are evaluated on UNSCALED residuals. The
// adaptive-rho schedule triggers on a fixed iteration interval, never on a
// measured duration, so the solver reads no clock. A delta-regularized
// reduced-KKT polish (active set from dual signs, Eigen::PartialPivLU with
// iterative refinement, accept-if-better) lifts the accepted iterate toward
// ~1e-10.
//
// Allocation contract: every buffer and decomposition object is a member sized
// at construction; no allocation occurs inside resolve() after construction at
// fixed N (the value-returning solve()/resolve() overloads allocate the result
// vectors once; the RT loop uses solve_into()/resolve_into()). The condensed K
// is a fixed n x n member, so a fixed-N instantiation is bounded by Eigen's
// EIGEN_STACK_ALLOCATION_LIMIT (default 131072 bytes): N <= 128 for double.
//
// Reference: Stellato, Banjac, Goulart, Bemporad, Boyd (2020), "OSQP: An
//            operator splitting solver for quadratic programs," Math. Prog.
//            Comp. 12:637-672 -- Algorithm 1 (Section 3), Section 5.1 (Ruiz
//            equilibration), Section 5.2 (adaptive rho), Section 5.3 (polish);
//            osqp.org/docs/solver (KKT systems, termination, infeasibility
//            certificates). The equality-row rho boost 1e3 and the clamp range
//            [1e-6, 1e6] follow Section 5.2 and the reference implementation;
//            the solved-inaccurate report uses a relaxed tolerance at factor
//            K = 10, OSQP's convention on the iteration cap.

#include "argmin/qp/qp_types.h"
#include "argmin/options/dense_qp_options.h"
#include "argmin/expected.h"
#include "argmin/types.h"

#include <Eigen/LU>
#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <cmath>
#include <limits>
#include <vector>
#include <optional>
#include <algorithm>

namespace argmin
{

template <typename Scalar = double, int N = argmin::dynamic_dimension>
class dense_admm_qp_solver
{
public:
    using result_type = qp_result<Scalar, N>;

    dense_admm_qp_solver(int n, int m)
        : n_cap_(n)
        , m_cap_(m)
        , P_(n, n)
        , q_(n)
        , A_(m, n)
        , l_(m)
        , u_(m)
        , P_scaled_(n, n)
        , q_scaled_(n)
        , A_scaled_(m, n)
        , l_scaled_(m)
        , u_scaled_(m)
        , D_(n)
        , E_(m)
        , Einv_(m)
        , scan_n_(n)
        , scan_m_(m)
        , rho_vec_(m)
        , rho_inv_vec_(m)
        , base_mult_(m)
        , K_(n, n)
        , llt_(n)
        , scratch_mn_(m, n)
        , x_(n)
        , x_tilde_(n)
        , rhs_(n)
        , z_(m)
        , z_tilde_(m)
        , z_prev_(m)
        , y_(m)
        , x_iter_prev_(n)
        , y_iter_prev_(m)
        , x_unscaled_(n)
        , Px_buf_(n)
        , Aty_buf_(n)
        , rd_(n)
        , y_unscaled_(m)
        , z_unscaled_(m)
        , Ax_buf_(m)
        , rp_(m)
        , ntmp_(n)
        , ntmp2_(n)
        , mtmp_(m)
        , mtmp2_(m)
        , ws_x_(n)
        , ws_y_(m)
        , K_red_(n + m, n + m)
        , K_red_reg_(n + m, n + m)
        , lu_(n + m)
        , rhs_red_(n + m)
        , sol_red_(n + m)
        , refine_res_(n + m)
        , refine_corr_(n + m)
        , px_x_(n)
        , px_y_(m)
    {
        active_lower_.reserve(static_cast<std::size_t>(m));
        active_upper_.reserve(static_cast<std::size_t>(m));
    }

    dense_admm_qp_solver() = default;

    expected<result_type, qp_error>
    solve(const matrix<Scalar, N, N>& P, const vector<Scalar, N>& q,
          const matrix<Scalar, dynamic_dimension, N>& A, const vector<Scalar>& l,
          const vector<Scalar>& u, const dense_qp_options& opts = {})
    {
        result_type out;
        if(auto err = solve_into(P, q, A, l, u, out, opts))
            return argmin::unexpected<qp_error>{*err};
        return out;
    }

    expected<result_type, qp_error>
    resolve(const vector<Scalar, N>& q, const vector<Scalar>& l,
            const vector<Scalar>& u, const dense_qp_options& opts = {})
    {
        result_type out;
        if(auto err = resolve_into(q, l, u, out, opts))
            return argmin::unexpected<qp_error>{*err};
        return out;
    }

    // Fill-into sibling of solve(): poses the problem (copy + Ruiz scaling +
    // factor K) and runs warm-started ADMM into a caller-owned result. Returns
    // an error only on an argument/precondition violation; solver outcomes
    // travel in out.status.
    std::optional<qp_error>
    solve_into(const matrix<Scalar, N, N>& P, const vector<Scalar, N>& q,
               const matrix<Scalar, dynamic_dimension, N>& A, const vector<Scalar>& l,
               const vector<Scalar>& u, result_type& out, const dense_qp_options& opts = {})
    {
        if(auto err = validate_full(P, q, A, l, u))
            return err;
        pose(P, q, A, l, u, opts);
        run_admm(opts);
        finalize(opts, out);
        return std::nullopt;
    }

    // Vectors-only hot path: updates (q, l, u), reuses the frozen Ruiz factors
    // and the K factorization, warm-starts from the retained iterate. Allocates
    // nothing after construction at fixed N.
    std::optional<qp_error>
    resolve_into(const vector<Scalar, N>& q, const vector<Scalar>& l,
                 const vector<Scalar>& u, result_type& out, const dense_qp_options& opts = {})
    {
        if(!posed_)
            return qp_error::invalid_problem;
        if(q.size() != n_ || static_cast<int>(l.size()) != m_
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
        q_.head(n_) = q;
        if(m_ > 0)
        {
            l_.head(m_) = l;
            u_.head(m_) = u;
        }
        rescale_vectors();
        seed_iterates(opts);
        run_admm(opts);
        finalize(opts, out);
        return std::nullopt;
    }

    void warm_start(const vector<Scalar, N>& x, const vector<Scalar>& y)
    {
        ws_x_.head(n_cap_) = x.head(n_cap_);
        if(y.size() > 0)
            ws_y_.head(static_cast<int>(y.size())) = y;
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
    const vector<Scalar, N>& scaling_primal() const { return D_; }
    const vector<Scalar>& scaling_dual() const { return E_; }

private:
    static constexpr double rho_min_ = 1e-6;
    static constexpr double rho_max_ = 1e6;
    static constexpr double eq_boost_ = 1e3;
    static constexpr double eq_tol_ = 1e-9;
    static constexpr int inaccurate_factor_ = 10;

    std::optional<qp_error>
    validate_full(const matrix<Scalar, N, N>& P, const vector<Scalar, N>& q,
                  const matrix<Scalar, dynamic_dimension, N>& A, const vector<Scalar>& l,
                  const vector<Scalar>& u) const
    {
        const int n = static_cast<int>(P.rows());
        const int m = static_cast<int>(A.rows());
        if(P.cols() != n || A.cols() != n || q.size() != n
           || static_cast<int>(l.size()) != m || static_cast<int>(u.size()) != m)
            return qp_error::dimension_mismatch;
        if(n != n_cap_ || m > m_cap_)
            return qp_error::capacity_exceeded;
        if(!P.allFinite() || !q.allFinite() || !A.allFinite())
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

    void pose(const matrix<Scalar, N, N>& P, const vector<Scalar, N>& q,
              const matrix<Scalar, dynamic_dimension, N>& A, const vector<Scalar>& l,
              const vector<Scalar>& u, const dense_qp_options& opts)
    {
        n_ = static_cast<int>(P.rows());
        m_ = static_cast<int>(A.rows());
        sigma_ = static_cast<Scalar>(opts.sigma);
        P_ = P;
        q_.head(n_) = q;
        if(m_ > 0)
        {
            A_.topRows(m_) = A;
            l_.head(m_) = l;
            u_.head(m_) = u;
        }
        ruiz(opts);
        if(m_ > 0)
            Einv_.head(m_) = E_.head(m_).cwiseInverse();
        init_rho(opts);
        build_and_factor_K();
        seed_iterates(opts);
        posed_ = true;
    }

    // Modified Ruiz equilibration on the stacked KKT form [P A^T; A 0] plus a
    // cost scaling c. Reference: Stellato et al. 2020, Section 5.1.
    void ruiz(const dense_qp_options& opts)
    {
        P_scaled_ = P_;
        q_scaled_.head(n_) = q_.head(n_);
        if(m_ > 0)
            A_scaled_.topRows(m_) = A_.topRows(m_);
        D_.head(n_).setOnes();
        if(m_ > 0)
            E_.head(m_).setOnes();
        c_ = Scalar(1);

        for(int it = 0; it < static_cast<int>(opts.scaling); ++it)
        {
            for(int j = 0; j < n_; ++j)
            {
                Scalar d = P_scaled_.col(j).cwiseAbs().maxCoeff();
                if(m_ > 0)
                    d = std::max(d, A_scaled_.col(j).head(m_).cwiseAbs().maxCoeff());
                scan_n_[j] = d > Scalar(0) ? Scalar(1) / std::sqrt(d) : Scalar(1);
            }
            for(int i = 0; i < m_; ++i)
            {
                const Scalar e = A_scaled_.row(i).cwiseAbs().maxCoeff();
                scan_m_[i] = e > Scalar(0) ? Scalar(1) / std::sqrt(e) : Scalar(1);
            }

            P_scaled_.array().rowwise() *= scan_n_.head(n_).transpose().array();
            P_scaled_.array().colwise() *= scan_n_.head(n_).array();
            q_scaled_.head(n_).array() *= scan_n_.head(n_).array();
            if(m_ > 0)
            {
                A_scaled_.topRows(m_).array().rowwise() *= scan_n_.head(n_).transpose().array();
                A_scaled_.topRows(m_).array().colwise() *= scan_m_.head(m_).array();
            }
            D_.head(n_).array() *= scan_n_.head(n_).array();
            if(m_ > 0)
                E_.head(m_).array() *= scan_m_.head(m_).array();

            Scalar mean_pcol = Scalar(0);
            for(int j = 0; j < n_; ++j)
                mean_pcol += P_scaled_.col(j).cwiseAbs().maxCoeff();
            mean_pcol /= static_cast<Scalar>(n_);
            const Scalar qn = q_scaled_.head(n_).cwiseAbs().maxCoeff();
            const Scalar denom = std::max(mean_pcol, qn);
            const Scalar gamma = denom > Scalar(0) ? Scalar(1) / denom : Scalar(1);
            P_scaled_ *= gamma;
            q_scaled_.head(n_) *= gamma;
            c_ *= gamma;
        }

        if(m_ > 0)
        {
            l_scaled_.head(m_) = E_.head(m_).cwiseProduct(l_.head(m_));
            u_scaled_.head(m_) = E_.head(m_).cwiseProduct(u_.head(m_));
        }
    }

    void rescale_vectors()
    {
        q_scaled_.head(n_) = c_ * D_.head(n_).cwiseProduct(q_.head(n_));
        if(m_ > 0)
        {
            l_scaled_.head(m_) = E_.head(m_).cwiseProduct(l_.head(m_));
            u_scaled_.head(m_) = E_.head(m_).cwiseProduct(u_.head(m_));
        }
    }

    void init_rho(const dense_qp_options& opts)
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
            rho_vec_[i] = std::clamp(current_rho_ * base_mult_[i], Scalar(rho_min_), Scalar(rho_max_));
        if(m_ > 0)
            rho_inv_vec_.head(m_) = rho_vec_.head(m_).cwiseInverse();
    }

    void build_and_factor_K()
    {
        K_ = P_scaled_;
        K_.diagonal().array() += sigma_;
        if(m_ > 0)
        {
            scratch_mn_.topRows(m_).noalias() =
                rho_vec_.head(m_).asDiagonal() * A_scaled_.topRows(m_);
            K_.noalias() += A_scaled_.topRows(m_).transpose() * scratch_mn_.topRows(m_);
        }
        llt_.compute(K_);
    }

    void seed_iterates(const dense_qp_options& opts)
    {
        if(opts.warm_start && has_warm_)
        {
            x_.head(n_) = ws_x_.head(n_).cwiseQuotient(D_.head(n_));
            if(m_ > 0)
            {
                y_.head(m_) = c_ * ws_y_.head(m_).cwiseQuotient(E_.head(m_));
                // The A*x product is evaluated into a scratch member first: a
                // matrix-vector product nested inside a coefficient-wise
                // cwiseMax/cwiseMin chain forces Eigen to heap-allocate the
                // intermediate, which would break the resolve() allocation
                // contract at fixed N.
                mtmp_.head(m_).noalias() = A_scaled_.topRows(m_) * x_.head(n_);
                z_.head(m_) = mtmp_.head(m_)
                                  .cwiseMax(l_scaled_.head(m_))
                                  .cwiseMin(u_scaled_.head(m_));
            }
        }
        else
        {
            x_.head(n_).setZero();
            if(m_ > 0)
            {
                y_.head(m_).setZero();
                z_.head(m_).setZero();
            }
        }
    }

    // Algorithm 1. All products land in members with .noalias(); the z-step is
    // the box projection cwiseMax(l).cwiseMin(u).
    void admm_step(const dense_qp_options& opts)
    {
        const Scalar alpha = static_cast<Scalar>(opts.alpha);
        x_iter_prev_.head(n_) = x_.head(n_);
        if(m_ > 0)
            y_iter_prev_.head(m_) = y_.head(m_);

        rhs_.head(n_) = sigma_ * x_.head(n_) - q_scaled_.head(n_);
        if(m_ > 0)
        {
            mtmp_.head(m_) = rho_vec_.head(m_).cwiseProduct(z_.head(m_)) - y_.head(m_);
            rhs_.head(n_).noalias() += A_scaled_.topRows(m_).transpose() * mtmp_.head(m_);
        }
        x_tilde_.head(n_) = llt_.solve(rhs_.head(n_));
        x_.head(n_) = alpha * x_tilde_.head(n_) + (Scalar(1) - alpha) * x_.head(n_);
        if(m_ > 0)
        {
            z_tilde_.head(m_).noalias() = A_scaled_.topRows(m_) * x_tilde_.head(n_);
            z_prev_.head(m_) = z_.head(m_);
            z_.head(m_) = (alpha * z_tilde_.head(m_) + (Scalar(1) - alpha) * z_prev_.head(m_)
                           + rho_inv_vec_.head(m_).cwiseProduct(y_.head(m_)))
                              .cwiseMax(l_scaled_.head(m_))
                              .cwiseMin(u_scaled_.head(m_));
            y_.head(m_) += rho_vec_.head(m_).cwiseProduct(
                alpha * z_tilde_.head(m_) + (Scalar(1) - alpha) * z_prev_.head(m_) - z_.head(m_));
        }
    }

    void run_admm(const dense_qp_options& opts)
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
        residual_norms(x_unscaled_, y_unscaled_, z_unscaled_, rp_norm_, rd_norm_,
                       prim_scale_, dual_scale_);
    }

    void compute_unscaled_from_scaled()
    {
        x_unscaled_.head(n_) = D_.head(n_).cwiseProduct(x_.head(n_));
        if(m_ > 0)
        {
            y_unscaled_.head(m_) = E_.head(m_).cwiseProduct(y_.head(m_)) / c_;
            z_unscaled_.head(m_) = Einv_.head(m_).cwiseProduct(z_.head(m_));
        }
    }

    void residual_norms(const vector<Scalar, N>& xu, const vector<Scalar>& yu,
                        const vector<Scalar>& zu, Scalar& rpn, Scalar& rdn, Scalar& ps,
                        Scalar& ds)
    {
        Px_buf_.head(n_).noalias() = P_ * xu.head(n_);
        rd_.head(n_) = Px_buf_.head(n_) + q_.head(n_);
        Scalar ax_inf = Scalar(0);
        Scalar z_inf = Scalar(0);
        Scalar aty_inf = Scalar(0);
        rpn = Scalar(0);
        if(m_ > 0)
        {
            Ax_buf_.head(m_).noalias() = A_.topRows(m_) * xu.head(n_);
            Aty_buf_.head(n_).noalias() = A_.topRows(m_).transpose() * yu.head(m_);
            rd_.head(n_) += Aty_buf_.head(n_);
            rp_.head(m_) = Ax_buf_.head(m_) - zu.head(m_);
            rpn = rp_.head(m_).cwiseAbs().maxCoeff();
            ax_inf = Ax_buf_.head(m_).cwiseAbs().maxCoeff();
            z_inf = zu.head(m_).cwiseAbs().maxCoeff();
            aty_inf = Aty_buf_.head(n_).cwiseAbs().maxCoeff();
        }
        rdn = rd_.head(n_).cwiseAbs().maxCoeff();
        const Scalar q_inf = q_.head(n_).cwiseAbs().maxCoeff();
        const Scalar px_inf = Px_buf_.head(n_).cwiseAbs().maxCoeff();
        ps = std::max(ax_inf, z_inf);
        ds = std::max(px_inf, std::max(aty_inf, q_inf));
    }

    bool converged(const dense_qp_options& o) const
    {
        const Scalar ep = static_cast<Scalar>(o.eps_abs);
        const Scalar er = static_cast<Scalar>(o.eps_rel);
        const bool p = m_ == 0 || rp_norm_ <= ep + er * prim_scale_;
        const bool d = rd_norm_ <= ep + er * dual_scale_;
        return p && d;
    }

    bool relaxed_converged(const dense_qp_options& o) const
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
    bool check_primal_infeasible(const dense_qp_options& o)
    {
        if(m_ == 0)
            return false;
        mtmp_.head(m_) = E_.head(m_).cwiseProduct(y_.head(m_) - y_iter_prev_.head(m_)) / c_;
        const Scalar dy_inf = mtmp_.head(m_).cwiseAbs().maxCoeff();
        if(dy_inf <= Scalar(0))
            return false;
        ntmp_.head(n_).noalias() = A_.topRows(m_).transpose() * mtmp_.head(m_);
        if(ntmp_.head(n_).cwiseAbs().maxCoeff() > static_cast<Scalar>(o.eps_prim_inf) * dy_inf)
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
        return sup < Scalar(0);
    }

    bool check_dual_infeasible(const dense_qp_options& o)
    {
        ntmp_.head(n_) = D_.head(n_).cwiseProduct(x_.head(n_) - x_iter_prev_.head(n_));
        const Scalar dx_inf = ntmp_.head(n_).cwiseAbs().maxCoeff();
        if(dx_inf <= Scalar(0))
            return false;
        ntmp2_.head(n_).noalias() = P_ * ntmp_.head(n_);
        if(ntmp2_.head(n_).cwiseAbs().maxCoeff() > static_cast<Scalar>(o.eps_dual_inf) * dx_inf)
            return false;
        if(q_.head(n_).dot(ntmp_.head(n_)) >= Scalar(0))
            return false;
        if(m_ > 0)
        {
            mtmp_.head(m_).noalias() = A_.topRows(m_) * ntmp_.head(n_);
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

    void maybe_update_rho(const dense_qp_options& o)
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
            build_and_factor_K();
        }
    }

    void finalize(const dense_qp_options& o, result_type& out)
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

        bool polished = false;
        if(o.polish
           && (status_ == qp_solve_status::solved || status_ == qp_solve_status::solved_inaccurate))
        {
            if(try_polish(o, rpn, rdn))
            {
                polished = true;
                x_unscaled_.head(n_) = px_x_.head(n_);
                if(m_ > 0)
                {
                    y_unscaled_.head(m_) = px_y_.head(m_);
                    // Scratch the A*x product before the box projection; see the
                    // note in seed_iterates on the cwise-chain allocation. Ax_buf_
                    // is overwritten by the residual_norms call below.
                    Ax_buf_.head(m_).noalias() = A_.topRows(m_) * x_unscaled_.head(n_);
                    z_unscaled_.head(m_) = Ax_buf_.head(m_)
                                               .cwiseMax(l_.head(m_))
                                               .cwiseMin(u_.head(m_));
                }
                residual_norms(x_unscaled_, y_unscaled_, z_unscaled_, rpn, rdn, ps, ds);
            }
        }

        out.x = x_unscaled_.head(n_);
        if(m_ > 0)
            out.y = y_unscaled_.head(m_);
        else
            out.y.resize(0);
        out.status = status_;
        out.iterations = iters_;
        out.polished = polished;
        out.primal_residual = m_ > 0 ? rpn : Scalar(0);
        out.dual_residual = rdn;
        ntmp_.head(n_).noalias() = P_ * x_unscaled_.head(n_);
        out.objective_value = Scalar(0.5) * x_unscaled_.head(n_).dot(ntmp_.head(n_))
                              + q_.head(n_).dot(x_unscaled_.head(n_));

        if(o.warm_start)
        {
            ws_x_.head(n_) = x_unscaled_.head(n_);
            if(m_ > 0)
                ws_y_.head(m_) = y_unscaled_.head(m_);
            has_warm_ = true;
        }
    }

    // Delta-regularized reduced-KKT polish (Stellato et al. 2020, Section 5.3).
    // The active set comes from the unscaled dual signs; the reduced system is
    // assembled into the leading block of a constant (n + m_cap) padded shape
    // with an identity trailing block, so PartialPivLU always factors the same
    // size (no reallocation). Contingency: partial pivoting CAN select a pivot
    // from the identity trailing block; a padded-vs-unpadded equivalence test
    // guards this, and if it ever fails the fallback is a fixed shape at exactly
    // (n + m) with every row present and the inactive rows delta-slacked.
    bool try_polish(const dense_qp_options& o, Scalar cur_rp, Scalar cur_rd)
    {
        active_lower_.clear();
        active_upper_.clear();
        for(int i = 0; i < m_; ++i)
        {
            if(y_unscaled_[i] < Scalar(0))
                active_lower_.push_back(i);
            else if(y_unscaled_[i] > Scalar(0))
                active_upper_.push_back(i);
        }
        const int nl = static_cast<int>(active_lower_.size());
        const int nu = static_cast<int>(active_upper_.size());
        const int mact = nl + nu;
        const int dim = n_ + mact;
        const int pad = n_ + m_cap_;
        const Scalar delta = static_cast<Scalar>(o.delta);

        K_red_.topLeftCorner(pad, pad).setZero();
        K_red_.topLeftCorner(n_, n_) = P_;
        for(int k = 0; k < nl; ++k)
        {
            const int i = active_lower_[static_cast<std::size_t>(k)];
            K_red_.block(n_ + k, 0, 1, n_) = A_.row(i).head(n_);
            K_red_.block(0, n_ + k, n_, 1) = A_.row(i).head(n_).transpose();
        }
        for(int k = 0; k < nu; ++k)
        {
            const int i = active_upper_[static_cast<std::size_t>(k)];
            K_red_.block(n_ + nl + k, 0, 1, n_) = A_.row(i).head(n_);
            K_red_.block(0, n_ + nl + k, n_, 1) = A_.row(i).head(n_).transpose();
        }
        for(int j = dim; j < pad; ++j)
            K_red_(j, j) = Scalar(1);

        rhs_red_.head(pad).setZero();
        rhs_red_.head(n_) = -q_.head(n_);
        for(int k = 0; k < nl; ++k)
            rhs_red_[n_ + k] = l_[active_lower_[static_cast<std::size_t>(k)]];
        for(int k = 0; k < nu; ++k)
            rhs_red_[n_ + nl + k] = u_[active_upper_[static_cast<std::size_t>(k)]];

        K_red_reg_.topLeftCorner(pad, pad) = K_red_.topLeftCorner(pad, pad);
        for(int j = 0; j < n_; ++j)
            K_red_reg_(j, j) += delta;
        for(int j = n_; j < dim; ++j)
            K_red_reg_(j, j) -= delta;
        lu_.compute(K_red_reg_.topLeftCorner(pad, pad));

        sol_red_.head(pad) = lu_.solve(rhs_red_.head(pad));
        for(int r = 0; r < static_cast<int>(o.polish_refine_iter); ++r)
        {
            refine_res_.head(pad).noalias() =
                rhs_red_.head(pad) - K_red_.topLeftCorner(pad, pad) * sol_red_.head(pad);
            refine_corr_.head(pad) = lu_.solve(refine_res_.head(pad));
            sol_red_.head(pad) += refine_corr_.head(pad);
        }

        px_x_.head(n_) = sol_red_.head(n_);
        if(m_ > 0)
        {
            px_y_.head(m_).setZero();
            for(int k = 0; k < nl; ++k)
                px_y_[active_lower_[static_cast<std::size_t>(k)]] = sol_red_[n_ + k];
            for(int k = 0; k < nu; ++k)
                px_y_[active_upper_[static_cast<std::size_t>(k)]] = sol_red_[n_ + nl + k];
        }

        if(m_ > 0)
        {
            // Scratch the A*x product before the box projection; see the note in
            // seed_iterates on the cwise-chain allocation. Ax_buf_ is overwritten
            // by the residual_norms call below.
            Ax_buf_.head(m_).noalias() = A_.topRows(m_) * px_x_.head(n_);
            mtmp2_.head(m_) = Ax_buf_.head(m_)
                                  .cwiseMax(l_.head(m_))
                                  .cwiseMin(u_.head(m_));
        }
        Scalar rpn = Scalar(0);
        Scalar rdn = Scalar(0);
        Scalar ps = Scalar(0);
        Scalar ds = Scalar(0);
        residual_norms(px_x_, px_y_, mtmp2_, rpn, rdn, ps, ds);
        return rdn < cur_rd && (m_ == 0 || rpn < cur_rp);
    }

    int n_cap_{0};
    int m_cap_{0};
    int n_{0};
    int m_{0};
    Scalar sigma_{Scalar(1e-6)};
    Scalar c_{Scalar(1)};
    Scalar current_rho_{Scalar(0.1)};
    bool posed_{false};
    bool has_warm_{false};

    matrix<Scalar, N, N> P_;
    vector<Scalar, N> q_;
    matrix<Scalar, dynamic_dimension, N> A_;
    vector<Scalar> l_;
    vector<Scalar> u_;

    matrix<Scalar, N, N> P_scaled_;
    vector<Scalar, N> q_scaled_;
    matrix<Scalar, dynamic_dimension, N> A_scaled_;
    vector<Scalar> l_scaled_;
    vector<Scalar> u_scaled_;

    vector<Scalar, N> D_;
    vector<Scalar> E_;
    vector<Scalar> Einv_;
    vector<Scalar, N> scan_n_;
    vector<Scalar> scan_m_;

    vector<Scalar> rho_vec_;
    vector<Scalar> rho_inv_vec_;
    vector<Scalar> base_mult_;

    // Condensed KKT is a fixed n x n member: the EIGEN_STACK_ALLOCATION_LIMIT
    // ceiling caps a fixed-N instantiation at N <= 128 for double.
    matrix<Scalar, N, N> K_;
    Eigen::LLT<matrix<Scalar, N, N>> llt_;
    matrix<Scalar, dynamic_dimension, N> scratch_mn_;

    vector<Scalar, N> x_;
    vector<Scalar, N> x_tilde_;
    vector<Scalar, N> rhs_;
    vector<Scalar> z_;
    vector<Scalar> z_tilde_;
    vector<Scalar> z_prev_;
    vector<Scalar> y_;
    vector<Scalar, N> x_iter_prev_;
    vector<Scalar> y_iter_prev_;

    vector<Scalar, N> x_unscaled_;
    vector<Scalar, N> Px_buf_;
    vector<Scalar, N> Aty_buf_;
    vector<Scalar, N> rd_;
    vector<Scalar> y_unscaled_;
    vector<Scalar> z_unscaled_;
    vector<Scalar> Ax_buf_;
    vector<Scalar> rp_;

    vector<Scalar, N> ntmp_;
    vector<Scalar, N> ntmp2_;
    vector<Scalar> mtmp_;
    vector<Scalar> mtmp2_;

    vector<Scalar, N> ws_x_;
    vector<Scalar> ws_y_;

    Scalar rp_norm_{0};
    Scalar rd_norm_{0};
    Scalar prim_scale_{0};
    Scalar dual_scale_{0};

    qp_solve_status status_{qp_solve_status::solved};
    int iters_{0};

    matrix<Scalar> K_red_;
    matrix<Scalar> K_red_reg_;
    Eigen::PartialPivLU<matrix<Scalar>> lu_;
    vector<Scalar> rhs_red_;
    vector<Scalar> sol_red_;
    vector<Scalar> refine_res_;
    vector<Scalar> refine_corr_;
    std::vector<int> active_lower_;
    std::vector<int> active_upper_;
    vector<Scalar, N> px_x_;
    vector<Scalar> px_y_;
};

}

#endif
