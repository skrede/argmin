#ifndef HPP_GUARD_ARGMIN_QP_DENSE_ACTIVE_SET_QP_H
#define HPP_GUARD_ARGMIN_QP_DENSE_ACTIVE_SET_QP_H

// Read-only OSQP-form adapter over the dense active-set QP.
//
// Exposes the internal active-set solver through the same public interface as
// the operator-splitting solver -- solve/resolve/solve_into/resolve_into/
// warm_start/reset over the canonical form
//
//   min 0.5 * x^T P x + q^T x  s.t.  l <= A x <= u
//
// by pure input transformation in adapter-owned buffers: each row i of A with
// bounds (l_i, u_i) becomes an equality a_i^T x = l_i when l_i == u_i, an
// inequality a_i^T x >= l_i for a finite lower bound, and a second inequality
// -a_i^T x >= -u_i for a finite upper bound (infinite bounds skipped, matching
// the finite-bound-fold idiom of the LSQ/LSEI QP). The recovered active-set
// multipliers are sign-mapped back onto the OSQP dual y. Not a single line of
// the delegated solver is touched: this header includes it and calls only its
// public solve_into surface.
//
// Inherited contract limits (checked up front and surfaced on the error
// channel, never hidden):
//   - Feasible start. The active-set method assumes a feasible x0 (N&W
//     Alg. 16.1); the delegated solver's equality-manifold feasibility
//     restoration corrects only the equality rows and never reports an
//     infeasible status, so a start that violates l <= A x0 <= u on the mapped
//     rows is rejected here as qp_error::infeasible_start.
//   - Equality-row count <= n. The delegated solver requires the equality
//     block to have at most n rows; more than n equality rows (l_i == u_i) is
//     rejected as qp_error::invalid_problem.
//   - Convexity. A non-positive-definite reduced Hessian is surfaced by the
//     delegated solver and mapped here to qp_error::invalid_problem.
//   - Warm start is working-set-level: the retained x seeds the initial working
//     set, not an ADMM dual trajectory.
//
// Reference: Nocedal & Wright, Numerical Optimization (2e), Section 16.1,
//            Algorithm 16.1, pp. 460-463 (active-set method for convex QP;
//            feasible-start assumption).

#include "argmin/detail/active_set_qp.h"
#include "argmin/options/dense_qp_options.h"
#include "argmin/options/qp_options.h"
#include "argmin/qp/qp_types.h"
#include "argmin/expected.h"
#include "argmin/types.h"

#include <Eigen/Core>

#include <cmath>
#include <vector>
#include <cstdint>
#include <optional>
#include <algorithm>

namespace argmin
{

template <typename Scalar = double, int N = argmin::dynamic_dimension>
class dense_active_set_qp_solver
{
public:
    using result_type = qp_result<Scalar, N>;

    // n fixes the problem dimension; m is the maximum constraint-row capacity.
    // A row can map to at most two inequality rows, so the delegated solver is
    // sized for the worst-case 2m constraint rows; that bound is a runtime
    // capacity, never a compile-time cap (the delegated M axis is dynamic).
    dense_active_set_qp_solver(int n, int m)
        : n_cap_(n)
        , m_cap_(m)
        , held_(n, 2 * m)
        , P_(n, n)
        , q_(n)
        , A_(m, n)
        , l_(m)
        , u_(m)
        , A_eq_(n, n)
        , b_eq_(n)
        , A_in_(2 * m, n)
        , b_in_(2 * m)
        , x0_(n)
        , ax_(m)
        , Px_(n)
        , Aty_(n)
    {
        eq_src_.reserve(static_cast<std::size_t>(n));
        in_src_.reserve(static_cast<std::size_t>(2 * m));
        in_upper_.reserve(static_cast<std::size_t>(2 * m));
        x0_.setZero();
    }

    dense_active_set_qp_solver() = default;

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

    std::optional<qp_error>
    solve_into(const matrix<Scalar, N, N>& P, const vector<Scalar, N>& q,
               const matrix<Scalar, dynamic_dimension, N>& A, const vector<Scalar>& l,
               const vector<Scalar>& u, result_type& out, const dense_qp_options& opts = {})
    {
        if(auto err = validate_full(P, q, A, l, u))
            return err;
        n_ = static_cast<int>(P.rows());
        m_ = static_cast<int>(A.rows());
        P_ = P;
        q_.head(n_) = q;
        if(m_ > 0)
        {
            A_.topRows(m_) = A;
            l_.head(m_) = l;
            u_.head(m_) = u;
        }
        posed_ = true;
        return run(out, opts);
    }

    // Vectors-only resolve: reuses the retained P and A, updates (q, l, u), and
    // warm-starts the active set from the last solution (working-set-level).
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
        return run(out, opts);
    }

    void warm_start(const vector<Scalar, N>& x, const vector<Scalar>&)
    {
        x0_.head(n_cap_) = x.head(n_cap_);
    }

    void reset()
    {
        x0_.setZero();
    }

private:
    static constexpr double eq_tol_ = 1e-9;
    static constexpr double feas_tol_ = 1e-7;

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

    bool is_equality_row(int i) const
    {
        return std::isfinite(l_[i]) && std::isfinite(u_[i])
               && (u_[i] - l_[i]) <= Scalar(eq_tol_);
    }

    std::optional<qp_error> run(result_type& out, const dense_qp_options& opts)
    {
        // Equality-row-count contract: the delegated solver requires at most n
        // equality rows.
        int m_eq = 0;
        for(int i = 0; i < m_; ++i)
            if(is_equality_row(i))
                ++m_eq;
        if(m_eq > n_)
            return qp_error::invalid_problem;

        // Feasible-start contract: reject a start violating l <= A x0 <= u on
        // the mapped rows (the delegated feasibility restoration corrects only
        // the equality manifold).
        if(m_ > 0)
        {
            ax_.head(m_).noalias() = A_.topRows(m_) * x0_.head(n_);
            for(int i = 0; i < m_; ++i)
            {
                const Scalar ai = ax_[i];
                if(is_equality_row(i))
                {
                    if(std::abs(ai - l_[i]) > Scalar(feas_tol_))
                        return qp_error::infeasible_start;
                }
                else
                {
                    if(std::isfinite(l_[i]) && ai < l_[i] - Scalar(feas_tol_))
                        return qp_error::infeasible_start;
                    if(std::isfinite(u_[i]) && ai > u_[i] + Scalar(feas_tol_))
                        return qp_error::infeasible_start;
                }
            }
        }

        split_rows(m_eq);

        argmin::qp_options inner;
        inner.max_iterations = opts.max_iterations;
        inner.tolerance = static_cast<double>(opts.eps_abs);

        held_.solve_into(P_, q_.head(n_), A_eq_, b_eq_, A_in_, b_in_, x0_.head(n_),
                         inner, held_res_);

        if(held_res_.status == detail::qp_status::indefinite_hessian)
            return qp_error::invalid_problem;

        scatter_result(out);
        return std::nullopt;
    }

    // Two-pass OSQP-row split into adapter-owned buffers: count then scatter,
    // tracking each mapped row's originating OSQP index (and, for inequality
    // rows, whether it is the upper-bound reflection) so the dual sign-map can
    // scatter multipliers back onto y.
    void split_rows(int m_eq)
    {
        int m_in = 0;
        for(int i = 0; i < m_; ++i)
        {
            if(is_equality_row(i))
                continue;
            if(std::isfinite(l_[i]))
                ++m_in;
            if(std::isfinite(u_[i]))
                ++m_in;
        }

        A_eq_.resize(m_eq, n_);
        b_eq_.resize(m_eq);
        A_in_.resize(m_in, n_);
        b_in_.resize(m_in);
        eq_src_.resize(static_cast<std::size_t>(m_eq));
        in_src_.resize(static_cast<std::size_t>(m_in));
        in_upper_.resize(static_cast<std::size_t>(m_in));

        int re = 0;
        int ri = 0;
        for(int i = 0; i < m_; ++i)
        {
            if(is_equality_row(i))
            {
                A_eq_.row(re) = A_.row(i).head(n_);
                b_eq_[re] = l_[i];
                eq_src_[static_cast<std::size_t>(re)] = i;
                ++re;
                continue;
            }
            if(std::isfinite(l_[i]))
            {
                A_in_.row(ri) = A_.row(i).head(n_);
                b_in_[ri] = l_[i];
                in_src_[static_cast<std::size_t>(ri)] = i;
                in_upper_[static_cast<std::size_t>(ri)] = 0;
                ++ri;
            }
            if(std::isfinite(u_[i]))
            {
                A_in_.row(ri) = -A_.row(i).head(n_);
                b_in_[ri] = -u_[i];
                in_src_[static_cast<std::size_t>(ri)] = i;
                in_upper_[static_cast<std::size_t>(ri)] = 1;
                ++ri;
            }
        }
    }

    // Sign-map the recovered active-set multipliers onto the OSQP dual y. The
    // delegated stationarity is G x + d = A_W^T lambda with lambda >= 0 on
    // active inequalities; OSQP wants P x + q + A^T y = 0 with y <= 0 lower-
    // active, y >= 0 upper-active. Hence per original row y_i = mu_i(upper) -
    // lambda_i(lower), and y_i = -lambda_i on equality rows. The delegated
    // lambda block orders equalities first, then inequalities.
    void scatter_result(result_type& out)
    {
        out.x = held_res_.x.head(n_);
        const int m_eq = static_cast<int>(eq_src_.size());
        const int m_in = static_cast<int>(in_src_.size());

        out.y.resize(m_);
        if(m_ > 0)
            out.y.setZero();
        for(int k = 0; k < m_eq; ++k)
            out.y[eq_src_[static_cast<std::size_t>(k)]] = -held_res_.lambda[k];
        for(int k = 0; k < m_in; ++k)
        {
            const Scalar lam = held_res_.lambda[m_eq + k];
            const int src = in_src_[static_cast<std::size_t>(k)];
            if(in_upper_[static_cast<std::size_t>(k)])
                out.y[src] += lam;
            else
                out.y[src] -= lam;
        }

        switch(held_res_.status)
        {
        case detail::qp_status::optimal:
            out.status = qp_solve_status::solved;
            break;
        case detail::qp_status::max_iterations:
            out.status = qp_solve_status::max_iterations;
            break;
        case detail::qp_status::infeasible:
            out.status = qp_solve_status::primal_infeasible;
            break;
        case detail::qp_status::indefinite_hessian:
            out.status = qp_solve_status::primal_infeasible;
            break;
        }
        out.iterations = held_res_.iterations;
        out.polished = false;

        Px_.head(n_).noalias() = P_ * out.x.head(n_);
        Scalar rp = Scalar(0);
        if(m_ > 0)
        {
            ax_.head(m_).noalias() = A_.topRows(m_) * out.x.head(n_);
            for(int i = 0; i < m_; ++i)
            {
                if(ax_[i] < l_[i])
                    rp = std::max(rp, l_[i] - ax_[i]);
                if(ax_[i] > u_[i])
                    rp = std::max(rp, ax_[i] - u_[i]);
            }
        }
        Aty_.head(n_).setZero();
        if(m_ > 0)
            Aty_.head(n_).noalias() = A_.topRows(m_).transpose() * out.y;
        const Scalar rd =
            (Px_.head(n_) + q_.head(n_) + Aty_.head(n_)).cwiseAbs().maxCoeff();

        out.primal_residual = rp;
        out.dual_residual = rd;
        out.objective_value =
            Scalar(0.5) * out.x.head(n_).dot(Px_.head(n_)) + q_.head(n_).dot(out.x.head(n_));

        // The last solution seeds the next resolve's working set.
        x0_.head(n_) = out.x.head(n_);
    }

    int n_cap_{0};
    int m_cap_{0};
    int n_{0};
    int m_{0};
    bool posed_{false};

    detail::active_set_qp_solver<Scalar, N, argmin::dynamic_dimension> held_;
    typename detail::active_set_qp_solver<Scalar, N, argmin::dynamic_dimension>::result_type
        held_res_;

    matrix<Scalar, N, N> P_;
    vector<Scalar, N> q_;
    matrix<Scalar, dynamic_dimension, N> A_;
    vector<Scalar> l_;
    vector<Scalar> u_;

    matrix<Scalar, dynamic_dimension, N> A_eq_;
    vector<Scalar> b_eq_;
    matrix<Scalar, dynamic_dimension, N> A_in_;
    vector<Scalar> b_in_;

    vector<Scalar, N> x0_;
    vector<Scalar> ax_;
    vector<Scalar, N> Px_;
    vector<Scalar, N> Aty_;

    std::vector<int> eq_src_;
    std::vector<int> in_src_;
    std::vector<std::uint8_t> in_upper_;
};

}

#endif
