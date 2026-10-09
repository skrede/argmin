#ifndef HPP_GUARD_ARGMIN_SOLVER_PROJECTED_GN_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_PROJECTED_GN_POLICY_H

// Projected Gauss-Newton solver policy with active-set box projection.
//
// Solves bound-constrained nonlinear least-squares problems by identifying
// free vs active variables at each iteration, then solving the reduced
// Gauss-Newton system (J^T J + lambda*D) h = -J^T r on the free subspace.
// Two globalization modes: Nielsen (1999) adaptive lambda damping (default)
// and dogleg trust-region interpolation (N&W Algorithm 4.1).
//
// The active-set identification follows N&W Section 16.6: a variable is
// active when it sits at a bound and the gradient points into that bound.
// Trial points are projected onto bounds before evaluation (N&W Section 16.7).
//
// The problem is bound by a stored pointer and dispatched at compile time:
// residuals/value come straight off the least-squares interface, and the
// Jacobian is analytic when the problem exposes jacobian() and a central
// finite-difference fallback otherwise (both selected with if constexpr, no
// type erasure). The decision variable and box are typed on the compile-time
// dimension N (Vector<double, N>); the residual count m stays a runtime axis,
// so the Jacobian is Matrix<double, Dynamic, N>.
//
// Reference: N&W Section 10.2-10.3 (Gauss-Newton method).
//            N&W Section 16.6 (active-set identification for bound constraints).
//            N&W Algorithm 4.1 (trust-region update).
//            Nielsen, H. B. (1999) "Damping Parameter in Marquardt's Method".
//            K&W Section 6.3 (Levenberg-Marquardt).

#include "argmin/detail/kkt_residual.h"
#include "argmin/detail/gain_ratio.h"
#include "argmin/detail/bound_projection.h"
#include "argmin/detail/projected_gn_step.h"
#include "argmin/derivative/finite_difference.h"
#include "argmin/result/step_result.h"
#include "argmin/solver/options.h"
#include "argmin/types.h"

#include "argmin/options/least_squares_defaults.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

namespace argmin
{

template <scalar Scalar = double, int N = dynamic_dimension>
struct projected_gn_policy
{
    using scalar_type = Scalar;

    template <int M>
    using rebind = projected_gn_policy<Scalar, M>;

    struct options_type
    {
        Scalar initial_lambda{Scalar(0)};  // 0 = auto via Nielsen init: tau * max(diag(J^T*J))
        Scalar tau{least_squares_defaults<Scalar>::tau};  // for initial lambda computation
        Scalar diagonal_min_clamp{least_squares_defaults<Scalar>::diagonal_min_clamp};
        Scalar lambda_min{least_squares_defaults<Scalar>::lambda_min};
        Scalar lambda_max{least_squares_defaults<Scalar>::lambda_max};

        bool use_dogleg{false};
        std::optional<Scalar> trust_region_radius{};
        std::optional<Scalar> trust_region_expand_threshold{};
        std::optional<Scalar> trust_region_shrink_threshold{};

        std::uint16_t stall_window{50};
        // feasibility_gate intentionally omitted: projected_gn_policy handles box bounds via
        // projection — no general nonlinear constraint_violation is emitted in step_result,
        // so the feasibility_gate criterion would gate on 0. Silent no-op made explicit.
    };

    options_type options{};

    template <typename P = void>
    struct state_type
    {
        const P* problem{nullptr};

        Eigen::Vector<Scalar, N> x;
        Eigen::Vector<Scalar, N> lower;
        Eigen::Vector<Scalar, N> upper;
        Scalar objective_value{};
        Scalar lambda{};
        Scalar nu{least_squares_defaults<Scalar>::nu_initial};
        Scalar delta{Scalar(1)};
        Eigen::VectorX<Scalar> r;
        Eigen::Matrix<Scalar, Eigen::Dynamic, N> J;
        std::uint32_t iteration{0};
        int num_residuals{};

        // Write scratch for the analytic/FD Jacobian evaluation. The problem's
        // jacobian()/fd_jacobian sink is a runtime-sized MatrixXd; on the
        // fixed-N axis it is filled here and copied into the compile-time-typed
        // J. On the dynamic axis J is itself a MatrixXd and is written directly,
        // leaving this buffer unused.
        Eigen::MatrixX<Scalar> J_raw;

        // Caller-owned scratch for the reduced solve and trial evaluation,
        // sized once and reused so the step is allocation-free at fixed N.
        detail::projected_gn_workspace<Scalar, N> ws;
    };

    template <typename Problem, typename Convergence>
    state_type<Problem> init(const Problem& problem, const Eigen::Vector<Scalar, N>& x0,
                    const solver_options<Convergence>& opts, const options_type& policy_opts)
    {
        options = policy_opts;
        return init(problem, x0, opts);
    }

    template <typename Problem, typename Convergence = default_convergence<scalar_type>>
    state_type<Problem> init(const Problem& problem, const Eigen::Vector<Scalar, N>& x0,
                    const solver_options<Convergence>&)
    {
        state_type<Problem> s;
        s.problem = &problem;
        const int n = static_cast<int>(x0.size());

        // Cache bounds and project initial point
        s.lower = problem.lower_bounds();
        s.upper = problem.upper_bounds();
        s.x = detail::project(x0, s.lower, s.upper);
        s.num_residuals = problem.num_residuals();
        s.delta = options.trust_region_radius.value_or(Scalar(1));

        // Evaluate initial residuals and Jacobian
        s.r.resize(s.num_residuals);
        s.J.resize(s.num_residuals, n);
        if constexpr(N != Eigen::Dynamic)
            s.J_raw.resize(s.num_residuals, n);
        s.ws.free_indices.reserve(static_cast<std::size_t>(n));
        s.ws.r_trial.resize(s.num_residuals);
        s.ws.Jd.resize(s.num_residuals);
        s.problem->residuals(s.x, s.r);
        evaluate_jacobian(s);
        s.objective_value = Scalar(0.5) * s.r.squaredNorm();

        init_lambda(s);
        s.nu = least_squares_defaults<Scalar>::nu_initial;
        return s;
    }

    // One projected GN iteration with active-set identification and
    // Nielsen (1999) / dogleg (N&W 4.1) globalization.
    template <typename P>
    step_result<Scalar> step(state_type<P>& s)
    {
        const int n = static_cast<int>(s.x.size());

        // Gauss-Newton Hessian approximation: H = J^T J
        Eigen::Matrix<Scalar, N, N> H = (s.J.transpose() * s.J).eval();

        // Gradient of 0.5*||r||^2: g = J^T r
        Eigen::Vector<Scalar, N> g = (s.J.transpose() * s.r).eval();

        // Active-set identification (N&W Section 16.6)
        detail::identify_free_set_into<Scalar>(s.x, g, s.lower, s.upper, s.ws.free_indices);

        Eigen::Vector<Scalar, N> h;
        if(options.use_dogleg)
        {
            // Extract reduced system for dogleg
            Eigen::MatrixX<Scalar> H_free;
            Eigen::VectorX<Scalar> g_free;
            detail::extract_reduced_system<Scalar>(H, g, s.ws.free_indices, H_free, g_free);

            h = detail::dogleg_step<Scalar>(s.J, H_free, g_free, s.ws.free_indices, n, s.delta);
        }
        else
        {
            // Nielsen/LM mode: solve damped reduced system
            detail::solve_reduced_gn_into<Scalar, N>(H, g, s.lambda,
                                                     options.diagonal_min_clamp, s.ws, h);
        }

        // Project trial point onto bounds before evaluating (N&W Section 16.7)
        Eigen::Vector<Scalar, N> x_plus_h = (s.x + h).eval();
        Eigen::Vector<Scalar, N> x_trial = detail::project(x_plus_h, s.lower, s.upper);

        // Effective step after projection
        Eigen::Vector<Scalar, N> d = (x_trial - s.x).eval();

        // Evaluate trial residuals
        s.problem->residuals(x_trial, s.ws.r_trial);
        Scalar f_trial = Scalar(0.5) * s.ws.r_trial.squaredNorm();

        // Gain ratio via the direct model reduction evaluated at the ACCEPTED
        // (projected/backtracked) step d:
        //   predicted = -g^T d - 0.5 * ||J d||^2
        // This is valid for any d. The Nielsen LM predicted-reduction identity
        // (0.5 h^T (lambda D h - g)) holds only when d is the exact unprojected
        // LM step; on a projected step near an active bound it yields a
        // fictitious denominator that spuriously inflates lambda. Both
        // globalization modes now share this direct form, and the trial is
        // accepted only when predicted > 0 and the trial objective is finite.
        Scalar actual = s.objective_value - f_trial;
        s.ws.Jd.noalias() = s.J * d;
        Scalar predicted = -(g.dot(d) + Scalar(0.5) * s.ws.Jd.squaredNorm());

        Scalar rho = detail::gain_ratio(actual, predicted);

        // Accept/reject with globalization update
        const Scalar old_value = s.objective_value;
        bool accepted = rho > Scalar(0);

        Scalar expand_thresh = options.trust_region_expand_threshold.value_or(Scalar(0.75));
        Scalar shrink_thresh = options.trust_region_shrink_threshold.value_or(Scalar(0.25));

        if(accepted)
        {
            s.x = x_trial;
            s.r = s.ws.r_trial;
            evaluate_jacobian(s);
            s.objective_value = f_trial;

            if(options.use_dogleg)
            {
                detail::update_trust_region<Scalar>(s.delta, rho, d.norm(),
                                                    expand_thresh, shrink_thresh);
            }
            else
            {
                // Nielsen (1999) lambda update
                Scalar factor = Scalar(1) - std::pow(Scalar(2) * rho - Scalar(1), Scalar(3));
                s.lambda *= std::max(least_squares_defaults<Scalar>::damping_factor, factor);
                s.nu = least_squares_defaults<Scalar>::nu_initial;
            }
        }
        else
        {
            if(options.use_dogleg)
            {
                detail::update_trust_region<Scalar>(s.delta, rho, d.norm(),
                                                    expand_thresh, shrink_thresh);
            }
            else
            {
                s.lambda *= s.nu;
                s.nu *= Scalar(2);
            }
        }

        // Clamp lambda
        s.lambda = std::clamp(s.lambda, options.lambda_min, options.lambda_max);

        ++s.iteration;

        // Report h.norm() as step_size even on rejection so step_budget_solver's
        // stall detection does not fire while lambda is still adapting.
        // objective_change carries the ACTUAL objective change (zero on a
        // rejected step); the former hack that reported lambda here leaked an
        // internal damping value into the user-visible step_result, and
        // ftol_reached is correctly gated on the KKT residual.
        Scalar effective_change = s.objective_value - old_value;

        // KKT residual for bound-constrained least-squares: projected-gradient
        // infinity-norm using the gradient g = J^T r of 0.5*||r||^2.
        // Reference: N&W 2e Section 16.7 (projected gradient optimality).
        Scalar kkt = detail::kkt_residual_bound<Scalar, N>(
            s.x, g, s.lower, s.upper);

        return step_result<Scalar>{
            .objective_value = s.objective_value,
            .gradient_norm = g.norm(),
            .step_size = h.norm(),
            .objective_change = effective_change,
            .improved = accepted,
            .kkt_residual = kkt,
        };
    }

    template <typename P>
    void reset(state_type<P>& s, Eigen::Ref<const Eigen::Vector<Scalar, N>> x0)
    {
        s.x = detail::project(x0, s.lower, s.upper);
        const int n = static_cast<int>(x0.size());
        s.r.resize(s.num_residuals);
        s.J.resize(s.num_residuals, n);
        if constexpr(N != Eigen::Dynamic)
            s.J_raw.resize(s.num_residuals, n);
        s.ws.free_indices.reserve(static_cast<std::size_t>(n));
        s.ws.r_trial.resize(s.num_residuals);
        s.ws.Jd.resize(s.num_residuals);
        s.problem->residuals(s.x, s.r);
        evaluate_jacobian(s);
        s.objective_value = Scalar(0.5) * s.r.squaredNorm();

        init_lambda(s);
        s.nu = least_squares_defaults<Scalar>::nu_initial;
        s.delta = options.trust_region_radius.value_or(Scalar(1));
        s.iteration = 0;
    }

    template <typename P>
    void reset_clear(state_type<P>& s, Eigen::Ref<const Eigen::Vector<Scalar, N>> x0)
    {
        reset(s, x0);
    }

private:
    // Analytic-Jacobian detection on the least-squares problem interface: the
    // problem exposes jacobian(x, J) with x on the compile-time dimension and a
    // runtime-sized MatrixXd sink (N&W least_squares concept). When absent, the
    // central finite-difference fallback is selected instead.
    template <typename P>
    static constexpr bool has_analytic_jacobian =
        requires(const P& p, const Eigen::Vector<Scalar, N>& xx, Eigen::MatrixX<Scalar>& JJ) {
            p.jacobian(xx, JJ);
        };

    // Evaluate the Jacobian at s.x into s.J. On the dynamic axis J is a MatrixXd
    // and is written in place; on the fixed-N axis the runtime sink J_raw is
    // filled and copied into the compile-time-typed J.
    template <typename P>
    void evaluate_jacobian(state_type<P>& s) const
    {
        auto write = [&s](Eigen::MatrixX<Scalar>& sink) {
            if constexpr(has_analytic_jacobian<P>)
            {
                s.problem->jacobian(s.x, sink);
            }
            else
            {
                auto residual_fn = [&s](const auto& xx, Eigen::VectorX<Scalar>& rr) {
                    s.problem->residuals(xx, rr);
                };
                fd_jacobian<N>(residual_fn, s.x, sink, s.num_residuals);
            }
        };

        if constexpr(N == Eigen::Dynamic)
        {
            write(s.J);
        }
        else
        {
            write(s.J_raw);
            s.J = s.J_raw;
        }
    }

    // Initialize lambda (Nielsen 1999), honoring the configured
    // initial_lambda / tau exactly. Reads the option fields directly rather
    // than duplicating the literature default.
    template <typename P>
    void init_lambda(state_type<P>& s) const
    {
        if(options.initial_lambda > Scalar(0))
        {
            s.lambda = options.initial_lambda;
        }
        else
        {
            Eigen::Vector<Scalar, N> diag = (s.J.transpose() * s.J).diagonal();
            Scalar max_diag = diag.maxCoeff();
            s.lambda = (max_diag > Scalar(0)) ? options.tau * max_diag : least_squares_defaults<Scalar>::fallback_lambda;
        }
    }
};

}

#endif
