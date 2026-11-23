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
#include "argmin/detail/problem_shape.h"
#include "argmin/detail/bound_projection.h"
#include "argmin/detail/evaluation_count.h"
#include "argmin/detail/projected_gn_step.h"
#include "argmin/detail/least_squares_trial.h"
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
        Scalar initial_lambda{Scalar(0)};  // 0: auto
        Scalar tau{least_squares_defaults<Scalar>::tau};
        Scalar diagonal_min_clamp{least_squares_defaults<Scalar>::diagonal_min_clamp};
        Scalar lambda_min{least_squares_defaults<Scalar>::lambda_min};
        Scalar lambda_max{least_squares_defaults<Scalar>::lambda_max};

        bool use_dogleg{false};
        std::optional<Scalar> trust_region_radius{};
        std::optional<Scalar> trust_region_expand_threshold{};
        std::optional<Scalar> trust_region_shrink_threshold{};

        std::uint16_t stall_window{convergence_defaults<scalar_type>::stall_window};
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
        std::uint32_t evaluations{0};
        int num_residuals{};
        bool invalid_problem{false};
        bool non_finite_rejected{false};

        // The runtime-sized sink the problem's jacobian() and fd_jacobian write, at the start or a trial, copied into J
        // only once it is known to be finite.
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
        s.r.resize(s.num_residuals);
        s.J.resize(s.num_residuals, n);
        s.J_raw.resize(s.num_residuals, n);
        s.ws.free_indices.reserve(static_cast<std::size_t>(n));
        s.ws.r_trial.resize(s.num_residuals);
        s.ws.Jd.resize(s.num_residuals);
        if(detail::latch_rejection<Scalar>(s, detail::declares_constraint_rows(problem)))
            return s;
        detail::counted_residuals(s.evaluations, *s.problem, s.x, s.r);
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
        if(const std::optional<step_result<Scalar>> refused = detail::refused_step<Scalar>(s))
            return *refused;
        const int n = static_cast<int>(s.x.size());

        Eigen::Matrix<Scalar, N, N> H = (s.J.transpose() * s.J).eval();

        Eigen::Vector<Scalar, N> g = (s.J.transpose() * s.r).eval();

        // Active-set identification (N&W Section 16.6)
        detail::identify_free_set_into<Scalar>(s.x, g, s.lower, s.upper, s.ws.free_indices);

        Eigen::Vector<Scalar, N> h;
        if(options.use_dogleg)
        {
            Eigen::MatrixX<Scalar> H_free;
            Eigen::VectorX<Scalar> g_free;
            detail::extract_reduced_system<Scalar>(H, g, s.ws.free_indices, H_free, g_free);

            h = detail::dogleg_step<Scalar>(s.J, H_free, g_free, s.ws.free_indices, n, s.delta);
        }
        else
        {
            detail::solve_reduced_gn_into<Scalar, N>(H, g, s.lambda,
                                                     options.diagonal_min_clamp, s.ws, h);
        }

        // Project trial point onto bounds before evaluating (N&W Section 16.7)
        Eigen::Vector<Scalar, N> x_plus_h = (s.x + h).eval();
        Eigen::Vector<Scalar, N> x_trial = detail::project(x_plus_h, s.lower, s.upper);
        Eigen::Vector<Scalar, N> d = (x_trial - s.x).eval();

        detail::counted_residuals(s.evaluations, *s.problem, x_trial, s.ws.r_trial);
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

        const Scalar old_value = s.objective_value;
        bool accepted = rho > Scalar(0);
        const bool failed = detail::failed_trial<N>(s, x_trial, h, f_trial, s.J_raw, accepted);
        rho = failed ? Scalar(0) : rho;
        const bool at_limit = options.use_dogleg ? s.delta <= detail::trust_radius_floor(Scalar(0), s.x.norm()) : s.lambda >= options.lambda_max;

        Scalar expand_thresh = options.trust_region_expand_threshold.value_or(Scalar(0.75));
        Scalar shrink_thresh = options.trust_region_shrink_threshold.value_or(Scalar(0.25));

        if(accepted)
        {
            s.x = x_trial;
            s.r = s.ws.r_trial;
            s.J = s.J_raw;
            s.objective_value = f_trial;
        }

        if(options.use_dogleg)
            detail::update_trust_region<Scalar>(s.delta, rho, d.norm(), expand_thresh, shrink_thresh);
        else if(accepted)
        {
            // Nielsen (1999) lambda update
            Scalar factor = Scalar(1) - std::pow(Scalar(2) * rho - Scalar(1), Scalar(3));
            s.lambda *= std::max(least_squares_defaults<Scalar>::damping_factor, factor);
            s.nu = least_squares_defaults<Scalar>::nu_initial;
        }
        else
        {
            s.lambda *= s.nu;
            s.nu *= Scalar(2);
        }

        s.lambda = std::clamp(s.lambda, options.lambda_min, options.lambda_max);

        ++s.iteration;

        // Report h.norm() as step_size even on rejection so step_budget_solver's stall detection does not fire while
        // lambda is still adapting; objective_change is the actual change, zero on a rejected step. A non-finite trial
        // is a null step, so the step criteria never read the shrinking damped step as a stall.
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
            .is_null_step = failed,
            .kkt_residual = kkt,
            .policy_status = detail::non_finite_trial_status(!failed, at_limit || x_trial == s.x),
        };
    }

    template <typename P>
    solver_status reset(state_type<P>& s, Eigen::Ref<const Eigen::Vector<Scalar, N>> x0)
    {
        if(detail::latch_rejection<Scalar>(s, s.problem->num_residuals() != s.num_residuals || detail::declares_constraint_rows(*s.problem)))
            return solver_status::invalid_problem;
        s.x = detail::project(x0, s.lower, s.upper);
        const int n = static_cast<int>(x0.size());
        s.r.resize(s.num_residuals);
        s.J.resize(s.num_residuals, n);
        s.J_raw.resize(s.num_residuals, n);
        s.ws.free_indices.reserve(static_cast<std::size_t>(n));
        s.ws.r_trial.resize(s.num_residuals);
        s.ws.Jd.resize(s.num_residuals);
        detail::counted_residuals(s.evaluations, *s.problem, s.x, s.r);
        evaluate_jacobian(s);
        s.objective_value = Scalar(0.5) * s.r.squaredNorm();

        init_lambda(s);
        s.nu = least_squares_defaults<Scalar>::nu_initial;
        s.delta = options.trust_region_radius.value_or(Scalar(1));
        s.iteration = 0;
        s.non_finite_rejected = false;
        return detail::latch_status(s);
    }

    template <typename P>
    solver_status reset_clear(state_type<P>& s, Eigen::Ref<const Eigen::Vector<Scalar, N>> x0)
    {
        return reset(s, x0);
    }

private:
    template <typename P>
    void evaluate_jacobian(state_type<P>& s) const
    {
        detail::residual_jacobian<N, Scalar>(s, s.x, s.J_raw);
        s.J = s.J_raw;
    }

    // Initialize lambda (Nielsen 1999) from the configured initial_lambda / tau.
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
