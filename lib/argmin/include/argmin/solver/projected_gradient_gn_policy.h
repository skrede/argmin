#ifndef HPP_GUARD_ARGMIN_SOLVER_PROJECTED_GRADIENT_GN_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_PROJECTED_GRADIENT_GN_POLICY_H

// Projected-gradient Gauss-Newton solver policy with backtracking line search.
//
// Solves bound-constrained nonlinear least-squares problems by computing the
// full Gauss-Newton step on the damped system (J^T J + lambda*D) h = -J^T r,
// projecting the result onto the feasible box, and accepting via projected-
// gradient backtracking line search (Armijo condition on the projected step).
//
// This is the sister policy to projected_gn_policy (active-set variant).
// Where projected_gn_policy identifies free/active variables and solves a
// reduced system, this policy operates on the full system and relies on
// projection + backtracking for feasibility.
//
// Two globalization modes:
//   1. Default: projected-gradient backtracking (N&W Algorithm 16.1).
//   2. Dogleg trust-region (N&W Algorithm 4.1) -- same as projected_gn_policy.
//
// Nielsen (1999) adaptive damping controls the LM parameter lambda.
//
// The problem is bound by a stored pointer and dispatched at compile time
// (residuals/value analytic; Jacobian analytic-or-finite-difference via
// if constexpr, no type erasure). The decision variable and box are typed on
// the compile-time dimension N; the residual count m stays a runtime axis.
//
// Reference: N&W Algorithm 16.1 (projected gradient with backtracking).
//            N&W Section 10.2-10.3 (Gauss-Newton method).
//            N&W Section 16.7 (projected search directions).
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
struct projected_gradient_gn_policy
{
    using scalar_type = Scalar;

    template <int M>
    using rebind = projected_gradient_gn_policy<Scalar, M>;

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

        // Backtracking line search parameters (N&W Algorithm 16.1):
        Scalar armijo_c{least_squares_defaults<Scalar>::armijo_c};
        Scalar backtrack_rho{least_squares_defaults<Scalar>::backtrack_rho};
        std::uint32_t max_backtrack{20};

        std::uint16_t stall_window{convergence_defaults<scalar_type>::stall_window};
        // No feasibility_gate: the box is handled by projection, so no constraint_violation is emitted to gate on.
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

        // Runtime-sized Jacobian sink, at the start or a trial, copied into J only once it is known to be finite.
        Eigen::MatrixX<Scalar> J_raw;

        // Sized once and reused so the step is allocation-free at fixed N.
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

        // Cache bounds and project initial point (N&W Section 16.7)
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

    // One projected-gradient GN iteration.
    //
    // Default mode: solves full damped system, projects onto bounds, applies
    // Armijo backtracking on the projected step (N&W Algorithm 16.1).
    // Dogleg mode: same trust-region approach as projected_gn_policy.
    template <typename P>
    step_result<Scalar> step(state_type<P>& s)
    {
        if(const std::optional<step_result<Scalar>> refused = detail::refused_step<Scalar>(s))
            return *refused;
        const int n = static_cast<int>(s.x.size());

        Eigen::Matrix<Scalar, N, N> H = (s.J.transpose() * s.J).eval();

        Eigen::Vector<Scalar, N> g = (s.J.transpose() * s.r).eval();

        if(options.use_dogleg)
            return step_dogleg(s, H, g, n);

        return step_backtracking(s, H, g, n);
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
    // Projected-gradient backtracking line search (N&W Algorithm 16.1).
    // Solves the full damped system (H + lambda*D)h = -g, projects onto bounds
    // to get direction d = P(x + h) - x, then backtracks alpha until the
    // Armijo sufficient decrease condition holds on the projected trial point.
    template <typename P>
    step_result<Scalar> step_backtracking(
        state_type<P>& s,
        const Eigen::Matrix<Scalar, N, N>& H, const Eigen::Vector<Scalar, N>& g, int n)
    {
        // Solve full damped system: (H + lambda*D) h = -g
        Eigen::Matrix<Scalar, N, N> H_damped = H;
        for(int i = 0; i < n; ++i)
            H_damped(i, i) += s.lambda * std::max(H(i, i), options.diagonal_min_clamp);

        Eigen::Vector<Scalar, N> h = H_damped.ldlt().solve(-g);

        // Projected direction (N&W Section 16.7):
        // d = P(x + h, l, u) - x
        Eigen::Vector<Scalar, N> x_plus_h = (s.x + h).eval();
        Eigen::Vector<Scalar, N> d = detail::project(x_plus_h, s.lower, s.upper) - s.x;

        Scalar gTd = g.dot(d);

        // Armijo backtracking (N&W Algorithm 16.1)
        Scalar alpha = Scalar(1);
        const Scalar c = options.armijo_c;
        const Scalar rho_bt = options.backtrack_rho;
        const std::uint32_t max_bt = options.max_backtrack;

        bool accepted = false;
        Eigen::Vector<Scalar, N> x_trial(n);
        Scalar f_trial{};

        for(std::uint32_t k = 0; k < max_bt; ++k)
        {
            Eigen::Vector<Scalar, N> x_candidate = (s.x + alpha * d).eval();
            x_trial = detail::project(x_candidate, s.lower, s.upper);
            detail::counted_residuals(s.evaluations, *s.problem, x_trial, s.ws.r_trial);
            f_trial = Scalar(0.5) * s.ws.r_trial.squaredNorm();

            // Armijo condition: f(x_trial) <= f(x) + c * alpha * g^T d
            if(f_trial <= s.objective_value + c * alpha * gTd)
            {
                accepted = true;
                break;
            }
            alpha *= rho_bt;
        }

        const Scalar old_value = s.objective_value;
        const bool failed = detail::failed_trial<N>(s, x_trial, d, f_trial, s.J_raw, accepted);
        const bool at_limit = s.lambda >= options.lambda_max;

        if(accepted)
        {
            // Gain ratio via the direct model reduction evaluated at the ACTUAL
            // accepted (projected, backtracked) displacement d_acc = x_trial - x:
            //   predicted = -g^T d_acc - 0.5 * ||J d_acc||^2
            // computed with the pre-update Jacobian. This replaces the Nielsen
            // LM predicted-reduction identity applied to the full alpha=1
            // projected direction, which is only valid for the exact unprojected
            // LM step and yields a fictitious denominator near an active bound.
            Eigen::Vector<Scalar, N> d_acc = (x_trial - s.x).eval();
            Scalar actual = old_value - f_trial;
            s.ws.Jd.noalias() = s.J * d_acc;
            Scalar predicted = -(g.dot(d_acc) + Scalar(0.5) * s.ws.Jd.squaredNorm());
            Scalar rho = detail::gain_ratio(actual, predicted);

            s.x = x_trial;
            s.r = s.ws.r_trial;
            s.J = s.J_raw;
            s.objective_value = f_trial;

            // Nielsen (1999) lambda update using the guarded gain ratio
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

        // objective_change carries the actual objective change, zero on a rejected step. A non-finite trial is a null
        // step, so the step criteria never read the shrinking damped step as a stall.
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

    // Dogleg trust-region step (N&W Algorithm 4.1).
    // Same as projected_gn_policy dogleg mode: active-set identification,
    // dogleg interpolation on free subspace, projection onto bounds, gain-ratio
    // acceptance with trust-region radius update.
    template <typename P>
    step_result<Scalar> step_dogleg(
        state_type<P>& s,
        const Eigen::Matrix<Scalar, N, N>& H, const Eigen::Vector<Scalar, N>& g, int n)
    {
        // Active-set identification for dogleg (N&W Section 16.6)
        detail::identify_free_set_into<Scalar>(s.x, g, s.lower, s.upper, s.ws.free_indices);

        Eigen::MatrixX<Scalar> H_free;
        Eigen::VectorX<Scalar> g_free;
        detail::extract_reduced_system<Scalar>(H, g, s.ws.free_indices, H_free, g_free);

        Eigen::Vector<Scalar, N> h = detail::dogleg_step<Scalar>(s.J, H_free, g_free, s.ws.free_indices, n, s.delta);

        // Project trial point onto bounds (N&W Section 16.7)
        Eigen::Vector<Scalar, N> x_plus_h = (s.x + h).eval();
        Eigen::Vector<Scalar, N> x_trial = detail::project(x_plus_h, s.lower, s.upper);
        Eigen::Vector<Scalar, N> d = (x_trial - s.x).eval();

        detail::counted_residuals(s.evaluations, *s.problem, x_trial, s.ws.r_trial);
        Scalar f_trial = Scalar(0.5) * s.ws.r_trial.squaredNorm();

        // Gain ratio from the direct model reduction at the projected dogleg step, valid for any d.
        Scalar actual = s.objective_value - f_trial;
        s.ws.Jd.noalias() = s.J * d;
        Scalar predicted = -(g.dot(d) + Scalar(0.5) * s.ws.Jd.squaredNorm());
        Scalar rho = detail::gain_ratio(actual, predicted);

        const Scalar old_value = s.objective_value;
        bool accepted = rho > Scalar(0);
        const bool failed = detail::failed_trial<N>(s, x_trial, h, f_trial, s.J_raw, accepted);
        rho = failed ? Scalar(0) : rho;
        const bool at_limit = s.delta <= detail::trust_radius_floor(Scalar(0), s.x.norm());

        Scalar expand_thresh = options.trust_region_expand_threshold.value_or(Scalar(0.75));
        Scalar shrink_thresh = options.trust_region_shrink_threshold.value_or(Scalar(0.25));

        if(accepted)
        {
            s.x = x_trial;
            s.r = s.ws.r_trial;
            s.J = s.J_raw;
            s.objective_value = f_trial;
        }

        detail::update_trust_region<Scalar>(s.delta, rho, d.norm(), expand_thresh, shrink_thresh);

        s.lambda = std::clamp(s.lambda, options.lambda_min, options.lambda_max);

        ++s.iteration;

        // objective_change carries the actual objective change, zero on a rejected step; a non-finite trial is a null step.
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
    void evaluate_jacobian(state_type<P>& s) const
    {
        detail::residual_jacobian<N, Scalar>(s, s.x, s.J_raw);
        s.J = s.J_raw;
    }

    // Initialize lambda (Nielsen 1999) honoring initial_lambda / tau.
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
