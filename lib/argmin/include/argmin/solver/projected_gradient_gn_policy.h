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
// Reference: N&W Algorithm 16.1 (projected gradient with backtracking).
//            N&W Section 10.2-10.3 (Gauss-Newton method).
//            N&W Section 16.7 (projected search directions).
//            Nielsen, H. B. (1999) "Damping Parameter in Marquardt's Method".
//            K&W Section 6.3 (Levenberg-Marquardt).

#include "argmin/detail/kkt_residual.h"
#include "argmin/detail/bound_projection.h"
#include "argmin/detail/projected_gn_step.h"
#include "argmin/derivative/finite_difference.h"
#include "argmin/result/step_result.h"
#include "argmin/solver/options.h"
#include "argmin/types.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

namespace argmin
{

struct projected_gradient_gn_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = projected_gradient_gn_policy;

    struct options_type
    {
        double initial_lambda{0.0};     // 0 = auto via Nielsen init: tau * max(diag(J^T*J))
        double tau{1e-3};               // for initial lambda computation
        double diagonal_min_clamp{1e-8};
        double lambda_min{1e-20};
        double lambda_max{1e20};

        bool use_dogleg{false};
        std::optional<double> trust_region_radius{};
        std::optional<double> trust_region_expand_threshold{};
        std::optional<double> trust_region_shrink_threshold{};

        // Backtracking line search parameters (N&W Algorithm 16.1):
        double armijo_c{1e-4};          // Armijo sufficient decrease parameter
        double backtrack_rho{0.5};      // step shrink factor per backtrack
        std::uint32_t max_backtrack{20}; // maximum backtracking iterations

        std::uint16_t stall_window{50};
        // feasibility_gate intentionally omitted: projected_gradient_gn_policy handles box
        // bounds via projection — no general nonlinear constraint_violation is emitted, so the
        // feasibility_gate criterion would gate on 0. Silent no-op made explicit.
    };

    options_type options{};

    struct state_type
    {
        Eigen::VectorXd x;
        Eigen::VectorXd lower;
        Eigen::VectorXd upper;
        double objective_value{};
        double lambda{};
        double nu{2.0};
        double delta{1.0};
        Eigen::VectorXd r;
        Eigen::MatrixXd J;
        std::uint32_t iteration{0};
        int num_residuals{};

        std::function<double(const Eigen::VectorXd&)> eval_value;
        std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> eval_residuals;
        std::function<void(const Eigen::VectorXd&, Eigen::MatrixXd&)> eval_jacobian;
    };

    template <typename Problem, typename Convergence>
    state_type init(const Problem& problem, const Eigen::VectorXd& x0,
                    const solver_options<Convergence>& opts, const options_type& policy_opts)
    {
        options = policy_opts;
        return init(problem, x0, opts);
    }

    template <typename Problem, typename Convergence = default_convergence>
    state_type init(const Problem& problem, const Eigen::VectorXd& x0,
                    const solver_options<Convergence>&)
    {
        state_type s;
        const int n = static_cast<int>(x0.size());

        // Cache bounds and project initial point (N&W Section 16.7)
        s.lower = problem.lower_bounds();
        s.upper = problem.upper_bounds();
        s.x = detail::project(x0, s.lower, s.upper);
        s.num_residuals = problem.num_residuals();
        s.delta = options.trust_region_radius.value_or(1.0);

        // Capture closures (problem must outlive solver)
        s.eval_value = [&problem](const Eigen::VectorXd& v) {
            return problem.value(v);
        };
        s.eval_residuals = [&problem](const Eigen::VectorXd& v, Eigen::VectorXd& rr) {
            problem.residuals(v, rr);
        };

        // Jacobian dispatch: analytic when available, FD fallback
        if constexpr(requires(const Problem& p, const Eigen::VectorXd& xx, Eigen::MatrixXd& JJ) {
                         p.jacobian(xx, JJ);
                     })
        {
            s.eval_jacobian = [&problem](const Eigen::VectorXd& v, Eigen::MatrixXd& JJ) {
                problem.jacobian(v, JJ);
            };
        }
        else
        {
            const int m = s.num_residuals;
            s.eval_jacobian = [&problem, m](const Eigen::VectorXd& v, Eigen::MatrixXd& JJ) {
                auto residual_fn = [&problem](const Eigen::VectorXd& xx, Eigen::VectorXd& rr) {
                    problem.residuals(xx, rr);
                };
                fd_jacobian(residual_fn, v, JJ, m);
            };
        }

        // Evaluate initial residuals and Jacobian
        s.r.resize(s.num_residuals);
        s.J.resize(s.num_residuals, n);
        s.eval_residuals(s.x, s.r);
        s.eval_jacobian(s.x, s.J);
        s.objective_value = 0.5 * s.r.squaredNorm();

        // Initialize lambda (Nielsen 1999)
        if(options.initial_lambda > 0.0)
        {
            s.lambda = options.initial_lambda;
        }
        else
        {
            Eigen::VectorXd diag = (s.J.transpose() * s.J).diagonal();
            double max_diag = diag.maxCoeff();
            s.lambda = (max_diag > 0.0) ? options.tau * max_diag : 1e-4;
        }

        s.nu = 2.0;
        return s;
    }

    // One projected-gradient GN iteration.
    //
    // Default mode: solves full damped system, projects onto bounds, applies
    // Armijo backtracking on the projected step (N&W Algorithm 16.1).
    // Dogleg mode: same trust-region approach as projected_gn_policy.
    step_result<double> step(state_type& s)
    {
        const int n = static_cast<int>(s.x.size());

        // Gauss-Newton Hessian approximation: H = J^T J
        Eigen::MatrixXd H = (s.J.transpose() * s.J).eval();

        // Gradient of 0.5*||r||^2: g = J^T r
        Eigen::VectorXd g = (s.J.transpose() * s.r).eval();

        if(options.use_dogleg)
            return step_dogleg(s, H, g, n);

        return step_backtracking(s, H, g, n);
    }

    void reset(state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = detail::project(x0, s.lower, s.upper);
        s.r.resize(s.num_residuals);
        s.J.resize(s.num_residuals, static_cast<int>(x0.size()));
        s.eval_residuals(s.x, s.r);
        s.eval_jacobian(s.x, s.J);
        s.objective_value = 0.5 * s.r.squaredNorm();

        // Nielsen init for lambda
        Eigen::VectorXd diag = (s.J.transpose() * s.J).diagonal();
        double max_diag = diag.maxCoeff();
        s.lambda = (max_diag > 0.0) ? 1e-3 * max_diag : 1e-4;
        s.nu = 2.0;
        s.delta = options.trust_region_radius.value_or(1.0);
        s.iteration = 0;
    }

    void reset_clear(state_type& s, const Eigen::VectorXd& x0)
    {
        reset(s, x0);
    }

private:
    // Projected-gradient backtracking line search (N&W Algorithm 16.1).
    //
    // Solves the full damped system (H + lambda*D)h = -g, projects onto bounds
    // to get direction d = P(x + h) - x, then backtracks alpha until the
    // Armijo sufficient decrease condition holds on the projected trial point.
    step_result<double> step_backtracking(
        state_type& s,
        const Eigen::MatrixXd& H, const Eigen::VectorXd& g, int n)
    {
        // Solve full damped system: (H + lambda*D) h = -g
        Eigen::MatrixXd H_damped = H;
        for(int i = 0; i < n; ++i)
            H_damped(i, i) += s.lambda * std::max(H(i, i), options.diagonal_min_clamp);

        Eigen::VectorXd h = H_damped.ldlt().solve(-g);

        // Projected direction (N&W Section 16.7):
        // d = P(x + h, l, u) - x
        Eigen::VectorXd x_plus_h = (s.x + h).eval();
        Eigen::VectorXd d = detail::project(x_plus_h, s.lower, s.upper) - s.x;

        // Directional derivative along projected direction
        double gTd = g.dot(d);

        // Armijo backtracking (N&W Algorithm 16.1)
        double alpha = 1.0;
        const double c = options.armijo_c;
        const double rho_bt = options.backtrack_rho;
        const std::uint32_t max_bt = options.max_backtrack;

        bool accepted = false;
        Eigen::VectorXd x_trial(n);
        Eigen::VectorXd r_trial(s.num_residuals);
        double f_trial{};

        for(std::uint32_t k = 0; k < max_bt; ++k)
        {
            Eigen::VectorXd x_candidate = (s.x + alpha * d).eval();
            x_trial = detail::project(x_candidate, s.lower, s.upper);
            s.eval_residuals(x_trial, r_trial);
            f_trial = 0.5 * r_trial.squaredNorm();

            // Armijo condition: f(x_trial) <= f(x) + c * alpha * g^T d
            if(f_trial <= s.objective_value + c * alpha * gTd)
            {
                accepted = true;
                break;
            }
            alpha *= rho_bt;
        }

        const double old_value = s.objective_value;

        if(accepted)
        {
            s.x = x_trial;
            s.r = r_trial;
            s.eval_jacobian(s.x, s.J);
            s.objective_value = f_trial;

            // Nielsen (1999) lambda update using gain ratio
            double actual = old_value - f_trial;
            double predicted = detail::predicted_reduction_lm(
                d, g, s.lambda, options.diagonal_min_clamp, H.diagonal());
            double rho = (std::abs(predicted) < 1e-30) ? 1.0 : actual / predicted;

            double factor = 1.0 - std::pow(2.0 * rho - 1.0, 3.0);
            s.lambda *= std::max(1.0 / 3.0, factor);
            s.nu = 2.0;
        }
        else
        {
            // Reject: increase damping
            s.lambda *= s.nu;
            s.nu *= 2.0;
        }

        // Clamp lambda
        s.lambda = std::clamp(s.lambda, options.lambda_min, options.lambda_max);

        ++s.iteration;

        double effective_change = accepted ? (s.objective_value - old_value) : s.lambda;

        // KKT residual for bound-constrained least-squares: projected-gradient
        // infinity-norm using the gradient g = J^T r of 0.5*||r||^2.
        // Reference: N&W 2e Section 16.7 (projected gradient optimality).
        double kkt = detail::kkt_residual_bound<double, dynamic_dimension>(
            s.x, g, s.lower, s.upper);

        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = g.norm(),
            .step_size = h.norm(),
            .objective_change = effective_change,
            .improved = accepted,
            .kkt_residual = kkt,
        };
    }

    // Dogleg trust-region step (N&W Algorithm 4.1).
    //
    // Same as projected_gn_policy dogleg mode: active-set identification,
    // dogleg interpolation on free subspace, projection onto bounds, gain-ratio
    // acceptance with trust-region radius update.
    step_result<double> step_dogleg(
        state_type& s,
        const Eigen::MatrixXd& H, const Eigen::VectorXd& g, int n)
    {
        // Active-set identification for dogleg (N&W Section 16.6)
        auto free_indices = detail::identify_free_set(s.x, g, s.lower, s.upper);

        Eigen::MatrixXd H_free;
        Eigen::VectorXd g_free;
        detail::extract_reduced_system(H, g, free_indices, H_free, g_free);

        Eigen::VectorXd h = detail::dogleg_step(s.J, H_free, g_free, free_indices, n, s.delta);

        // Project trial point onto bounds (N&W Section 16.7)
        Eigen::VectorXd x_plus_h = (s.x + h).eval();
        Eigen::VectorXd x_trial = detail::project(x_plus_h, s.lower, s.upper);
        Eigen::VectorXd d = (x_trial - s.x).eval();

        // Evaluate trial residuals
        Eigen::VectorXd r_trial(s.num_residuals);
        s.eval_residuals(x_trial, r_trial);
        double f_trial = 0.5 * r_trial.squaredNorm();

        // Gain ratio
        double actual = s.objective_value - f_trial;
        double predicted = detail::predicted_reduction_tr(d, g, H);
        double rho = (std::abs(predicted) < 1e-30) ? 1.0 : actual / predicted;

        const double old_value = s.objective_value;
        bool accepted = rho > 0.0;

        double expand_thresh = options.trust_region_expand_threshold.value_or(0.75);
        double shrink_thresh = options.trust_region_shrink_threshold.value_or(0.25);

        if(accepted)
        {
            s.x = x_trial;
            s.r = r_trial;
            s.eval_jacobian(s.x, s.J);
            s.objective_value = f_trial;
        }

        detail::update_trust_region(s.delta, rho, d.norm(), expand_thresh, shrink_thresh);

        // Clamp lambda
        s.lambda = std::clamp(s.lambda, options.lambda_min, options.lambda_max);

        ++s.iteration;

        double effective_change = accepted ? (s.objective_value - old_value) : s.lambda;

        // KKT residual for bound-constrained least-squares: projected-gradient
        // infinity-norm using the gradient g = J^T r of 0.5*||r||^2.
        // Reference: N&W 2e Section 16.7 (projected gradient optimality).
        double kkt = detail::kkt_residual_bound<double, dynamic_dimension>(
            s.x, g, s.lower, s.upper);

        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = g.norm(),
            .step_size = h.norm(),
            .objective_change = effective_change,
            .improved = accepted,
            .kkt_residual = kkt,
        };
    }
};

}

#endif
