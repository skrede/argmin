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
// Reference: N&W Section 10.2-10.3 (Gauss-Newton method).
//            N&W Section 16.6 (active-set identification for bound constraints).
//            N&W Algorithm 4.1 (trust-region update).
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

struct projected_gn_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = projected_gn_policy;

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

        std::uint16_t stall_window{50};
        // feasibility_gate intentionally omitted: projected_gn_policy handles box bounds via
        // projection — no general nonlinear constraint_violation is emitted in step_result,
        // so the feasibility_gate criterion would gate on 0. Silent no-op made explicit.
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

        // Cache bounds and project initial point
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

    // One projected GN iteration with active-set identification and
    // Nielsen (1999) / dogleg (N&W 4.1) globalization.
    step_result<double> step(state_type& s)
    {
        const int n = static_cast<int>(s.x.size());

        // Gauss-Newton Hessian approximation: H = J^T J
        Eigen::MatrixXd H = (s.J.transpose() * s.J).eval();

        // Gradient of 0.5*||r||^2: g = J^T r
        Eigen::VectorXd g = (s.J.transpose() * s.r).eval();

        // Active-set identification (N&W Section 16.6)
        auto free_indices = detail::identify_free_set(s.x, g, s.lower, s.upper);

        Eigen::VectorXd h;
        if(options.use_dogleg)
        {
            // Extract reduced system for dogleg
            Eigen::MatrixXd H_free;
            Eigen::VectorXd g_free;
            detail::extract_reduced_system(H, g, free_indices, H_free, g_free);

            h = detail::dogleg_step(s.J, H_free, g_free, free_indices, n, s.delta);
        }
        else
        {
            // Nielsen/LM mode: solve damped reduced system
            h = detail::solve_reduced_gn(H, g, free_indices,
                                         s.lambda, options.diagonal_min_clamp);
        }

        // Project trial point onto bounds before evaluating (N&W Section 16.7)
        Eigen::VectorXd x_plus_h = (s.x + h).eval();
        Eigen::VectorXd x_trial = detail::project(x_plus_h, s.lower, s.upper);

        // Effective step after projection
        Eigen::VectorXd d = (x_trial - s.x).eval();

        // Evaluate trial residuals
        Eigen::VectorXd r_trial(s.num_residuals);
        s.eval_residuals(x_trial, r_trial);
        double f_trial = 0.5 * r_trial.squaredNorm();

        // Gain ratio via the direct model reduction evaluated at the ACCEPTED
        // (projected/backtracked) step d:
        //   predicted = -g^T d - 0.5 * ||J d||^2
        // This is valid for any d. The Nielsen LM predicted-reduction identity
        // (0.5 h^T (lambda D h - g)) holds only when d is the exact unprojected
        // LM step; on a projected step near an active bound it yields a
        // fictitious denominator that spuriously inflates lambda. Both
        // globalization modes now share this direct form, and the trial is
        // accepted only when predicted > 0 and the trial objective is finite.
        double actual = s.objective_value - f_trial;
        double predicted = -(g.dot(d) + 0.5 * (s.J * d).squaredNorm());

        const bool valid = std::isfinite(f_trial) && predicted > 0.0;
        double rho = valid ? actual / predicted : 0.0;

        // Accept/reject with globalization update
        const double old_value = s.objective_value;
        bool accepted = valid && rho > 0.0;

        double expand_thresh = options.trust_region_expand_threshold.value_or(0.75);
        double shrink_thresh = options.trust_region_shrink_threshold.value_or(0.25);

        if(accepted)
        {
            s.x = x_trial;
            s.r = r_trial;
            s.eval_jacobian(s.x, s.J);
            s.objective_value = f_trial;

            if(options.use_dogleg)
            {
                detail::update_trust_region(s.delta, rho, d.norm(),
                                            expand_thresh, shrink_thresh);
            }
            else
            {
                // Nielsen (1999) lambda update
                double factor = 1.0 - std::pow(2.0 * rho - 1.0, 3.0);
                s.lambda *= std::max(1.0 / 3.0, factor);
                s.nu = 2.0;
            }
        }
        else
        {
            if(options.use_dogleg)
            {
                detail::update_trust_region(s.delta, rho, d.norm(),
                                            expand_thresh, shrink_thresh);
            }
            else
            {
                s.lambda *= s.nu;
                s.nu *= 2.0;
            }
        }

        // Clamp lambda
        s.lambda = std::clamp(s.lambda, options.lambda_min, options.lambda_max);

        ++s.iteration;

        // Report h.norm() as step_size even on rejection so basic_solver's
        // stall detection does not fire while lambda is still adapting.
        // objective_change carries the ACTUAL objective change (zero on a
        // rejected step); the former hack that reported lambda here leaked an
        // internal damping value into the user-visible step_result, and
        // ftol_reached is correctly gated on the KKT residual.
        double effective_change = s.objective_value - old_value;

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

    void reset(state_type& s, const Eigen::VectorXd& x0)
    {
        s.lower = s.lower;  // bounds already cached
        s.upper = s.upper;
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
};

}

#endif
