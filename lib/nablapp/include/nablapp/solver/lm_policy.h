#ifndef HPP_GUARD_NABLAPP_SOLVER_LM_POLICY_H
#define HPP_GUARD_NABLAPP_SOLVER_LM_POLICY_H

// Levenberg-Marquardt solver policy for basic_solver.
//
// Implements the Levenberg-Marquardt method for nonlinear least-squares
// problems. Trust-region approach with Gauss-Newton Hessian approximation
// J^T*J and Nielsen (1999) adaptive damping. Each step solves the damped
// normal equations (J^T*J + lambda*diag(J^T*J)) * h = -J^T*r via LDLT
// factorization, then accepts or rejects based on the gain ratio.
//
// The Jacobian source is dispatched at compile time: analytic when the
// problem provides a jacobian() method (satisfies least_squares), FD
// fallback via fd_jacobian otherwise.
//
// Reference: Nielsen, H. B. (1999) "Damping Parameter in Marquardt's
//            Method", IMM-REP-1999-05.
//            K&W Section 6.3-6.4 Algorithm 6.3 (Levenberg-Marquardt).
//            N&W Section 10.2-10.3 (nonlinear least-squares).

#include "nablapp/derivative/finite_difference.h"
#include "nablapp/result/step_result.h"
#include "nablapp/solver/options.h"

#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>

namespace nablapp
{

struct lm_policy
{
    using scalar_type = double;

    struct options_type
    {
        std::optional<double> initial_lambda{};      // LM damping init (default: auto via Nielsen, Nielsen 1999)
        std::optional<double> tau{};                 // initial damping scale (default: 1e-3, Nielsen 1999)
        std::optional<double> nu_initial{};          // damping direction factor (default: 2.0, Nielsen 1999)
        std::optional<double> damping_factor{};      // lambda update 1/3 factor (default: 1/3, Nielsen 1999)
        std::optional<double> diagonal_min_clamp{};  // min diagonal value (default: 1e-8)
        std::optional<double> lambda_min{};          // lambda floor (default: 1e-20)
        std::optional<double> lambda_max{};          // lambda ceiling (default: 1e20)
    };

    options_type options{};

    struct state_type
    {
        Eigen::VectorXd x;
        double objective_value{};     // 0.5 * ||r||^2
        double lambda{};              // damping parameter
        double nu{2.0};              // lambda multiplier on rejection
        Eigen::VectorXd r;           // current residuals
        Eigen::MatrixXd J;           // current Jacobian
        double initial_objective{};  // for divergence detection
        std::uint32_t iteration{0};

        std::function<double(const Eigen::VectorXd&)> eval_value;
        std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> eval_residuals;
        std::function<void(const Eigen::VectorXd&, Eigen::MatrixXd&)> eval_jacobian;
        int num_residuals{};
    };

    template <typename Problem, typename Convergence>
    state_type init(this auto&& self, const Problem& problem, const Eigen::VectorXd& x0,
                    const solver_options<Convergence>& opts, const options_type& policy_opts)
    {
        self.options = policy_opts;
        return self.init(problem, x0, opts);
    }

    template <typename Problem, typename Convergence = default_convergence>
    state_type init(this auto&& self, const Problem& problem, const Eigen::VectorXd& x0,
                    const solver_options<Convergence>& /*opts*/)
    {
        state_type s;
        const int n = x0.size();

        s.x = x0;
        s.num_residuals = problem.num_residuals();

        // Capture closures (problem must outlive solver)
        s.eval_value = [&problem](const Eigen::VectorXd& v) {
            return problem.value(v);
        };
        s.eval_residuals = [&problem](const Eigen::VectorXd& v, Eigen::VectorXd& rr) {
            problem.residuals(v, rr);
        };

        // Jacobian dispatch: analytic when available, FD fallback otherwise
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
        s.eval_residuals(x0, s.r);
        s.eval_jacobian(x0, s.J);
        s.objective_value = 0.5 * s.r.squaredNorm();
        s.initial_objective = s.objective_value;

        // Initialize lambda (Nielsen 1999)
        double init_lambda = self.options.initial_lambda.value_or(0.0);
        double tau = self.options.tau.value_or(1e-3);
        if(init_lambda > 0.0)
        {
            s.lambda = init_lambda;
        }
        else
        {
            Eigen::VectorXd diag = (s.J.transpose() * s.J).diagonal();
            double max_diag = diag.maxCoeff();
            s.lambda = (max_diag > 0.0) ? tau * max_diag : 1e-4;
        }

        s.nu = self.options.nu_initial.value_or(2.0);
        return s;
    }

    // One LM iteration per K&W Algorithm 6.3 with Nielsen (1999) lambda update.
    step_result<double> step(this auto&& self, state_type& s)
    {
        const int n = s.x.size();
        const double diag_min = self.options.diagonal_min_clamp.value_or(1e-8);
        const double lam_min = self.options.lambda_min.value_or(1e-20);
        const double lam_max = self.options.lambda_max.value_or(1e20);
        const double damp_factor = self.options.damping_factor.value_or(1.0 / 3.0);

        // Gauss-Newton Hessian approximation
        Eigen::MatrixXd H = (s.J.transpose() * s.J).eval();

        // Gradient of 0.5*||r||^2: g = J^T * r
        Eigen::VectorXd g = (s.J.transpose() * s.r).eval();

        // Diagonal-scaled damping (K&W Eq. 6.11)
        for(int i = 0; i < n; ++i)
            H(i, i) += s.lambda * std::max(H(i, i), diag_min);

        // Solve damped normal equations via LDLT (D-07)
        Eigen::VectorXd h = H.ldlt().solve(-g);

        // Trial point
        Eigen::VectorXd x_trial = (s.x + h).eval();

        // Evaluate trial residuals
        Eigen::VectorXd r_trial(s.num_residuals);
        s.eval_residuals(x_trial, r_trial);
        double f_trial = 0.5 * r_trial.squaredNorm();

        // Gain ratio (Nielsen 1999): predicted reduction using original J
        // predicted = -h^T g - 0.5 * ||J*h||^2
        double actual = s.objective_value - f_trial;
        double predicted = -(h.dot(g) + 0.5 * (s.J * h).squaredNorm());

        // Guard against near-zero predicted reduction
        double rho = (std::abs(predicted) < 1e-30) ? 1.0 : actual / predicted;

        // Accept/reject with Nielsen (1999) lambda update (D-06)
        const double old_value = s.objective_value;
        bool accepted = rho > 0.0;

        if(accepted)
        {
            s.x = x_trial;
            s.r = r_trial;
            s.eval_jacobian(s.x, s.J);
            s.objective_value = f_trial;

            double factor = 1.0 - std::pow(2.0 * rho - 1.0, 3.0);
            s.lambda *= std::max(damp_factor, factor);
            s.nu = self.options.nu_initial.value_or(2.0);
        }
        else
        {
            s.lambda *= s.nu;
            s.nu *= 2.0;
        }

        // Clamp lambda to prevent overflow/underflow
        s.lambda = std::clamp(s.lambda, lam_min, lam_max);

        ++s.iteration;

        // Divergence detection: objective grows 100x from initial
        std::optional<solver_status> policy_status{};
        if(s.initial_objective > 0.0 && s.objective_value > 100.0 * s.initial_objective)
            policy_status = solver_status::diverged;

        // Report h.norm() as step_size even on rejection to prevent
        // basic_solver stall detection from firing prematurely. The solver
        // is still making progress by adjusting lambda.
        // On rejection, report lambda as objective_change proxy to prevent
        // ftol_reached from firing.
        double effective_change = accepted ? (s.objective_value - old_value) : s.lambda;

        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = g.norm(),
            .step_size = h.norm(),
            .objective_change = effective_change,
            .improved = accepted,
            .x_norm = s.x.norm(),
            .policy_status = policy_status,
        };
    }

    void reset(this auto&& self, state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = x0;
        s.r.resize(s.num_residuals);
        s.J.resize(s.num_residuals, x0.size());
        s.eval_residuals(x0, s.r);
        s.eval_jacobian(x0, s.J);
        s.objective_value = 0.5 * s.r.squaredNorm();
        s.initial_objective = s.objective_value;

        // Nielsen init for lambda using policy options
        double tau = self.options.tau.value_or(1e-3);
        Eigen::VectorXd diag = (s.J.transpose() * s.J).diagonal();
        double max_diag = diag.maxCoeff();
        s.lambda = (max_diag > 0.0) ? tau * max_diag : 1e-4;
        s.nu = self.options.nu_initial.value_or(2.0);
        s.iteration = 0;
    }

    void reset_clear(this auto&& self, state_type& s, const Eigen::VectorXd& x0)
    {
        self.reset(s, x0);
    }
};

}

#endif
