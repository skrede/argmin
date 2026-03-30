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
#include <functional>

namespace nablapp
{

struct lm_policy
{
    using scalar_type = double;

    struct options_type
    {
        double initial_lambda{0.0};  // 0 = auto via Nielsen init: tau * max(diag(J^T*J))
        double tau{1e-3};            // for initial lambda computation
    };

    struct state_type
    {
        Eigen::VectorXd x;
        double objective_value{};     // 0.5 * ||r||^2
        double lambda{};              // damping parameter
        double nu{2.0};              // lambda multiplier on rejection
        Eigen::VectorXd r;           // current residuals
        Eigen::MatrixXd J;           // current Jacobian
        int iteration{0};

        std::function<double(const Eigen::VectorXd&)> eval_value;
        std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> eval_residuals;
        std::function<void(const Eigen::VectorXd&, Eigen::MatrixXd&)> eval_jacobian;
        int num_residuals{};
    };

    template <typename Problem>
    state_type init(this auto&&, const Problem& problem, const Eigen::VectorXd& x0,
                    const solver_options<double>& /*opts*/)
    {
        options_type options;
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

    // One LM iteration per K&W Algorithm 6.3 with Nielsen (1999) lambda update.
    step_result<double> step(this auto&&, state_type& s)
    {
        const int n = s.x.size();

        // Gauss-Newton Hessian approximation
        Eigen::MatrixXd H = (s.J.transpose() * s.J).eval();

        // Gradient of 0.5*||r||^2: g = J^T * r
        Eigen::VectorXd g = (s.J.transpose() * s.r).eval();

        // Diagonal-scaled damping (K&W Eq. 6.11)
        for(int i = 0; i < n; ++i)
            H(i, i) += s.lambda * std::max(H(i, i), 1e-8);

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
            s.lambda *= std::max(1.0 / 3.0, factor);
            s.nu = 2.0;
        }
        else
        {
            s.lambda *= s.nu;
            s.nu *= 2.0;
        }

        // Clamp lambda to prevent overflow/underflow
        s.lambda = std::clamp(s.lambda, 1e-20, 1e20);

        ++s.iteration;

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
        };
    }

    void reset(this auto&&, state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = x0;
        s.r.resize(s.num_residuals);
        s.J.resize(s.num_residuals, x0.size());
        s.eval_residuals(x0, s.r);
        s.eval_jacobian(x0, s.J);
        s.objective_value = 0.5 * s.r.squaredNorm();

        // Nielsen init for lambda
        Eigen::VectorXd diag = (s.J.transpose() * s.J).diagonal();
        double max_diag = diag.maxCoeff();
        s.lambda = (max_diag > 0.0) ? 1e-3 * max_diag : 1e-4;
        s.nu = 2.0;
        s.iteration = 0;
    }

    void reset_clear(this auto&& self, state_type& s, const Eigen::VectorXd& x0)
    {
        self.reset(s, x0);
    }
};

}

#endif
