#ifndef HPP_GUARD_ARGMIN_SOLVER_LM_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_LM_POLICY_H

// Levenberg-Marquardt solver policy for step_budget_solver.
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

#include "argmin/detail/gain_ratio.h"
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

template <int N = dynamic_dimension>
struct lm_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = lm_policy<M>;

    struct options_type
    {
        std::optional<double> initial_lambda{};      // LM damping init (default: auto via Nielsen, Nielsen 1999)
        std::optional<double> tau{};                 // initial damping scale (default: 1e-3, Nielsen 1999)
        std::optional<double> nu_initial{};          // damping direction factor (default: 2.0, Nielsen 1999)
        std::optional<double> damping_factor{};      // lambda update 1/3 factor (default: 1/3, Nielsen 1999)
        std::optional<double> diagonal_min_clamp{};  // min diagonal value (default: 1e-8)
        std::optional<double> lambda_min{};          // lambda floor (default: 1e-20)
        std::optional<double> lambda_max{};          // lambda ceiling (default: 1e20)
        std::uint16_t stall_window{50};
        // feasibility_gate intentionally omitted: lm_policy is unconstrained least-squares;
        // step_result reports no constraint_violation signal, so gating ftol_reached on a
        // feasibility threshold is vacuous. Silent no-op made explicit here.
    };

    options_type options{};

    template <typename P = void>
    struct state_type
    {
        const P* problem{nullptr};
        Eigen::Vector<double, N> x;
        double objective_value{};     // 0.5 * ||r||^2
        double lambda{};              // damping parameter
        double nu{2.0};              // lambda multiplier on rejection
        Eigen::VectorXd r;           // current residuals (m-dimensional, stays dynamic)
        Eigen::MatrixXd J;           // current Jacobian (m x n, stays dynamic for concept compat)
        double initial_objective{};  // for divergence detection
        std::uint32_t iteration{0};
        int num_residuals{};
    };

    template <typename Problem, typename Convergence>
    state_type<Problem> init(const Problem& problem,
                    const Eigen::Vector<double, N>& x0,
                    const solver_options<Convergence>& opts, const options_type& policy_opts)
    {
        options = policy_opts;
        return init(problem, x0, opts);
    }

    template <typename Problem, typename Convergence = default_convergence>
    state_type<Problem> init(const Problem& problem,
                    const Eigen::Vector<double, N>& x0,
                    const solver_options<Convergence>& /*opts*/)
    {
        state_type<Problem> s;
        s.problem = &problem;
        const int n = x0.size();

        s.x = x0;
        s.num_residuals = problem.num_residuals();

        // Evaluate initial residuals and Jacobian
        s.r.resize(s.num_residuals);
        s.J.resize(s.num_residuals, n);
        s.problem->residuals(x0, s.r);
        if constexpr(requires(const Problem& p, const Eigen::Vector<double, N>& xx, Eigen::MatrixXd& JJ) {
                         p.jacobian(xx, JJ);
                     })
            s.problem->jacobian(x0, s.J);
        else
        {
            auto residual_fn = [&](const Eigen::Vector<double, N>& xx, Eigen::VectorXd& rr) {
                s.problem->residuals(xx, rr);
            };
            fd_jacobian<N>(residual_fn, x0, s.J, s.num_residuals);
        }
        s.objective_value = 0.5 * s.r.squaredNorm();
        s.initial_objective = s.objective_value;

        // Initialize lambda (Nielsen 1999)
        double init_lambda = options.initial_lambda.value_or(0.0);
        double tau = options.tau.value_or(1e-3);
        if(init_lambda > 0.0)
        {
            s.lambda = init_lambda;
        }
        else
        {
            Eigen::Vector<double, N> diag = (s.J.transpose() * s.J).diagonal();
            double max_diag = diag.maxCoeff();
            s.lambda = (max_diag > 0.0) ? tau * max_diag : 1e-4;
        }

        s.nu = options.nu_initial.value_or(2.0);
        return s;
    }

    // One LM iteration per K&W Algorithm 6.3 with Nielsen (1999) lambda update.
    template <typename P>
    step_result<double> step(state_type<P>& s)
    {
        const int n = s.x.size();
        const double diag_min = options.diagonal_min_clamp.value_or(1e-8);
        const double lam_min = options.lambda_min.value_or(1e-20);
        const double lam_max = options.lambda_max.value_or(1e20);
        const double damp_factor = options.damping_factor.value_or(1.0 / 3.0);

        // Gauss-Newton Hessian approximation
        Eigen::Matrix<double, N, N> H(n, n);
        H.noalias() = s.J.transpose() * s.J;

        // Gradient of 0.5*||r||^2: g = J^T * r
        Eigen::Vector<double, N> g(n);
        g.noalias() = s.J.transpose() * s.r;

        // Diagonal-scaled damping (K&W Eq. 6.11)
        for(int i = 0; i < n; ++i)
            H(i, i) += s.lambda * std::max(H(i, i), diag_min);

        // Solve damped normal equations via LDLT
        Eigen::Vector<double, N> h = H.ldlt().solve(-g);

        // Trial point
        Eigen::Vector<double, N> x_trial = (s.x + h).eval();

        // Evaluate trial residuals
        Eigen::VectorXd r_trial(s.num_residuals); // residual dimension, stays dynamic
        s.problem->residuals(x_trial, r_trial);
        double f_trial = 0.5 * r_trial.squaredNorm();

        // Gain ratio (Nielsen 1999): direct model reduction using the original
        // Jacobian, predicted = -h^T g - 0.5 * ||J*h||^2. For the exact LM step
        // this equals 0.5*||J*h||^2 + lambda*h^T D h >= 0, so a non-positive
        // predicted reduction signals a degenerate or garbage solve rather than
        // a real model decrease -- in that case the trial must NOT be accepted.
        double actual = s.objective_value - f_trial;
        double predicted = -(h.dot(g) + 0.5 * (s.J * h).squaredNorm());

        // Guarded gain ratio (detail::gain_ratio): requires a strictly positive
        // predicted reduction and a finite actual reduction before dividing,
        // returning 0 otherwise. This replaces the |predicted|<1e-30 -> rho=1.0
        // silent accept and gates a NaN/Inf residual evaluation (which makes
        // `actual` non-finite) so a divergent trial cannot masquerade as an
        // improvement.
        double rho = detail::gain_ratio(actual, predicted);

        // Accept/reject with Nielsen (1999) lambda update
        const double old_value = s.objective_value;
        bool accepted = rho > 0.0;

        if(accepted)
        {
            s.x = x_trial;
            s.r = r_trial;
            if constexpr(requires(const P& p, const Eigen::Vector<double, N>& xx, Eigen::MatrixXd& JJ) {
                             p.jacobian(xx, JJ);
                         })
                s.problem->jacobian(s.x, s.J);
            else
            {
                auto residual_fn = [&](const Eigen::Vector<double, N>& xx, Eigen::VectorXd& rr) {
                    s.problem->residuals(xx, rr);
                };
                fd_jacobian<N>(residual_fn, s.x, s.J, s.num_residuals);
            }
            s.objective_value = f_trial;

            double factor = 1.0 - std::pow(2.0 * rho - 1.0, 3.0);
            s.lambda *= std::max(damp_factor, factor);
            s.nu = options.nu_initial.value_or(2.0);
        }
        else
        {
            s.lambda *= s.nu;
            s.nu *= 2.0;
        }

        // Clamp lambda to prevent overflow/underflow
        s.lambda = std::clamp(s.lambda, lam_min, lam_max);

        ++s.iteration;

        // Divergence detection: a non-finite trial that the guard rejected, or
        // an objective that has grown 100x from its initial value, is a genuine
        // failure rather than a routine damping increase.
        std::optional<solver_status> policy_status{};
        if(!std::isfinite(f_trial))
            policy_status = solver_status::diverged;
        else if(s.initial_objective > 0.0 && s.objective_value > 100.0 * s.initial_objective)
            policy_status = solver_status::diverged;

        // Report h.norm() as step_size even on rejection so step_budget_solver's stall
        // detection does not fire while the solver is still adjusting lambda.
        // objective_change carries the ACTUAL objective change (zero on a
        // rejected step, where the iterate did not move); the former hack that
        // reported lambda here leaked an internal damping value into the
        // user-visible step_result. ftol_reached is now correctly gated on the
        // KKT residual (convergence.h), so no proxy is needed to suppress it.
        double effective_change = s.objective_value - old_value;

        // KKT residual for unconstrained least-squares: infinity-norm of the
        // gradient g = J^T r of 0.5*||r||^2. First-order optimality holds iff
        // ||J^T r||_inf = 0, so the gradient infinity-norm IS the KKT residual
        // here; no detail::kkt_residual helper is needed because there are no
        // multipliers or bound projections to compose.
        // Reference: N&W 2e Section 10.3 (nonlinear least-squares first-order
        //            conditions); N&W 2e Section 12.1 (KKT conditions reduce
        //            to stationarity when no constraints are present).
        double kkt = g.template lpNorm<Eigen::Infinity>();

        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = g.norm(),
            .step_size = h.norm(),
            .objective_change = effective_change,
            .improved = accepted,
            .x_norm = s.x.norm(),
            .kkt_residual = kkt,
            .policy_status = policy_status,
        };
    }

    template <typename P>
    void reset(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        s.x = x0;
        s.r.resize(s.num_residuals);
        s.J.resize(s.num_residuals, x0.size());
        s.problem->residuals(x0, s.r);
        if constexpr(requires(const P& p, const Eigen::Vector<double, N>& xx, Eigen::MatrixXd& JJ) {
                         p.jacobian(xx, JJ);
                     })
            s.problem->jacobian(x0, s.J);
        else
        {
            auto residual_fn = [&](const Eigen::Vector<double, N>& xx, Eigen::VectorXd& rr) {
                s.problem->residuals(xx, rr);
            };
            fd_jacobian<N>(residual_fn, x0, s.J, s.num_residuals);
        }
        s.objective_value = 0.5 * s.r.squaredNorm();
        s.initial_objective = s.objective_value;

        // Nielsen init for lambda using policy options
        double tau = options.tau.value_or(1e-3);
        Eigen::Vector<double, N> diag = (s.J.transpose() * s.J).diagonal();
        double max_diag = diag.maxCoeff();
        s.lambda = (max_diag > 0.0) ? tau * max_diag : 1e-4;
        s.nu = options.nu_initial.value_or(2.0);
        s.iteration = 0;
    }

    template <typename P>
    void reset_clear(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        reset(s, x0);
    }
};

}

#endif
