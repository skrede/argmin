#ifndef HPP_GUARD_ARGMIN_SOLVER_COBYLA_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_COBYLA_POLICY_H

// COBYLA solver policy for basic_solver.
//
// Implements Powell's Constrained Optimization BY Linear Approximation: a
// derivative-free trust-region method that maintains a simplex of n+1 points,
// builds linear models of the objective and every constraint by interpolation,
// solves a two-stage linear trust-region subproblem for the step, and adapts a
// merit penalty toward feasibility only when reaching feasibility costs
// objective. The heavy lifting lives in detail::cobyla_engine, a faithful port
// of Powell's cobylb driver; this policy adapts the argmin problem interface to
// the engine and threads the incumbent through basic_solver.
//
// Requires: objective<P,S> && constrained_values<P,S> && bound_constrained<P,S>.
// No gradient or constraint Jacobian is needed -- COBYLA uses only function and
// constraint value evaluations.
//
// Constraint convention: argmin stores equalities first (c_eq, wanted zero) and
// inequalities second (c_ineq, wanted non-negative). The policy rewrites these
// into Powell's all-non-negative form: each inequality maps to itself, each
// equality to the pair {h, -h}, and every finite box bound to a linear
// constraint, matching the reference COBYLA treatment of bounds.
//
// Provenance: this port follows NLopt's cobyla.c (an f2c translation of
// Powell's Fortran plus Steven G. Johnson's modifications), not Powell's
// paper verbatim. Two behaviors come from the NLopt/SGJ line and are NOT
// in Powell 1994: the trust-region radius may be doubled on an accurate
// predicted-vs-actual reduction (Powell's rho is monotone non-increasing),
// and a small pseudo-random jitter perturbs degenerate geometry steps. Do
// not "correct" either against the 1994 paper -- they are intentional.
//
// Reference: Powell, M. J. D. (1994) "A direct search optimization method that
//            models the objective and constraint functions by linear
//            interpolation." K&W 2e, Section 10.7.

#include "argmin/detail/cobyla_simplex.h"
#include "argmin/detail/bound_projection.h"
#include "argmin/detail/lagrangian.h"
#include "argmin/result/step_result.h"
#include "argmin/solver/options.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>

namespace argmin
{

struct cobyla_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = cobyla_policy;

    struct options_type
    {
        double initial_trust_radius{0.0};   // 0 means auto (10% of max bound range, else 1.0)
        double final_trust_radius{1e-8};
        std::uint16_t stall_window{50};
        double feasibility_gate{1e-4};
    };

    options_type options{};

    template <typename P = void>
    struct state_type
    {
        const P* problem{nullptr};
        Eigen::VectorXd x;
        Eigen::VectorXd lower;
        Eigen::VectorXd upper;
        Eigen::VectorXd c_eq;
        Eigen::VectorXd c_ineq;
        double objective_value{};
        // Powell merit penalty. Adapts upward only through
        // barmu = (predicted objective increase) / (predicted reduction in max
        // violation) -- i.e. only when reaching feasibility costs objective.
        double parmu{0.0};
        int n_eq{0};
        int n_ineq{0};
        int m_powell{0};
        std::uint32_t iteration{0};

        // The engine is heap-owned so state moves leave its internal pointers
        // (captured by the evaluator) valid.
        std::shared_ptr<detail::cobyla_engine<double>> engine;
    };

    template <typename Problem, typename Convergence>
        requires objective<Problem> && constrained_values<Problem> && bound_constrained<Problem>
    state_type<Problem> init(const Problem& problem, const Eigen::VectorXd& x0,
                    const solver_options<Convergence>& opts, const options_type& policy_opts)
    {
        options = policy_opts;
        return init(problem, x0, opts);
    }

    template <typename Problem, typename Convergence = default_convergence>
        requires objective<Problem> && constrained_values<Problem> && bound_constrained<Problem>
    state_type<Problem> init(const Problem& problem, const Eigen::VectorXd& x0,
                    const solver_options<Convergence>& /*opts*/)
    {
        const int n = problem.dimension();
        state_type<Problem> s;
        s.problem = &problem;
        s.lower = problem.lower_bounds();
        s.upper = problem.upper_bounds();
        s.x = detail::project(x0, s.lower, s.upper);
        s.n_eq = problem.num_equality();
        s.n_ineq = problem.num_inequality();

        // Count finite box bounds (added as explicit Powell constraints).
        int n_bound = 0;
        for(int i = 0; i < n; ++i)
        {
            if(std::isfinite(s.lower[i])) ++n_bound;
            if(std::isfinite(s.upper[i])) ++n_bound;
        }
        s.m_powell = s.n_ineq + 2 * s.n_eq + n_bound;

        double rhobeg = auto_trust_radius(s.lower, s.upper, n);
        double rhoend = options.final_trust_radius;

        s.engine = std::make_shared<detail::cobyla_engine<double>>();
        make_evaluator_and_init(s, rhobeg, rhoend);

        sync_from_engine(s);
        s.iteration = 0;
        return s;
    }

    template <typename P>
    step_result<double> step(state_type<P>& s)
    {
        const double old_f = s.objective_value;

        bool running = s.engine->advance();
        ++s.iteration;
        sync_from_engine(s);

        const double obj_change = s.objective_value - old_f;
        const bool improved = s.objective_value < old_f;

        std::optional<solver_status> policy_status{};
        if(!running)
            policy_status = s.engine->status();

        // COBYLA owns its own termination: it iterates until the trust radius
        // RHO contracts to RHOEND and then reports xtol_reached via
        // policy_status. Until then the reported progress magnitudes are kept
        // at the trust-radius scale so basic_solver's step / objective / stall
        // criteria do not declare convergence prematurely -- the derivative-
        // free incumbent (the simplex pole) can hold still for several
        // evaluations while the model and geometry improve. The gradient proxy
        // is likewise floored while RHO has not reached RHOEND.
        const double rho = s.engine->rho();
        double grad_proxy = s.engine->gradient_proxy();
        if(running && rho > options.final_trust_radius)
            grad_proxy = std::max(grad_proxy, 1.0);

        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = grad_proxy,
            .step_size = running ? rho : 0.0,
            .objective_change = running ? rho : 0.0,
            .improved = improved,
            .is_null_step = !running,
            .constraint_violation = detail::constraint_violation(s.c_eq, s.c_ineq),
            .policy_status = policy_status,
        };
    }

    template <typename P>
    void reset(state_type<P>& s, const Eigen::VectorXd& x0)
    {
        reset_clear(s, x0);
    }

    template <typename P>
    void reset_clear(state_type<P>& s, const Eigen::VectorXd& x0)
    {
        s.x = detail::project(x0, s.lower, s.upper);
        s.iteration = 0;
        s.parmu = 0.0;

        const int n = static_cast<int>(s.x.size());
        double rhobeg = auto_trust_radius(s.lower, s.upper, n);
        double rhoend = options.final_trust_radius;
        make_evaluator_and_init(s, rhobeg, rhoend);
        sync_from_engine(s);
    }

private:
    // Auto initial trust radius: 10% of the largest finite bound range, or 1.0
    // when no coordinate is finitely bounded (matching the reference rho0=1.0).
    double auto_trust_radius(const Eigen::VectorXd& lower,
                             const Eigen::VectorXd& upper, int n) const
    {
        double h = options.initial_trust_radius;
        if(h > 0.0)
            return h;
        double max_range = 0.0;
        for(int i = 0; i < n; ++i)
        {
            double range_i = upper[i] - lower[i];
            if(std::isfinite(range_i))
                max_range = std::max(max_range, range_i);
        }
        return (max_range > 0.0) ? 0.1 * max_range : 1.0;
    }

    // Build the Powell-convention evaluator (captures the problem by pointer)
    // and initialize the engine. The evaluator also refreshes s.c_eq / s.c_ineq
    // so basic_solver's constraint_violation reporting sees the latest point.
    template <typename P>
    void make_evaluator_and_init(state_type<P>& s, double rhobeg, double rhoend)
    {
        const P* problem = s.problem;
        const int n = static_cast<int>(s.x.size());
        const int n_eq = s.n_eq;
        const int n_ineq = s.n_ineq;
        Eigen::VectorXd lower = s.lower;
        Eigen::VectorXd upper = s.upper;

        auto eval = [problem, n, n_eq, n_ineq, lower, upper]
            (const Eigen::VectorXd& x, Eigen::VectorXd& con) -> double
        {
            const int m_total = n_eq + n_ineq;
            Eigen::VectorXd c_full(std::max(m_total, 0));
            if(m_total > 0)
                problem->constraints(x, c_full);

            int idx = 0;
            for(int i = 0; i < n_ineq; ++i)
                con(idx++) = c_full[n_eq + i];
            for(int j = 0; j < n_eq; ++j)
            {
                con(idx++) = c_full[j];
                con(idx++) = -c_full[j];
            }
            for(int i = 0; i < n; ++i)
            {
                if(std::isfinite(lower[i]))
                    con(idx++) = x[i] - lower[i];
                if(std::isfinite(upper[i]))
                    con(idx++) = upper[i] - x[i];
            }
            return problem->value(x);
        };

        s.engine->init(n, s.m_powell, s.x, s.lower, s.upper, rhobeg, rhoend,
                       std::move(eval));
    }

    // Copy the engine incumbent into the policy state and refresh the split
    // constraint vectors used for violation reporting.
    template <typename P>
    void sync_from_engine(state_type<P>& s)
    {
        s.x = s.engine->x();
        s.objective_value = s.engine->objective_value();
        s.parmu = s.engine->parmu();

        const int m_total = s.n_eq + s.n_ineq;
        if(m_total > 0)
        {
            Eigen::VectorXd c_full(m_total);
            s.problem->constraints(s.x, c_full);
            s.c_eq = c_full.head(s.n_eq);
            s.c_ineq = c_full.tail(s.n_ineq);
        }
        else
        {
            s.c_eq.resize(0);
            s.c_ineq.resize(0);
        }
    }
};

}

#endif
