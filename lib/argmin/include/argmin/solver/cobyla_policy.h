#ifndef HPP_GUARD_ARGMIN_SOLVER_COBYLA_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_COBYLA_POLICY_H

// COBYLA solver policy for step_budget_solver.
//
// Implements Powell's Constrained Optimization BY Linear Approximation: a
// derivative-free trust-region method that maintains a simplex of n+1 points,
// builds linear models of the objective and every constraint by interpolation,
// solves a two-stage linear trust-region subproblem for the step, and adapts a
// merit penalty toward feasibility only when reaching feasibility costs
// objective. The heavy lifting lives in detail::cobyla_engine, a faithful port
// of Powell's cobylb driver; this policy adapts the argmin problem interface to
// the engine and threads the incumbent through step_budget_solver.
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
#include "argmin/options/derivative_free_defaults.h"
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

template <scalar Scalar = double>
struct cobyla_policy
{
    using scalar_type = Scalar;

    template <int M>
    using rebind = cobyla_policy<Scalar>;

    struct options_type
    {
        Scalar initial_trust_radius{Scalar(0.0)};   // 0 means auto (10% of max bound range, else 1.0)
        Scalar final_trust_radius{derivative_free_defaults<Scalar>::final_trust_radius};
        std::uint16_t stall_window{50};
        Scalar feasibility_gate{derivative_free_defaults<Scalar>::cobyla_feasibility_gate};
    };

    options_type options{};

    template <typename P = void>
    struct state_type
    {
        const P* problem{nullptr};
        Eigen::VectorX<Scalar> x;
        Eigen::VectorX<Scalar> lower;
        Eigen::VectorX<Scalar> upper;
        Eigen::VectorX<Scalar> c_eq;
        Eigen::VectorX<Scalar> c_ineq;
        Scalar objective_value{};
        // Powell merit penalty. Adapts upward only through
        // barmu = (predicted objective increase) / (predicted reduction in max
        // violation) -- i.e. only when reaching feasibility costs objective.
        Scalar parmu{Scalar(0.0)};
        int n_eq{0};
        int n_ineq{0};
        int m_powell{0};
        std::uint32_t iteration{0};

        // The engine is heap-owned so state moves leave its internal pointers
        // (captured by the evaluator) valid.
        std::shared_ptr<detail::cobyla_engine<Scalar>> engine;
    };

    template <typename Problem, typename Convergence>
        requires objective<Problem, Scalar> && constrained_values<Problem, Scalar> && bound_constrained<Problem, Scalar>
    state_type<Problem> init(const Problem& problem, const Eigen::VectorX<Scalar>& x0,
                    const solver_options<Convergence>& opts, const options_type& policy_opts)
    {
        options = policy_opts;
        return init(problem, x0, opts);
    }

    template <typename Problem, typename Convergence = default_convergence<scalar_type>>
        requires objective<Problem, Scalar> && constrained_values<Problem, Scalar> && bound_constrained<Problem, Scalar>
    state_type<Problem> init(const Problem& problem, const Eigen::VectorX<Scalar>& x0,
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

        Scalar rhobeg = auto_trust_radius(s.lower, s.upper, n);
        Scalar rhoend = options.final_trust_radius;

        s.engine = std::make_shared<detail::cobyla_engine<Scalar>>();
        make_evaluator_and_init(s, rhobeg, rhoend);

        sync_from_engine(s);
        s.iteration = 0;
        return s;
    }

    template <typename P>
    step_result<Scalar> step(state_type<P>& s)
    {
        const Scalar old_f = s.objective_value;

        bool running = s.engine->advance();
        ++s.iteration;
        sync_from_engine(s);

        const bool improved = s.objective_value < old_f;

        std::optional<solver_status> policy_status{};
        if(!running)
            policy_status = s.engine->status();

        // COBYLA owns its own termination: it iterates until the trust radius
        // RHO contracts to RHOEND and then reports xtol_reached via
        // policy_status. Until then the reported progress magnitudes are kept
        // at the trust-radius scale so step_budget_solver's step / objective / stall
        // criteria do not declare convergence prematurely -- the derivative-
        // free incumbent (the simplex pole) can hold still for several
        // evaluations while the model and geometry improve. The gradient proxy
        // is likewise floored while RHO has not reached RHOEND.
        const Scalar rho = s.engine->rho();
        Scalar grad_proxy = s.engine->gradient_proxy();
        if(running && rho > options.final_trust_radius)
            grad_proxy = std::max(grad_proxy, Scalar(1.0));

        return step_result<Scalar>{
            .objective_value = s.objective_value,
            .gradient_norm = grad_proxy,
            .step_size = running ? rho : Scalar(0.0),
            .objective_change = running ? rho : Scalar(0.0),
            .improved = improved,
            .is_null_step = !running,
            .constraint_violation = detail::constraint_violation(s.c_eq, s.c_ineq),
            .policy_status = policy_status,
        };
    }

    template <typename P>
    void reset(state_type<P>& s, Eigen::Ref<const Eigen::VectorX<Scalar>> x0)
    {
        reset_clear(s, x0);
    }

    template <typename P>
    void reset_clear(state_type<P>& s, Eigen::Ref<const Eigen::VectorX<Scalar>> x0)
    {
        s.x = detail::project(x0, s.lower, s.upper);
        s.iteration = 0;
        s.parmu = Scalar(0.0);

        const int n = static_cast<int>(s.x.size());
        Scalar rhobeg = auto_trust_radius(s.lower, s.upper, n);
        Scalar rhoend = options.final_trust_radius;
        make_evaluator_and_init(s, rhobeg, rhoend);
        sync_from_engine(s);
    }

private:
    // Auto initial trust radius: 10% of the largest finite bound range, or 1.0
    // when no coordinate is finitely bounded (matching the reference rho0=1.0).
    Scalar auto_trust_radius(const Eigen::VectorX<Scalar>& lower,
                             const Eigen::VectorX<Scalar>& upper, int n) const
    {
        Scalar h = options.initial_trust_radius;
        if(h > Scalar(0.0))
            return h;
        Scalar max_range = Scalar(0.0);
        for(int i = 0; i < n; ++i)
        {
            Scalar range_i = upper[i] - lower[i];
            if(std::isfinite(range_i))
                max_range = std::max(max_range, range_i);
        }
        return (max_range > Scalar(0.0)) ? Scalar(0.1) * max_range : Scalar(1.0);
    }

    // Build the Powell-convention evaluator (captures the problem by pointer)
    // and initialize the engine. The evaluator also refreshes s.c_eq / s.c_ineq
    // so step_budget_solver's constraint_violation reporting sees the latest point.
    template <typename P>
    void make_evaluator_and_init(state_type<P>& s, Scalar rhobeg, Scalar rhoend)
    {
        const P* problem = s.problem;
        const int n = static_cast<int>(s.x.size());
        const int n_eq = s.n_eq;
        const int n_ineq = s.n_ineq;
        Eigen::VectorX<Scalar> lower = s.lower;
        Eigen::VectorX<Scalar> upper = s.upper;

        auto eval = [problem, n, n_eq, n_ineq, lower, upper]
            (const Eigen::VectorX<Scalar>& x, Eigen::VectorX<Scalar>& con) -> Scalar
        {
            const int m_total = n_eq + n_ineq;
            Eigen::VectorX<Scalar> c_full(std::max(m_total, 0));
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
            Eigen::VectorX<Scalar> c_full(m_total);
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
