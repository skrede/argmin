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

#include "argmin/solver/options.h"

#include "argmin/detail/cobyla_simplex.h"
#include "argmin/detail/bound_projection.h"
#include "argmin/detail/cobyla_policy_bridge.h"
#include "argmin/detail/cobyla_powell_evaluator.h"

#include "argmin/options/derivative_free_defaults.h"

#include "argmin/result/step_result.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <memory>
#include <cstdint>

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
        Scalar initial_trust_radius{Scalar(0.0)};   // 0 means auto (10% of max bound range, at most half the min, else 1.0)
        Scalar final_trust_radius{derivative_free_defaults<Scalar>::final_trust_radius};
        std::uint16_t stall_window{50};
        Scalar feasibility_gate{derivative_free_defaults<Scalar>::cobyla_feasibility_gate};
    };

    options_type options{};

    template <typename P = void>
    struct state_type
    {
        state_type() : problem{nullptr}, objective_value{}, parmu{Scalar(0.0)}, n_eq{0}, n_ineq{0}, m_powell{0}, iteration{0}, clamped_evaluations{0}, invalid_coordinate{-1}, invalid_problem{false} {}

        const P* problem;
        Eigen::VectorX<Scalar> x;
        Eigen::VectorX<Scalar> lower;
        Eigen::VectorX<Scalar> upper;
        Eigen::VectorX<Scalar> c_eq;
        Eigen::VectorX<Scalar> c_ineq;
        Scalar objective_value;
        // Powell merit penalty. Adapts upward only through
        // barmu = (predicted objective increase) / (predicted reduction in max
        // violation) -- i.e. only when reaching feasibility costs objective.
        Scalar parmu;
        int n_eq;
        int n_ineq;
        int m_powell;
        std::uint32_t iteration;
        std::uint32_t clamped_evaluations;
        std::int32_t invalid_coordinate;
        bool invalid_problem;

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
        if(detail::cobyla_reject_invalid_box(s, x0))
            return s;
        s.x = detail::project(x0, s.lower, s.upper);
        s.n_eq = problem.num_equality();
        s.n_ineq = problem.num_inequality();
        s.m_powell = s.n_ineq + 2 * s.n_eq + detail::cobyla_finite_bound_count(s.lower, s.upper);
        Scalar rhobeg = detail::cobyla_auto_trust_radius(options.initial_trust_radius, s.lower, s.upper, n);
        Scalar rhoend = options.final_trust_radius;
        s.engine = std::make_shared<detail::cobyla_engine<Scalar>>();
        detail::cobyla_init_engine(s, rhobeg, rhoend);
        detail::cobyla_sync_from_engine(s);
        s.iteration = 0;
        return s;
    }

    template <typename P>
    step_result<Scalar> step(state_type<P>& s)
    {
        if(s.invalid_problem)
            return detail::cobyla_invalid_step_report<Scalar>(s);
        const Scalar old_f = s.objective_value;

        bool running = s.engine->advance();
        ++s.iteration;
        detail::cobyla_sync_from_engine(s);

        const bool improved = s.objective_value < old_f;
        return detail::cobyla_step_report(s, running, improved, options.final_trust_radius);
    }

    template <typename P>
    void reset(state_type<P>& s, Eigen::Ref<const Eigen::VectorX<Scalar>> x0)
    {
        reset_clear(s, x0);
    }

    template <typename P>
    void reset_clear(state_type<P>& s, Eigen::Ref<const Eigen::VectorX<Scalar>> x0)
    {
        if(detail::cobyla_reject_invalid_box(s, x0))
            return;
        s.x = detail::project(x0, s.lower, s.upper);
        s.iteration = 0;
        s.parmu = Scalar(0.0);

        const int n = static_cast<int>(s.x.size());
        Scalar rhobeg = detail::cobyla_auto_trust_radius(options.initial_trust_radius, s.lower, s.upper, n);
        Scalar rhoend = options.final_trust_radius;
        detail::cobyla_init_engine(s, rhobeg, rhoend);
        detail::cobyla_sync_from_engine(s);
    }
};

}

#endif
