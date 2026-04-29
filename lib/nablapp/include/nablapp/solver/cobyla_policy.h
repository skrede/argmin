#ifndef HPP_GUARD_NABLAPP_SOLVER_COBYLA_POLICY_H
#define HPP_GUARD_NABLAPP_SOLVER_COBYLA_POLICY_H

// COBYLA solver policy for basic_solver.
//
// Implements Powell's Constrained Optimization BY Linear Approximation.
// A trust-region derivative-free method that maintains a simplex of n+1
// points and builds linear models of the objective and all constraints
// by interpolation. Each step solves a linearised trust-region subproblem,
// evaluates the trial point, updates the simplex, and adjusts the trust
// radius based on model accuracy.
//
// Requires: objective<P,S> && constrained_values<P,S> && bound_constrained<P,S>.
// No gradient or constraint Jacobian is needed -- COBYLA uses only function
// and constraint value evaluations.
//
// Reference: Powell, M. J. D. (1994) "A direct search optimization
//            method that models the objective and constraint functions
//            by linear interpolation."
//            K&W 2e, Section 10.7 (derivative-free constrained optimization).

#include "nablapp/detail/cobyla_simplex.h"
#include "nablapp/detail/cobyla_trust_region.h"
#include "nablapp/detail/bound_projection.h"
#include "nablapp/detail/lagrangian.h"
#include "nablapp/result/step_result.h"
#include "nablapp/solver/options.h"

#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

namespace nablapp
{

struct cobyla_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = cobyla_policy;

    struct options_type
    {
        double initial_trust_radius{0.0};   // 0 means auto (10% of max bound range)
        double final_trust_radius{1e-8};
        double step_convergence_factor{1e-3};
        std::uint16_t stall_window{50};
        double feasibility_gate{1e-4};
    };

    options_type options{};

    template <typename P = void>
    struct state_type
    {
        static constexpr int M = [] {
            if constexpr(has_constraint_count<P>) return constraint_count_v<P>;
            else return dynamic_dimension;
        }();

        const P* problem{nullptr};
        Eigen::VectorXd x;
        Eigen::MatrixXd simplex;       // n x (n+1)
        Eigen::VectorXd f_simplex;     // objective at each vertex
        Eigen::MatrixXd c_simplex;     // all constraints at each vertex, m x (n+1)
        double rho{};
        double rho_end{};
        double objective_value{};
        // Adaptive merit penalty (Powell 1994 §5). Starts at zero and
        // adapts upward via barmu = prerec / prerem with the rule
        // parmu_new = max(parmu, 2*barmu) when parmu < 1.5*barmu. Used
        // by the trust-region inner loop, the actual/predicted reduction
        // comparison, and the find_best_point merit. Replaces the
        // pre-fix hardcoded penalty=10.0 (trust region + step()) and
        // 1e6 (find_best_point) flagged by static-audit C1.
        double parmu{0.0};
        Eigen::VectorXd c_eq;
        Eigen::VectorXd c_ineq;
        Eigen::VectorXd lower;
        Eigen::VectorXd upper;
        std::uint32_t iteration{0};
        int n_eq{0};
        int n_ineq{0};
        int m_total{0};
        int best_idx{0};

        std::optional<detail::cobyla_simplex_solver<double>> simplex_solver;
        std::optional<detail::cobyla_trust_region_solver<double>> trust_region_solver;
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
        s.m_total = s.n_eq + s.n_ineq;

        // Initial trust radius (same auto logic as BOBYQA)
        double h = options.initial_trust_radius;
        if(h <= 0.0)
        {
            double max_range = 0.0;
            for(int i = 0; i < n; ++i)
            {
                double range_i = s.upper[i] - s.lower[i];
                if(std::isfinite(range_i))
                    max_range = std::max(max_range, range_i);
            }
            h = (max_range > 0.0) ? 0.1 * max_range : 1.0;
        }
        s.rho = h;
        s.rho_end = options.final_trust_radius;

        // Pre-allocate stateful solvers
        s.simplex_solver.emplace(n, s.m_total);
        s.trust_region_solver.emplace(n, s.m_total, s.n_eq);

        // Build initial simplex
        s.simplex = detail::build_simplex(s.x, s.rho, s.lower, s.upper);

        // Evaluate objective and constraints at all simplex vertices
        s.f_simplex.resize(n + 1);
        s.c_simplex.resize(s.m_total, n + 1);

        for(int i = 0; i <= n; ++i)
        {
            s.f_simplex[i] = s.problem->value(s.simplex.col(i));
            if(s.m_total > 0)
            {
                Eigen::VectorXd c(s.m_total);
                s.problem->constraints(s.simplex.col(i), c);
                s.c_simplex.col(i) = c;
            }
        }

        // Find best point using a merit combining objective and feasibility
        s.best_idx = find_best_point(s);
        s.x = s.simplex.col(s.best_idx);
        s.objective_value = s.f_simplex[s.best_idx];

        // Populate c_eq / c_ineq for basic_solver constraint_violation
        populate_constraint_state(s);

        s.iteration = 0;
        return s;
    }

    template <typename P>
    step_result<double> step(state_type<P>& s)
    {
        const int n = static_cast<int>(s.x.size());
        double old_f = s.objective_value;

        // Build linear models from current simplex using pre-allocated solver
        const auto& models = s.simplex_solver->build_models(
            s.simplex, s.f_simplex, s.c_simplex, s.best_idx);

        // Compute linearised constraint offsets at best point
        Eigen::VectorXd constraint_offsets = compute_constraint_offsets(s);

        // Solve trust-region subproblem using pre-allocated solver.
        // Adaptive parmu drives the inner projected-gradient penalty;
        // the solver internally floors at max(parmu, 1) so iter-0 with
        // parmu = 0 still produces a feasible-leaning step.
        Eigen::VectorXd d = s.trust_region_solver->solve(
            models.objective_gradient,
            models.constraint_gradients,
            constraint_offsets,
            s.n_eq,
            s.rho,
            s.lower,
            s.upper,
            s.x,
            s.parmu);

        double d_norm = d.norm();

        // Convergence: step too small relative to trust radius
        if(d_norm < s.rho * options.step_convergence_factor &&
           s.rho <= s.rho_end)
        {
            return make_converged_result(s, models.objective_gradient.norm());
        }

        // Trial point
        Eigen::VectorXd x_trial = detail::project(
            (s.x + d).eval(), s.lower, s.upper);
        double f_trial = s.problem->value(x_trial);

        Eigen::VectorXd c_trial(s.m_total);
        if(s.m_total > 0)
            s.problem->constraints(x_trial, c_trial);

        // Powell adaptive parmu (Powell 1994 §5). Compute predicted
        // reductions in objective (prerec) and constraint violation
        // (prerem); set barmu = prerec / prerem when prerem > 0; if the
        // current parmu is below 1.5 * barmu, raise it to 2 * barmu so
        // the merit comparison and find_best_point honour the new
        // constraint-vs-objective trade-off. Then re-find the best
        // simplex vertex under the new parmu, since the merit ranking
        // can change. Pre-fix this routine used a constant parmu = 10
        // that trapped HS024 at f = -0.30 instead of f* = -1.0
        // (static-audit C1).
        double viol_old = current_violation(s);
        double viol_trial = compute_violation(c_trial, s.n_eq, s.n_ineq);

        double prerec = -models.objective_gradient.dot(d);
        double linearised_viol_d = linearised_violation_at(
            models.constraint_gradients, constraint_offsets, s.n_eq, d);
        double prerem = viol_old - linearised_viol_d;

        if(prerem > 0.0)
        {
            double barmu = prerec / prerem;
            if(s.parmu < 1.5 * barmu)
                s.parmu = std::max(s.parmu, 2.0 * barmu);
        }

        // Re-find best vertex under updated parmu (merit ranking may shift).
        s.best_idx = find_best_point(s);

        double penalty = s.parmu;
        double actual = (old_f + penalty * viol_old) - (f_trial + penalty * viol_trial);
        double predicted = prerec + penalty * prerem;

        // Update trust radius
        s.rho = detail::compute_rho_update(s.rho, s.rho_end, actual, predicted);

        // Select vertex to replace and update simplex
        int k = detail::select_replacement_vertex(
            s.simplex, s.f_simplex, s.best_idx, x_trial);
        detail::replace_vertex(s.simplex, s.f_simplex, s.c_simplex, k, x_trial, f_trial, c_trial);

        // Update best point
        bool improved = false;
        int new_best = find_best_point(s);
        if(new_best != s.best_idx || f_trial < old_f)
        {
            s.best_idx = new_best;
            s.x = s.simplex.col(s.best_idx);
            s.objective_value = s.f_simplex[s.best_idx];
            improved = (s.objective_value < old_f);
        }

        // Geometry maintenance using pre-allocated solver
        int geo_idx = s.simplex_solver->check_geometry(s.simplex, s.best_idx, s.rho);
        if(geo_idx >= 0)
            repair_geometry(s, geo_idx);

        populate_constraint_state(s);
        ++s.iteration;

        // Derivative-free convergence signalling (same pattern as BOBYQA)
        double obj_change = s.objective_value - old_f;
        double effective_step = improved ? d_norm : s.rho;
        double effective_change = improved ? obj_change : s.rho;

        double grad_proxy = models.objective_gradient.norm();
        if(s.rho > s.rho_end)
            grad_proxy = std::max(grad_proxy, 1.0);

        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = grad_proxy,
            .step_size = effective_step,
            .objective_change = effective_change,
            .improved = improved,
            .constraint_violation = detail::constraint_violation(s.c_eq, s.c_ineq),
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
        const int n = static_cast<int>(x0.size());
        s.x = detail::project(x0, s.lower, s.upper);
        s.iteration = 0;

        s.simplex = detail::build_simplex(s.x, s.rho, s.lower, s.upper);

        for(int i = 0; i <= n; ++i)
        {
            s.f_simplex[i] = s.problem->value(s.simplex.col(i));
            if(s.m_total > 0)
            {
                Eigen::VectorXd c(s.m_total);
                s.problem->constraints(s.simplex.col(i), c);
                s.c_simplex.col(i) = c;
            }
        }

        s.best_idx = find_best_point(s);
        s.x = s.simplex.col(s.best_idx);
        s.objective_value = s.f_simplex[s.best_idx];
        populate_constraint_state(s);
    }

private:

    // Find the best simplex vertex via Powell's adaptive merit
    // psi = f + parmu * viol. The merit weight is the same parmu used
    // by the trust-region inner loop and the actual/predicted reduction
    // comparison; when parmu has not yet adapted (early iterations,
    // value still small), a feasibility-first floor keeps an infeasible
    // vertex from winning over a feasible one purely on objective. Once
    // parmu adapts upward past the floor, the adaptive value takes over
    // and feasibility tradeoffs follow Powell's recipe.
    //
    // Reference: Powell 1994 §5 (merit function); static-audit C1.
    template <typename P>
    static int find_best_point(const state_type<P>& s)
    {
        const int nv = static_cast<int>(s.f_simplex.size());
        int best = 0;
        double best_merit = std::numeric_limits<double>::infinity();
        // Floor at 1e3 -- enough to keep a small-violation infeasible
        // vertex from beating a feasible one on a slightly higher f,
        // small enough that any meaningful adapted parmu (typical
        // values 10-1e6 on HS-class problems) takes over.
        const double weight = std::max(s.parmu, 1e3);

        for(int i = 0; i < nv; ++i)
        {
            double viol = 0.0;
            if(s.m_total > 0)
                viol = compute_violation(s.c_simplex.col(i), s.n_eq, s.n_ineq);

            double merit = s.f_simplex[i] + weight * viol;
            if(merit < best_merit)
            {
                best_merit = merit;
                best = i;
            }
        }
        return best;
    }

    // Compute constraint violation from a combined constraint vector.
    static double compute_violation(const Eigen::VectorXd& c, int n_eq, int n_ineq)
    {
        Eigen::VectorXd ceq = c.head(n_eq);
        Eigen::VectorXd cineq = c.tail(n_ineq);
        return detail::constraint_violation(ceq, cineq);
    }

    // Current violation at the best point.
    template <typename P>
    static double current_violation(const state_type<P>& s)
    {
        return detail::constraint_violation(s.c_eq, s.c_ineq);
    }

    // Compute linearised constraint offsets (constraint values at best point).
    template <typename P>
    static Eigen::VectorXd compute_constraint_offsets(const state_type<P>& s)
    {
        if(s.m_total == 0)
            return Eigen::VectorXd{};
        return s.c_simplex.col(s.best_idx);
    }

    // Compute linearised violation at displacement d.
    static double linearised_violation_at(
        const Eigen::MatrixXd& constraint_gradients,
        const Eigen::VectorXd& constraint_offsets,
        int n_eq,
        const Eigen::VectorXd& d)
    {
        return detail::linearised_violation(constraint_gradients, constraint_offsets, n_eq, d);
    }

    // Populate c_eq and c_ineq from the best point's constraint values.
    template <typename P>
    static void populate_constraint_state(state_type<P>& s)
    {
        if(s.m_total > 0)
        {
            Eigen::VectorXd c = s.c_simplex.col(s.best_idx);
            s.c_eq = c.head(s.n_eq);
            s.c_ineq = c.tail(s.n_ineq);
        }
        else
        {
            s.c_eq.resize(0);
            s.c_ineq.resize(0);
        }
    }

    // Repair degenerate simplex geometry at vertex geo_idx.
    template <typename P>
    static void repair_geometry(state_type<P>& s, int geo_idx)
    {
        Eigen::VectorXd geo_pt = detail::geometry_improving_point(
            s.simplex, s.best_idx, geo_idx, s.rho, s.lower, s.upper);
        double geo_f = s.problem->value(geo_pt);

        Eigen::VectorXd geo_c(s.m_total);
        if(s.m_total > 0)
            s.problem->constraints(geo_pt, geo_c);

        detail::replace_vertex(s.simplex, s.f_simplex, s.c_simplex,
                               geo_idx, geo_pt, geo_f, geo_c);
    }

    template <typename P>
    static step_result<double> make_converged_result(const state_type<P>& s, double grad_norm)
    {
        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = grad_norm,
            .step_size = 0.0,
            .objective_change = 0.0,
            .improved = false,
            .constraint_violation = detail::constraint_violation(s.c_eq, s.c_ineq),
        };
    }
};

}

#endif
