#ifndef HPP_GUARD_ARGMIN_SOLVER_AUGMENTED_LAGRANGIAN_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_AUGMENTED_LAGRANGIAN_POLICY_H

// Augmented Lagrangian (method of multipliers) wrapper policy.
//
// Converts any unconstrained or bound-constrained inner policy (e.g.,
// lbfgsb_policy, bobyqa_policy) into a constrained solver. Each outer
// step() performs a full inner solve on the augmented Lagrangian
// subproblem followed by a multiplier/penalty update. The inner
// subproblem satisfies differentiable + bound_constrained so that
// gradient-based and derivative-free inner solvers both work.
//
// Warm-start: the inner step_budget_solver is persisted across outer
// iterations, preserving compact_lbfgs curvature pairs via
// lbfgsb_policy::reset(). The subproblem's penalty parameter is
// stored as a pointer to the outer state's mu, so multiplier and
// penalty updates propagate automatically without reconstruction.
//
// Adaptive inner tolerance: follows the Conn-Gould-Toint (1991)
// schedule where inner tolerance loosens in early outer iterations
// and tightens as mu decreases (N&W Algorithm 17.4).
//
// Constrained inner solvers (the moving-asymptotes family) are also
// supported, by partial elimination (Bertsekas 1982, Section 4.2; the
// NLopt AUGLAG_EQ pattern): ONLY the equalities are lifted into the
// augmented objective, while the inequalities and the box are forwarded
// unmodified to the inner solver, which handles them natively. The inner
// solver is fed the equality-lifting subproblem view (eq_subproblem) via a
// compile-time concept-probe fork; the outer method of multipliers then
// updates the equality multipliers only.
//
// Convergence-guarantee scope for the constrained-inner composition. Under
// the standard augmented-Lagrangian assumptions -- the inner stopping
// tolerance driven to zero on the inner problem's KKT residual, finite box
// bounds, a constraint qualification (MFCQ) on the pass-through
// inequalities, and a constraint qualification at limit points -- the
// composition inherits a convergence guarantee when the inner solver is
// the globally convergent variant or its quadratic-penalty sibling, both of
// which converge to a KKT point of their (relaxed) inner subproblem. The
// plain moving-asymptotes inner solver carries NO such guarantee: the
// original 1987 method is a heuristic with no convergence theorem and can
// cycle, so composing it under the augmented Lagrangian is permitted but
// guarantee-free. Inner termination therefore keys on the inner KKT
// residual (step_result::kkt_residual), NOT the raw objective-gradient
// norm, which for the moving-asymptotes family is not a stationarity
// measure of the constrained inner subproblem.
//
// Reference: K&W Section 10.9, Algorithm 10.2;
//            N&W Section 17.4, Algorithm 17.4;
//            Conn, Gould, Toint (1991) "A globally convergent
//            augmented Lagrangian algorithm for optimization with
//            general constraints and simple bounds".

#include "argmin/detail/augmented_lagrangian.h"
#include "argmin/detail/lagrangian.h"
#include "argmin/detail/kkt_residual.h"
#include "argmin/detail/bound_projection.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/options.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/result/step_result.h"
#include "argmin/types.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>

namespace argmin
{

namespace detail
{

// Classifies an inner-solver state as constrained or bound-constrained by
// the structural presence of an inequality-constraint field. A constrained
// separable-approximation inner solver (moving-asymptotes and its globally
// convergent / quadratic-penalty variants) lifts the problem's inequality
// constraints into its own per-problem state; a bound-constrained inner
// solver (limited-memory BFGS for bounds, derivative-free quadratic
// interpolation) carries no such field. The augmented Lagrangian uses this
// signal to decide whether the inner solver handles inequalities natively,
// and therefore which synthetic subproblem view to feed it.
template <typename State>
concept inner_state_tracks_inequalities = requires(const State& s)
{
    { s.c_ineq };
};

}

template <typename InnerPolicy = lbfgsb_policy<>, int N = dynamic_dimension>
struct augmented_lagrangian_policy
{
    using scalar_type = typename InnerPolicy::scalar_type;

    template <int M>
    using rebind = augmented_lagrangian_policy<InnerPolicy, M>;

    struct options_type
    {
        scalar_type mu_init{scalar_type(0.1)};                     // Conn-Gould-Toint
        scalar_type mu_decrease{scalar_type(0.25)};                // Conn-Gould-Toint
        scalar_type mu_min{scalar_type(1e-6)};                     // Conn-Gould-Toint
        std::uint32_t inner_max_iterations{200};
        scalar_type constraint_tolerance{scalar_type(1e-6)};       // K&W 10.9
        scalar_type inner_gradient_tolerance{scalar_type(1e-6)};   // K&W 10.9
        scalar_type feasibility_progress{scalar_type(0.25)};       // N&W 17.4
        typename InnerPolicy::options_type inner_opts = {};

        // Persist the inner solver across outer iterations, preserving
        // compact_lbfgs curvature pairs via lbfgsb_policy::reset().
        bool warm_start_inner{true};

        // Conn-Gould-Toint (1991) / N&W Algorithm 17.4 adaptive inner
        // stopping-tolerance (omega_k) schedule. omega_0 = eta * mu^alpha;
        // thereafter omega tightens multiplicatively (omega *= mu^alpha) on
        // every multiplier update and is re-derived (omega = eta * mu^alpha)
        // whenever the penalty is reduced. Because mu < 1, both branches
        // shrink omega, so omega_k -> 0 even under multiplier-only progress.
        // Reference: N&W Algorithm 17.4, Step 2; CGT 1991 Section 3.
        scalar_type inner_tolerance_eta{scalar_type(0.1)};
        scalar_type inner_tolerance_alpha{scalar_type(0.9)};
        std::uint16_t stall_window{100};
    };

    options_type options{};

    template <typename P = void>
    struct state_type
    {
        static constexpr int M = [] {
            if constexpr(has_constraint_count<P>) return constraint_count_v<P>;
            else return dynamic_dimension;
        }();

        // Inner-solver dimension for the gradient bridge buffer.
        // P::problem_dimension when P is a real problem; falls back to
        // Eigen::Dynamic for state_type<void> (concept-satisfaction probe).
        static constexpr int g_tmp_buf_dim = [] {
            if constexpr(has_problem_dimension<P>) return P::problem_dimension;
            else return Eigen::Dynamic;
        }();

        const P* problem{nullptr};
        Eigen::Vector<scalar_type, N> x;
        scalar_type f{};
        Eigen::VectorX<scalar_type> c_eq;
        Eigen::VectorX<scalar_type> c_ineq;

        Eigen::VectorX<scalar_type> lambda_eq;
        Eigen::VectorX<scalar_type> lambda_ineq;

        scalar_type mu{};
        scalar_type prev_viol{};
        // Penalty value used by the most recent inner solve. Tracked so
        // step() can detect a mu change between outer iters and force a
        // cold-start: the inner solver's L-BFGS curvature pairs are
        // built from the *previous* penalty's augmented Lagrangian and
        // become a stale Hessian approximation once mu shrinks (CGT 1991
        // §3 prescribes a fresh inner state after each penalty
        // reduction). Multiplier-only updates leave mu fixed and are
        // small enough that warm-starting is OK.
        std::optional<scalar_type> mu_at_last_inner_solve{};

        // Adaptive inner stopping tolerance (omega_k) of the CGT/N&W
        // Algorithm 17.4 two-branch schedule. Persisted across outer
        // iterations so the schedule can tighten multiplicatively on
        // multiplier updates (penalty pinned) and re-derive from the
        // penalty regime on penalty reductions -- driving omega_k -> 0
        // even when the penalty parameter never moves. Lazily initialized
        // on the first step from the initial penalty.
        std::optional<scalar_type> omega{};

        Eigen::Vector<scalar_type, N> lower;
        Eigen::Vector<scalar_type, N> upper;

        int n_eq = 0;
        int n_ineq = 0;
        std::uint32_t outer_iter = 0;

        options_type opts;

        // Synthetic subproblem for the inner solver.
        // Stores const P* and penalty state; satisfies differentiable + bound_constrained.
        // Penalty parameter is stored as a pointer to outer state's mu, so
        // multiplier/penalty updates propagate automatically without reconstruction.
        struct subproblem
        {
            enum : int { problem_dimension = N };
            const P* outer;
            int dim;
            const Eigen::Vector<scalar_type, N>* lo;
            const Eigen::Vector<scalar_type, N>* hi;
            const Eigen::VectorX<scalar_type>* lam_eq;
            const Eigen::VectorX<scalar_type>* lam_ineq;
            const scalar_type* pen;
            // Pointers to the boxed per-inner-iter buffers; mutated by
            // value() / gradient() through the indirection. Eliminates the
            // O(outer * inner * m) c_all / J_all / g_tmp allocations the
            // pre-fix subproblem incurred.
            Eigen::VectorX<scalar_type>* c_all_buf;
            Eigen::MatrixX<scalar_type>* J_all_buf;
            Eigen::Vector<scalar_type, g_tmp_buf_dim>* g_tmp_buf;
            int neq, nineq;

            [[nodiscard]] int dimension() const { return dim; }

            [[nodiscard]] scalar_type value(const Eigen::Vector<scalar_type, N>& x) const
            {
                scalar_type fval = outer->value(x);
                const int m = neq + nineq;
                if(m > 0)
                    outer->constraints(x, *c_all_buf);
                return detail::augmented_lagrangian_value(
                    fval,
                    c_all_buf->head(neq),
                    c_all_buf->tail(nineq),
                    *lam_eq, *lam_ineq, *pen);
            }

            void gradient(const Eigen::Vector<scalar_type, N>& x,
                          Eigen::Vector<scalar_type, N>& g) const
            {
                constexpr int PD = P::problem_dimension;
                if constexpr(N == PD)
                {
                    outer->gradient(x, g);
                }
                else
                {
                    outer->gradient(Eigen::Vector<scalar_type, N>(x), *g_tmp_buf);
                    g = *g_tmp_buf;
                }
                const int m = neq + nineq;
                if(m > 0)
                {
                    outer->constraints(x, *c_all_buf);
                    outer->constraint_jacobian(x, *J_all_buf);
                }
                // In-place mat-vec gradient (AL9). topRows / bottomRows /
                // head / tail are passed as Eigen expressions that the
                // helper consumes via DenseBase template deduction --
                // zero materialization.
                detail::augmented_lagrangian_gradient_inplace(
                    g,
                    J_all_buf->topRows(neq),
                    J_all_buf->bottomRows(nineq),
                    c_all_buf->head(neq),
                    c_all_buf->tail(nineq),
                    *lam_eq, *lam_ineq, *pen);
            }

            [[nodiscard]] Eigen::Vector<scalar_type, N> lower_bounds() const { return *lo; }
            [[nodiscard]] Eigen::Vector<scalar_type, N> upper_bounds() const { return *hi; }
        };

        // Equality-lifting synthetic subproblem for a constrained inner
        // solver (moving-asymptotes family). This is the partial-elimination
        // (Bertsekas 1982, Section 4.2) / NLopt AUGLAG_EQ view: ONLY the
        // equalities are absorbed into the augmented objective (quadratic
        // penalty + Hestenes-Powell multiplier term over the equalities),
        // while the inequalities and the box are forwarded UNMODIFIED so the
        // constrained inner solver handles them natively with its own
        // guarantees intact. It reports num_equality() -> 0 (the equalities
        // are gone from the constraint set, folded into value/gradient) and
        // num_inequality() -> the outer inequality count, so it satisfies the
        // constrained concept the inner solver static_asserts.
        //
        // Holds the same raw pointers into outer state as `subproblem` and is
        // re-seeded identically at the top of every step() (move-safety).
        struct eq_subproblem
        {
            enum : int { problem_dimension = N };
            const P* outer;
            int dim;
            const Eigen::Vector<scalar_type, N>* lo;
            const Eigen::Vector<scalar_type, N>* hi;
            const Eigen::VectorX<scalar_type>* lam_eq;
            const Eigen::VectorX<scalar_type>* lam_ineq;
            const scalar_type* pen;
            Eigen::VectorX<scalar_type>* c_all_buf;
            Eigen::MatrixX<scalar_type>* J_all_buf;
            Eigen::Vector<scalar_type, g_tmp_buf_dim>* g_tmp_buf;
            int neq, nineq;

            [[nodiscard]] int dimension() const { return dim; }

            [[nodiscard]] scalar_type value(const Eigen::Vector<scalar_type, N>& x) const
            {
                scalar_type fval = outer->value(x);
                if(neq + nineq > 0)
                    outer->constraints(x, *c_all_buf);
                // Augment ONLY the equalities. The empty inequality block
                // (tail(0)) zeroes the inequality contribution: the helper's
                // inequality loop runs over c_ineq.size() == 0 iterations and
                // never touches lam_ineq.
                return detail::augmented_lagrangian_value(
                    fval,
                    c_all_buf->head(neq),
                    c_all_buf->tail(0),
                    *lam_eq, *lam_ineq, *pen);
            }

            void gradient(const Eigen::Vector<scalar_type, N>& x,
                          Eigen::Vector<scalar_type, N>& g) const
            {
                constexpr int PD = P::problem_dimension;
                if constexpr(N == PD)
                {
                    outer->gradient(x, g);
                }
                else
                {
                    outer->gradient(Eigen::Vector<scalar_type, N>(x), *g_tmp_buf);
                    g = *g_tmp_buf;
                }
                if(neq > 0)
                {
                    outer->constraints(x, *c_all_buf);
                    outer->constraint_jacobian(x, *J_all_buf);
                }
                // Only the equality rows contribute; the empty inequality
                // block leaves J_ineq unread (guarded by c_ineq.size() > 0).
                detail::augmented_lagrangian_gradient_inplace(
                    g,
                    J_all_buf->topRows(neq),
                    J_all_buf->topRows(0),
                    c_all_buf->head(neq),
                    c_all_buf->tail(0),
                    *lam_eq, *lam_ineq, *pen);
            }

            // Forward ONLY the outer inequalities (argmin convention: c >= 0
            // feasible). outer->constraints writes neq + nineq entries with
            // the equalities first, so the inequalities are the trailing
            // nineq entries.
            void constraints(const Eigen::Vector<scalar_type, N>& x,
                             Eigen::VectorX<scalar_type>& c) const
            {
                if(neq + nineq > 0)
                    outer->constraints(x, *c_all_buf);
                c = c_all_buf->tail(nineq);
            }

            void constraint_jacobian(const Eigen::Vector<scalar_type, N>& x,
                                     Eigen::MatrixX<scalar_type>& J) const
            {
                if(neq + nineq > 0)
                    outer->constraint_jacobian(x, *J_all_buf);
                J = J_all_buf->bottomRows(nineq);
            }

            [[nodiscard]] int num_equality() const { return 0; }
            [[nodiscard]] int num_inequality() const { return nineq; }

            [[nodiscard]] Eigen::Vector<scalar_type, N> lower_bounds() const { return *lo; }
            [[nodiscard]] Eigen::Vector<scalar_type, N> upper_bounds() const { return *hi; }
        };

        using rebound_inner = typename InnerPolicy::template rebind<N>;

        // AUGLAG_EQ concept-probe fork. A constrained inner solver (the
        // moving-asymptotes family) tracks the problem's inequalities in its
        // per-problem state; a bound-constrained inner solver does not. When
        // the inner solver is constrained, lift ONLY the equalities into the
        // augmented objective and pass the inequalities + box straight
        // through (eq_subproblem); when it is bound-constrained, fold ALL
        // constraints into the augmented objective (subproblem).
        static constexpr bool lift_equalities_only = []
        {
            if constexpr(requires {
                typename rebound_inner::template state_type<eq_subproblem>; })
                return detail::inner_state_tracks_inequalities<
                    typename rebound_inner::template state_type<eq_subproblem>>;
            else
                return false;
        }();

        using active_subproblem =
            std::conditional_t<lift_equalities_only, eq_subproblem, subproblem>;

        // Heap-boxed inner context.
        //
        // The subproblem stores raw pointers to its evaluation buffers and to
        // outer state (multipliers, penalty, bounds), and the persisted inner
        // solver caches the address of the subproblem. step_budget_solver's move is
        // an unconditional noexcept memberwise move, so holding these directly
        // in the state would leave every such pointer dangling after a move.
        //
        // Boxing the subproblem, its buffers, and the inner solver in one heap
        // node keeps their addresses stable across a move: the node is reached
        // through a single owning pointer, so a memberwise move transfers
        // ownership and every address inside the node survives. The subproblem
        // pointers that must address genuine outer-state fields (mu,
        // multipliers, bounds) -- which a memberwise move does relocate -- are
        // re-seeded against the live state at the top of every step().
        struct al_inner_context
        {
            // Pre-allocated subproblem evaluation buffers. Mutated per-inner-
            // iter by subproblem::value / gradient via pointers held in the
            // subproblem struct; storing them here lets the subproblem remain
            // a stateless view while inner f / g calls stay alloc-free.
            // c_all_buf is sized (m_eq + m_ineq); J_all_buf is sized
            // (m_eq + m_ineq, dim); g_tmp_buf is the dimension-bridge
            // fast-path temporary used when the inner solver's compile-time N
            // differs from the outer problem's.
            Eigen::VectorX<scalar_type> c_all_buf;
            Eigen::MatrixX<scalar_type> J_all_buf;
            Eigen::Vector<scalar_type, g_tmp_buf_dim> g_tmp_buf;

            // Synthetic subproblem for the inner solver, plus the persisted
            // inner solver itself (warm-started across outer iterations to
            // preserve compact_lbfgs curvature pairs).
            std::optional<active_subproblem> sub_storage;
            std::optional<step_budget_solver<rebound_inner, N, active_subproblem>> inner_solver;
        };

        std::unique_ptr<al_inner_context> ctx;
    };

    template <typename Problem, typename Convergence>
    state_type<Problem> init(const Problem& problem,
                             const Eigen::Vector<scalar_type, N>& x0,
                             const solver_options<Convergence>& solver_opts,
                             const options_type& policy_opts)
    {
        options = policy_opts;
        auto s = init(problem, x0, solver_opts);
        s.opts = policy_opts;
        s.mu = policy_opts.mu_init;
        s.prev_viol = detail::constraint_violation(s.c_eq, s.c_ineq);
        return s;
    }

    template <typename Problem, typename Convergence = default_convergence>
    state_type<Problem> init(const Problem& problem,
                             const Eigen::Vector<scalar_type, N>& x0,
                             const solver_options<Convergence>& /*solver_opts*/)
    {
        static_assert(constrained<Problem, scalar_type>,
                      "augmented_lagrangian_policy requires constrained<Problem>");

        const int n = problem.dimension();
        state_type<Problem> s;
        s.problem = &problem;
        s.opts = options;

        s.n_eq = problem.num_equality();
        s.n_ineq = problem.num_inequality();

        s.x = x0;
        s.f = problem.value(x0);

        // Allocate the heap-boxed inner context and pre-size its per-inner-
        // iter subproblem buffers. Reused across all inner f / g calls; the
        // subproblem mutates them via pointers held in its struct. Boxing
        // keeps these addresses stable across a solver move.
        s.ctx = std::make_unique<typename state_type<Problem>::al_inner_context>();
        auto& ctx = *s.ctx;
        const int m = s.n_eq + s.n_ineq;
        ctx.c_all_buf.resize(m);
        ctx.J_all_buf.resize(m, n);
        ctx.g_tmp_buf.resize(n);

        if(m > 0)
            problem.constraints(x0, ctx.c_all_buf);
        s.c_eq = ctx.c_all_buf.head(s.n_eq);
        s.c_ineq = ctx.c_all_buf.tail(s.n_ineq);

        // Initialize multipliers to zero
        s.lambda_eq = Eigen::VectorX<scalar_type>::Zero(s.n_eq);
        s.lambda_ineq = Eigen::VectorX<scalar_type>::Zero(s.n_ineq);

        // Penalty
        s.mu = s.opts.mu_init;
        s.prev_viol = detail::constraint_violation(s.c_eq, s.c_ineq);
        s.outer_iter = 0;

        // Box bounds
        if constexpr(bound_constrained<Problem, scalar_type>)
        {
            s.lower = problem.lower_bounds();
            s.upper = problem.upper_bounds();
        }
        else
        {
            constexpr scalar_type inf = std::numeric_limits<scalar_type>::infinity();
            s.lower = Eigen::Vector<scalar_type, N>::Constant(n, -inf);
            s.upper = Eigen::Vector<scalar_type, N>::Constant(n, inf);
        }

        return s;
    }

    template <typename P>
    step_result<scalar_type> step(state_type<P>& s)
    {
        const int n = static_cast<int>(s.x.size());

        const scalar_type inner_grad_tol = s.opts.inner_gradient_tolerance;
        const std::uint32_t inner_max = s.opts.inner_max_iterations;
        const scalar_type mu_dec = s.opts.mu_decrease;
        const scalar_type mu_floor = s.opts.mu_min;
        const scalar_type feas_progress = s.opts.feasibility_progress;
        const scalar_type con_tol = s.opts.constraint_tolerance;

        // Evaluate constraints at current x
        {
            const int m = s.n_eq + s.n_ineq;
            Eigen::VectorX<scalar_type> c_all(m);
            if(m > 0)
                s.problem->constraints(s.x, c_all);
            s.c_eq = c_all.head(s.n_eq);
            s.c_ineq = c_all.tail(s.n_ineq);
        }

        // Build the subproblem on first call, then re-seed its self-
        // references at the top of every step. The buffer pointers address
        // node-owned storage (stable across a move); the outer-state pointers
        // (bounds, multipliers, penalty) address state fields that a
        // memberwise move relocates, so they must be refreshed against the
        // live state before the inner solve dereferences them. The subproblem
        // itself lives in the heap node, so its address -- cached by the
        // persisted inner solver -- stays valid across the move.
        using active_subproblem = typename state_type<P>::active_subproblem;
        constexpr bool lift_equalities_only = state_type<P>::lift_equalities_only;
        auto& ctx = *s.ctx;
        if(!ctx.sub_storage.has_value())
            ctx.sub_storage.emplace();
        {
            active_subproblem& sp = *ctx.sub_storage;
            sp.outer = s.problem;
            sp.dim = n;
            sp.lo = &s.lower;
            sp.hi = &s.upper;
            sp.lam_eq = &s.lambda_eq;
            sp.lam_ineq = &s.lambda_ineq;
            sp.pen = &s.mu;
            sp.c_all_buf = &ctx.c_all_buf;
            sp.J_all_buf = &ctx.J_all_buf;
            sp.g_tmp_buf = &ctx.g_tmp_buf;
            sp.neq = s.n_eq;
            sp.nineq = s.n_ineq;
        }

        // Adaptive inner stopping tolerance (Conn-Gould-Toint 1991; N&W
        // Algorithm 17.4). omega_k is carried in state across outer
        // iterations and updated by the two-branch schedule at the bottom of
        // step(): it tightens multiplicatively on multiplier updates and is
        // re-derived from the penalty regime on penalty reductions. Here we
        // only consume the current omega_k, lazily seeding it from the
        // initial penalty on the first outer iteration.
        //
        // omega_0 = eta * mu^alpha and, since mu < 1 (penalty term
        // 1/(2 mu) ||c||^2, mu decreasing), every subsequent update strictly
        // shrinks omega -- so omega_k -> 0 (floored at inner_grad_tol) even
        // when the penalty parameter is pinned across multiplier-only outer
        // iterations. This is the reverse of a schedule that would loosen as
        // mu shrinks; the inner solve must tighten as the penalty regime
        // tightens. CGT 1991 §3 prescribes alpha >= 0.9 to couple the inner
        // and outer iterations meaningfully.
        const scalar_type tol_eta = s.opts.inner_tolerance_eta;
        const scalar_type tol_alpha = s.opts.inner_tolerance_alpha;
        if(!s.omega.has_value())
            s.omega = static_cast<scalar_type>(tol_eta * std::pow(s.mu, tol_alpha));
        const scalar_type inner_tol = std::max(*s.omega, inner_grad_tol);

        solver_options<> inner_opts;
        inner_opts.max_iterations = inner_max;
        // The inner adaptive tolerance gates the inner solve on its KKT
        // residual, not the raw objective-gradient norm: gradient_tolerance_
        // criterion compares kkt_residual.value_or(gradient_norm) against
        // this threshold, and every supported inner solver (both the bound-
        // constrained solvers and the constrained moving-asymptotes family)
        // populates step_result::kkt_residual with a genuine stationarity
        // measure of its subproblem. For the constrained inner that measure
        // is the projected inner-KKT residual from the dual solve -- the
        // correct omega_k -> 0 quantity for the augmented-Lagrangian outer
        // schedule; the raw gradient norm would be the wrong measure there.
        std::get<gradient_tolerance_criterion>(inner_opts.convergence.criteria)
            .threshold = inner_tol;
        std::get<objective_tolerance_criterion>(inner_opts.convergence.criteria)
            .threshold = scalar_type(1e-15);
        std::get<step_tolerance_criterion>(inner_opts.convergence.criteria)
            .threshold = scalar_type(1e-15);

        // Warm-start or cold-start inner solver.
        //
        // Warm-start is correct only when the augmented Lagrangian has
        // not changed materially since the last inner solve. CGT 1991
        // §3 prescribes a fresh inner state after each penalty
        // reduction because the L-BFGS curvature pairs encode the
        // inverse Hessian of the *previous* penalty's AL function and
        // are a stale approximation once mu shrinks. Multiplier-only
        // updates leave mu fixed and the change is small (lambda shifts
        // by O(c/mu)), so warm-starting through them is OK.
        //
        // Force cold-start when mu changed since the last inner solve;
        // honor the user's warm_start_inner request only when mu is
        // stable (the multiplier-only-update branch).
        const bool warm_start_requested = s.opts.warm_start_inner;
        const bool mu_changed = s.mu_at_last_inner_solve.has_value()
                                && *s.mu_at_last_inner_solve != s.mu;
        const bool warm_start = warm_start_requested && !mu_changed;
        if(warm_start && ctx.inner_solver.has_value())
        {
            // Warm restart: preserves compact_lbfgs curvature pairs (S, Y, theta).
            // Reference: N&W Section 9.2 (compact L-BFGS warm-start rationale).
            ctx.inner_solver->reset(s.x);
        }
        else
        {
            // Cold start: construct fresh inner solver.
            ctx.inner_solver.emplace(*ctx.sub_storage, s.x, inner_opts);
        }
        s.mu_at_last_inner_solve = s.mu;

        auto inner_result = ctx.inner_solver->solve(inner_opts);

        // Extract solution from inner solver
        scalar_type f_old = s.f;
        Eigen::Vector<scalar_type, N> x_old = s.x;
        s.x = inner_result.x;
        s.f = s.problem->value(s.x);
        {
            const int m = s.n_eq + s.n_ineq;
            Eigen::VectorX<scalar_type> c_all(m);
            if(m > 0)
                s.problem->constraints(s.x, c_all);
            s.c_eq = c_all.head(s.n_eq);
            s.c_ineq = c_all.tail(s.n_ineq);
        }

        // Constraint violation (L-infinity).
        //
        // step_result.constraint_violation reports primal feasibility at the
        // L-infinity norm: max(||c_eq||_inf, ||max(-c_ineq, 0)||_inf). The
        // single-scalar form is scale-invariant per CGT 1991 Section 3 and
        // dimensionally consistent with kkt_residual (N&W 2e Definition 12.1).
        // augmented_lagrangian adopts this convention via the combined
        // measure; the outer-loop multiplier update machinery
        // (Hestenes-Powell) is unaffected.
        //
        // Reference: N&W 2e Definition 12.1 (KKT primal feasibility);
        //            Conn-Gould-Toint 1991 Section 3 (scale invariance of
        //            feasibility-progress gates).
        // Reported feasibility is always the full (equality + inequality)
        // residual. The feasibility-progress gate that drives the outer
        // multiplier/penalty update, however, must track only the
        // constraints the outer loop owns: on the AUGLAG_EQ branch the
        // inequalities are the inner solver's responsibility, so the gate
        // keys on the equality residual alone.
        scalar_type viol = detail::primal_feasibility_inf(s.c_eq, s.c_ineq);
        scalar_type gate_viol;
        if constexpr(lift_equalities_only)
        {
            const Eigen::VectorX<scalar_type> empty_ineq(0);
            gate_viol = detail::primal_feasibility_inf(s.c_eq, empty_ineq);
        }
        else
        {
            gate_viol = viol;
        }

        scalar_type step_norm = (s.x - x_old).norm();

        // Actual augmented Lagrangian gradient norm at current iterate,
        // and KKT residual via the outer-loop multiplier estimates
        // (s.lambda_eq, s.lambda_ineq). For derivative-free inner
        // solvers we have no objective gradient at hand and leave the
        // KKT residual unset; the convergence criteria then fall back
        // to gradient_norm.value_or(step_norm).
        //
        // Reference: N&W 2e Section 12.3 / eq. 12.34 (Lagrangian
        //            stationarity); N&W 2e Section 12.1 (KKT
        //            conditions).
        scalar_type reported_grad_norm;
        std::optional<scalar_type> kkt{};
        if constexpr(differentiable<P, scalar_type>)
        {
            constexpr int PD = P::problem_dimension;
            Eigen::Vector<scalar_type, N> grad_f(n);
            if constexpr(N == PD)
                s.problem->gradient(s.x, grad_f);
            else
            {
                Eigen::Vector<scalar_type, PD> g_tmp(n);
                s.problem->gradient(Eigen::Vector<scalar_type, N>(s.x), g_tmp);
                grad_f = g_tmp;
            }

            const int m = s.n_eq + s.n_ineq;
            Eigen::MatrixX<scalar_type> J_all(m, n);
            if(m > 0)
                s.problem->constraint_jacobian(s.x, J_all);
            Eigen::Matrix<scalar_type, Eigen::Dynamic, N> J_eq_new = J_all.topRows(s.n_eq);
            Eigen::Matrix<scalar_type, Eigen::Dynamic, N> J_ineq_new = J_all.bottomRows(s.n_ineq);

            if constexpr(lift_equalities_only)
            {
                // AUGLAG_EQ KKT residual. The equality multipliers are the
                // outer Hestenes-Powell estimates (s.lambda_eq); the
                // inequality multipliers live in the inner constrained
                // solver's dual (state().y_dual), surfaced here as mu_ineq
                // with no sign flip (the inner dual already uses the argmin
                // c_ineq >= 0 convention). The reported gradient norm is the
                // stationarity leg of the ORIGINAL Lagrangian, not the
                // equality-augmented objective, since the inequalities are
                // now carried by mu_ineq rather than folded into the penalty.
                //
                // Reference: N&W 2e Definition 12.1 / eq. 12.34; Bertsekas
                //            1982 Section 4.2 (partial elimination).
                Eigen::VectorX<scalar_type> mu_ineq = ctx.inner_solver->state().y_dual;
                Eigen::Vector<scalar_type, N> grad_L = grad_f;
                if(s.n_eq > 0)
                    grad_L.noalias() -= J_eq_new.transpose() * s.lambda_eq;
                if(s.n_ineq > 0)
                    grad_L.noalias() -= J_ineq_new.transpose() * mu_ineq;
                reported_grad_norm = grad_L.norm();

                kkt = detail::kkt_residual<scalar_type,
                                           Eigen::Dynamic,
                                           Eigen::Dynamic,
                                           Eigen::Dynamic>(
                    grad_f, J_eq_new, J_ineq_new,
                    s.lambda_eq, mu_ineq,
                    s.c_eq, s.c_ineq);
            }
            else
            {
                auto aug_grad = detail::augmented_lagrangian_gradient(
                    grad_f, J_eq_new, J_ineq_new, s.c_eq, s.c_ineq,
                    s.lambda_eq, s.lambda_ineq, s.mu);
                reported_grad_norm = aug_grad.norm();

                // Reference: N&W 2e Definition 12.1 (KKT conditions:
                //            stationarity, primal feasibility, dual
                //            feasibility, complementarity); eq. 12.34
                //            (Lagrangian stationarity leg). Dual
                //            feasibility leg is identically zero here
                //            because detail::update_multipliers projects
                //            lambda_ineq onto the non-negative orthant
                //            per N&W 17.55.
                //
                // Template parameters are specified explicitly because
                // auglag's locally-constructed grad_f (Vector<Scalar, N>)
                // and J_eq_new (Matrix<Scalar, Dynamic, N>) combine
                // compile-time and dynamic-sized dimensions in a way that
                // cannot be deduced from the fully-dynamic multiplier and
                // constraint vectors (VectorX<Scalar>). We instantiate the
                // fully-dynamic variant so the Ref-based parameters bind
                // all argument types uniformly.
                kkt = detail::kkt_residual<scalar_type,
                                           Eigen::Dynamic,
                                           Eigen::Dynamic,
                                           Eigen::Dynamic>(
                    grad_f, J_eq_new, J_ineq_new,
                    s.lambda_eq, s.lambda_ineq,
                    s.c_eq, s.c_ineq);
            }
        }
        else
        {
            reported_grad_norm = step_norm;
        }

        // K&W Algorithm 10.2 / N&W Algorithm 17.4:
        // Conn, Gould & Toint (1991) recommend updating multipliers only
        // when sufficient feasibility progress is observed, and decreasing
        // the penalty parameter otherwise. The inner stopping tolerance
        // omega_k follows the same two-branch split: tighten on the
        // multiplier-update (success) branch, re-derive from the new penalty
        // on the penalty-reduction (failure) branch.
        if(gate_viol < feas_progress * s.prev_viol || s.outer_iter == 0)
        {
            if constexpr(lift_equalities_only)
            {
                // AUGLAG_EQ (Bertsekas 1982 Section 4.2 partial elimination):
                // the outer method of multipliers owns the equalities only.
                // Update lambda_eq alone (Hestenes-Powell); the inequality
                // multipliers live in the inner constrained solver's dual.
                // Passing a fresh empty vector as the inequality-multiplier
                // argument leaves the real s.lambda_ineq untouched (the
                // inequality loop iterates over the empty vector's size).
                Eigen::VectorX<scalar_type> empty_ineq(0);
                detail::update_multipliers(
                    s.lambda_eq, empty_ineq, s.c_eq, empty_ineq, s.mu);
            }
            else
            {
                detail::update_multipliers(
                    s.lambda_eq, s.lambda_ineq, s.c_eq, s.c_ineq, s.mu);
            }

            // Success branch: penalty is pinned, so tighten the inner
            // tolerance multiplicatively (mu < 1). This is what drives
            // omega_k -> 0 across multiplier-only outer iterations, where
            // the from-scratch mu^alpha form would sit flat forever.
            s.omega = *s.omega * std::pow(s.mu, tol_alpha);
        }
        else
        {
            s.mu = std::max(s.mu * mu_dec, mu_floor);
            // Failure branch: the penalty regime just tightened; re-derive
            // omega from the new (smaller) mu rather than carrying the
            // possibly-unreachable tolerance accumulated in the previous
            // regime.
            s.omega = static_cast<scalar_type>(tol_eta * std::pow(s.mu, tol_alpha));
        }
        s.prev_viol = gate_viol;

        ++s.outer_iter;

        return step_result<scalar_type>{
            .objective_value = s.f,
            .gradient_norm = reported_grad_norm,
            .step_size = step_norm,
            .objective_change = s.f - f_old,
            .improved = s.f < f_old || viol < con_tol,
            .constraint_violation = viol,
            .x_norm = s.x.norm(),
            .kkt_residual = kkt,
        };
    }

    template <typename P>
    void reset(state_type<P>& s, const Eigen::Vector<scalar_type, N>& x0)
    {
        s.x = x0;
        s.f = s.problem->value(x0);
        {
            const int m = s.n_eq + s.n_ineq;
            Eigen::VectorX<scalar_type> c_all(m);
            if(m > 0)
                s.problem->constraints(x0, c_all);
            s.c_eq = c_all.head(s.n_eq);
            s.c_ineq = c_all.tail(s.n_ineq);
        }
        s.outer_iter = 0;
        // Re-seed the adaptive inner tolerance on the next step from the
        // current penalty regime.
        s.omega.reset();
        // Destroy inner solver; will be reconstructed next step.
        // Subproblem storage is kept (its pointers are re-seeded next step).
        if(s.ctx)
            s.ctx->inner_solver.reset();
    }

    template <typename P>
    void reset_clear(state_type<P>& s,
                     const Eigen::Vector<scalar_type, N>& x0)
    {
        reset(s, x0);
        s.lambda_eq.setZero();
        s.lambda_ineq.setZero();
        s.mu = s.opts.mu_init;
        s.prev_viol = detail::constraint_violation(s.c_eq, s.c_ineq);
        if(s.ctx)
        {
            s.ctx->sub_storage.reset();
            s.ctx->inner_solver.reset();
        }
    }
};

}

#endif
