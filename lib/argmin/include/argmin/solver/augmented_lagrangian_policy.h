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
// Warm-start: the inner basic_solver is persisted across outer
// iterations, preserving compact_lbfgs curvature pairs via
// lbfgsb_policy::reset(). The subproblem's penalty parameter is
// stored as a pointer to the outer state's mu, so multiplier and
// penalty updates propagate automatically without reconstruction.
//
// Adaptive inner tolerance: follows the Conn-Gould-Toint (1991)
// schedule where inner tolerance loosens in early outer iterations
// and tightens as mu decreases (N&W Algorithm 17.4).
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
#include "argmin/solver/basic_solver.h"
#include "argmin/solver/options.h"
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

namespace argmin
{

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

        // Conn-Gould-Toint (1991) adaptive inner tolerance schedule.
        // inner_tol = max(eta / mu^alpha, inner_grad_tol)
        // Reference: N&W Algorithm 17.4, Step 2.
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

        using rebound_inner = typename InnerPolicy::template rebind<N>;

        // Heap-boxed inner context.
        //
        // The subproblem stores raw pointers to its evaluation buffers and to
        // outer state (multipliers, penalty, bounds), and the persisted inner
        // solver caches the address of the subproblem. basic_solver's move is
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
            std::optional<subproblem> sub_storage;
            std::optional<basic_solver<rebound_inner, N, subproblem>> inner_solver;
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
        using subproblem = typename state_type<P>::subproblem;
        auto& ctx = *s.ctx;
        if(!ctx.sub_storage.has_value())
            ctx.sub_storage.emplace();
        {
            subproblem& sp = *ctx.sub_storage;
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

        // Adaptive inner tolerance (Conn-Gould-Toint 1991; N&W Algorithm 17.4).
        // inner_tol = max(eta / mu^alpha, tol_final)
        // Early outer iterations (large mu) get loose tolerance; as mu
        // decreases the inner solve tightens toward the final tolerance.
        // CGT 1991 §3 prescribes alpha >= 0.9 to couple inner/outer
        // iterations meaningfully; the prior alpha = 0.1 default left
        // the schedule essentially flat across the mu range, effectively
        // running the inner solver to its default tolerance on every
        // outer iter regardless of penalty.
        const scalar_type tol_eta = s.opts.inner_tolerance_eta;
        const scalar_type tol_alpha = s.opts.inner_tolerance_alpha;
        const scalar_type inner_tol = std::max(
            static_cast<scalar_type>(tol_eta / std::pow(s.mu, tol_alpha)),
            inner_grad_tol);

        solver_options<> inner_opts;
        inner_opts.max_iterations = inner_max;
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
        scalar_type viol = detail::primal_feasibility_inf(s.c_eq, s.c_ineq);

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
        else
        {
            reported_grad_norm = step_norm;
        }

        // K&W Algorithm 10.2 / N&W Algorithm 17.4:
        // Conn, Gould & Toint (1991) recommend updating multipliers only
        // when sufficient feasibility progress is observed, and decreasing
        // the penalty parameter otherwise.
        if(viol < feas_progress * s.prev_viol || s.outer_iter == 0)
        {
            detail::update_multipliers(
                s.lambda_eq, s.lambda_ineq, s.c_eq, s.c_ineq, s.mu);
        }
        else
        {
            s.mu = std::max(s.mu * mu_dec, mu_floor);
        }
        s.prev_viol = viol;

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
