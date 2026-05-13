#ifndef HPP_GUARD_ARGMIN_SOLVER_TR_SQP_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_TR_SQP_POLICY_H

// Trust-region SQP policy implementing the Byrd-Omojokun composite step.
//
// The policy avoids the L1-merit Maratos failure mode by construction: no
// merit function, no Armijo backtracker, no penalty-parameter calibration.
// Step acceptance is the canonical actual-vs-predicted reduction ratio.
//
// Inequalities are reformulated via slacks (c_ineq(x) + s = 0, s >= 0).
// The joint primal is z = (x, s) in R^{n + n_ineq}; the Lagrangian Hessian
// approximation lives on the joint space via dense_ldl_bfgs<double, N>.
//
// argmin variant: bounds projection inside Steihaug-CG handles box bounds
//                 on x and slack bounds on s uniformly; TR ratio thresholds
//                 and radius update factors are uniform across modes per
//                 literature consensus; per-mode dispatch (CG inner-iter cap,
//                 forcing-sequence variant, BFGS-skip-on-non-positive-
//                 curvature, multiplier-reest stride) is policy-local via
//                 `if constexpr (Mode == sqp_mode::fast)`.
//
// Reference: Nocedal and Wright 2e Section 18.5 Algorithm 18.4
//            (Byrd-Omojokun composite step);
//            Lalee, Nocedal, Plantenga 1998 SIAM J. Optim. 8(3):682-706;
//            Conn, Gould, Toint 2000 MOS-SIAM Trust-Region Methods
//            Chapters 7, 12, 17;
//            Nocedal and Wright 2e Section 4.1 (TR ratio test and radius
//            update);
//            scipy/optimize/_trustregion_constr (public-API reference
//            shape).

#include "argmin/detail/lagrangian.h"
#include "argmin/detail/sqp_common.h"
#include "argmin/detail/kkt_residual.h"
#include "argmin/detail/byrd_omojokun.h"
#include "argmin/detail/steihaug_cg.h"
#include "argmin/detail/dense_ldl_bfgs.h"
#include "argmin/detail/bound_projection.h"

#include "argmin/result/step_result.h"

#include "argmin/solver/options.h"
#include "argmin/solver/sqp_mode.h"

#include "argmin/types.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace argmin
{

// argmin variant: closed-set Mode NTTP threaded through tr_sqp_policy.
//                 `rebind<M>` preserves Mode on N rebind. Per-mode
//                 Steihaug-CG inner-iter cap, forcing-sequence variant,
//                 BFGS-skip-on-non-positive-curvature gate, and
//                 multiplier-reest stride are exposed as static-constexpr
//                 members. TR ratio thresholds and radius-update factors
//                 are uniform across modes per literature consensus.
//                 `accurate` is the default to preserve cross-family
//                 consistency with the line-search SQP policies.
//
// Reference: Nocedal and Wright 2e Section 18.5 Algorithm 18.4
//            (Byrd-Omojokun composite step);
//            Section 7.3 Algorithm 7.2 (truncated CG);
//            Eisenstat and Walker 1996 SIAM J. Sci. Comput. 17(1):16-32
//            (accurate-mode adaptive forcing);
//            Dembo, Eisenstat, Steihaug 1982 SIAM J. Numer. Anal.
//            19(2):400-408 (fast-mode forcing sequence).
template <int N = dynamic_dimension, sqp_mode Mode = sqp_mode::accurate>
struct tr_sqp_policy
{
    using scalar_type = double;
    static constexpr sqp_mode mode_ = Mode;

    template <int M>
    using rebind = tr_sqp_policy<M, Mode>;

    // Cross-cutting convergence tolerances (cross-family consistency
    // with kraft_slsqp, nw_sqp, filter_slsqp, filter_nw_sqp). A consumer
    // racing line-search vs trust-region policies under
    // basic_solver_group sees a uniform tolerance contract.
    static constexpr double default_gradient_tolerance =
        (Mode == sqp_mode::fast) ? 1e-4 : 1e-8;
    static constexpr double default_step_tolerance_rel =
        (Mode == sqp_mode::fast) ? 1e-6 : 1e-12;
    static constexpr double default_feasibility_tolerance =
        (Mode == sqp_mode::fast) ? 1e-4 : 1e-6;

    // BFGS-skip-on-non-positive-curvature gate (policy-level; the
    // dense_ldl_bfgs helper itself is mode-blind). Fast mode skips the
    // Powell-damping path when s^T y <= 0 and increments
    // diagnostics.bfgs_skip_count; accurate mode lets dense_ldl_bfgs
    // apply Powell damping per N&W eq. 18.22-18.24.
    //
    // Reference: Nocedal and Wright 2e Procedure 18.2 (damping guard);
    //            eq. 18.22-18.24 (Powell damping, accurate path).
    static constexpr bool default_bfgs_skip_on_nonpositive_curvature =
        (Mode == sqp_mode::fast);

    // Active-set Lagrange-multiplier re-estimation stride. The post-step
    // KKT-leg invokes detail::compute_kkt_multipliers_active_set only
    // when (s.iteration % multiplier_reest_every_k == 0); on the skip
    // path the prior step's kkt_lambda_eq_buf / kkt_mu_ineq_buf are
    // reused (state-resident, zero-alloc).
    //
    // Reference: Bertsekas 1996 §4.2 (stale-multiplier reuse rationale);
    //            Nocedal and Wright 2e Section 18.3 + Algorithm 18.3
    //            (working-set identification).
    static constexpr std::size_t default_multiplier_reest_every_k =
        (Mode == sqp_mode::fast) ? std::size_t{5} : std::size_t{1};

    // Steihaug-CG inner-iteration cap MULTIPLIER (resolved against
    // n + n_ineq at policy init time). Fast caps at the textbook
    // exact-arithmetic bound; accurate doubles it to absorb inexact-
    // Hessian (BFGS) effects. This is the primary wall-time lever for
    // fast mode — Steihaug-CG iterations dominate per-step cost in TRSQP
    // (each iteration runs one O((n+n_ineq)^2) Hessian-vector product
    // through dense_ldl_bfgs).
    //
    // Reference: Nocedal and Wright 2e Section 7.3 (truncated CG
    //            converges in <= n exact-arithmetic iterations on SPD B).
    static constexpr std::size_t default_max_cg_iterations_multiplier =
        (Mode == sqp_mode::fast) ? std::size_t{1} : std::size_t{2};

    // Forcing-sequence dispatch is resolved at the step() site via
    // `if constexpr (Mode == sqp_mode::fast)`:
    //   accurate: Eisenstat-Walker 1996  eps_k = min(0.5, sqrt(||grad L||))
    //   fast:     Dembo-Eisenstat-Steihaug 1982  eps_k = min(0.1, ||grad L||)
    // Both forcing sequences are documented in Nocedal and Wright 2e
    // Section 7.3 + Conn, Gould, Toint 2000 Chapter 7 Section 7.4.

    // Uniform-across-modes trust-region parameters. Mirrored from the
    // detail::byrd_omojokun constexpr block so callers can read tunable
    // defaults from the policy's options_type while the helper continues
    // to receive the resolved value as a function argument.
    //
    // Reference: Nocedal and Wright 2e Section 4.1 Algorithm 4.1
    //            (universal trust-region defaults; consistent across
    //            scipy trust-constr, Ceres trust-region minimizer,
    //            KNITRO).
    static constexpr double default_initial_trust_radius = 1.0;
    static constexpr double default_min_trust_radius     = 1e-12;

    struct options_type
    {
        // Direct-field-default form: literature-defaulted scalars are
        // brace-initialized from the per-mode default_* static-constexpr
        // members above. Step site reads the field directly (no
        // value_or / has_value indirection). Callers who want a non-
        // default value assign the field directly.
        //
        // Reference: Nocedal and Wright 2e Section 4.1 (universal TR
        //            radius defaults).
        double initial_trust_radius{default_initial_trust_radius};
        double min_trust_radius{default_min_trust_radius};

        // Steihaug-CG inner-iter cap MULTIPLIER; resolved against
        // n + n_ineq at the step() call site.
        std::size_t max_cg_iterations_multiplier{
            default_max_cg_iterations_multiplier};

        // Stall window carried over from the line-search SQP family for
        // basic_solver convergence-framework parity.
        std::uint16_t stall_window{50};

        // Active-set multiplier re-estimation stride. A value of 0 is
        // treated as 1 (re-estimate every step) by the read-site clamp.
        std::size_t multiplier_reest_every_k{
            default_multiplier_reest_every_k};
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
        Eigen::Vector<double, N> x;
        Eigen::Vector<double, N> g;
        Eigen::VectorXd c_eq;
        Eigen::VectorXd c_ineq;
        Eigen::Matrix<double, Eigen::Dynamic, N> J_eq;
        Eigen::Matrix<double, Eigen::Dynamic, N> J_ineq;
        Eigen::Vector<double, N> lower;
        Eigen::Vector<double, N> upper;
        Eigen::VectorXd lambda;

        // Cross-policy state-resident buffer struct. TRSQP resizes this
        // with the JOINT dimension n + n_ineq so the per-step iterate /
        // direction / curvature-pair buffers all carry the slack-
        // augmented primal width.
        //
        // Adopted from: argmin/detail/sqp_common.h sqp_state_buffers.
        argmin::detail::sqp_state_buffers<double, N> bufs;

        // TRSQP-specific slack vector for the inequality reformulation
        // c_ineq(x) + s = 0, s >= 0. The full step() integration in a
        // later commit operates on the joint (x, s) primal.
        Eigen::VectorXd s_slack;

        // Current trust-region radius; updated by the ratio test inside
        // step(). The skeleton seeds it from options.initial_trust_radius
        // at init / reset time.
        double trust_radius{default_initial_trust_radius};

        // LDLT workspace pair used by detail::equality_feasibility_warmstart
        // on the joint Jacobian inside the byrd_omojokun normal-step leg.
        Eigen::MatrixXd AAt_workspace;
        Eigen::LDLT<Eigen::MatrixXd> ldlt_feasibility;
        Eigen::VectorXd w_workspace;

        double objective_value{};
        // Lagrangian-Hessian approximation on the JOINT (x, s) space;
        // sized n + n_ineq.
        //
        // Reference: Fletcher and Powell 1974 Math. Computation 28:1067-
        //            1078 (LDL^T rank-1 update);
        //            Nocedal and Wright 2e Section 7.2 (limited-memory
        //            BFGS);
        //            Kraft 1988 DFVLR-FB 88-28 Section 2.2.3 (BFGS in
        //            the SQP outer loop).
        detail::dense_ldl_bfgs<double, N> hessian;
        std::uint32_t iteration{0};
        int n_eq{0};
        int n_ineq{0};
    };

    template <typename Problem, typename Convergence>
    state_type<Problem> init(const Problem& problem,
                             const Eigen::Vector<double, N>& x0,
                             const solver_options<Convergence>& opts,
                             const options_type& policy_opts)
    {
        options = policy_opts;
        return init(problem, x0, opts);
    }

    template <typename Problem, typename Convergence = default_convergence>
    state_type<Problem> init(const Problem& problem,
                             const Eigen::Vector<double, N>& x0,
                             const solver_options<Convergence>& /*opts*/)
    {
        static_assert(differentiable<Problem>,
                      "tr_sqp_policy requires differentiable<Problem>");
        static_assert(constrained<Problem>,
                      "tr_sqp_policy requires constrained<Problem>");

        const int n = problem.dimension();
        state_type<Problem> s;
        s.problem = &problem;

        s.n_eq = problem.num_equality();
        s.n_ineq = problem.num_inequality();
        const int m = s.n_eq + s.n_ineq;

        s.x = x0;
        s.g.setZero(n);
        s.objective_value = problem.value(x0);
        problem.gradient(x0, s.g);

        // Cross-policy state-resident buffer struct sized on the JOINT
        // (x, s) dimension. The slack reformulation widens the primal to
        // n + n_ineq; all per-step iterate / direction / curvature-pair
        // buffers carry that width. The constraint-axis buffers are
        // sized on n_eq + n_ineq (the joint constraint count: original
        // equalities plus the new slack equalities c_ineq + s = 0).
        s.bufs.resize(n + s.n_ineq, s.n_eq + s.n_ineq, 0);

        // LDLT workspace for the normal-step LSQ leg. Sized on the joint
        // constraint count m_joint = n_eq + n_ineq (the normal step
        // solves min ||A v + c||^2 against the slack-augmented A).
        const int m_joint = s.n_eq + s.n_ineq;
        if(m_joint > 0)
        {
            s.AAt_workspace.resize(m_joint, m_joint);
            s.ldlt_feasibility = Eigen::LDLT<Eigen::MatrixXd>(m_joint);
            s.w_workspace.resize(m_joint);
        }

        // Constraint evaluation. The state holds c_eq / c_ineq /
        // J_eq / J_ineq in the un-reformulated (x-only) shape so the
        // public-API-facing constraint_violation diagnostic continues
        // to report on the original problem; the slack-augmented joint
        // (A, c) is assembled at the step() site in a later commit.
        Eigen::VectorXd c_eval(m);
        if(m > 0)
            problem.constraints(x0, c_eval);
        s.c_eq = c_eval.head(s.n_eq);
        s.c_ineq = c_eval.tail(s.n_ineq);

        Eigen::MatrixXd J_eval(m, n);
        if(m > 0)
            problem.constraint_jacobian(x0, J_eval);
        s.J_eq = J_eval.topRows(s.n_eq);
        s.J_ineq = J_eval.bottomRows(s.n_ineq);

        // Box bounds: detected via concept. The slack bounds [0, +inf]
        // are implicit and applied inside Steihaug-CG by the joint-
        // dimension displaced box in the later step() integration.
        if constexpr(bound_constrained<Problem>)
        {
            s.lower = problem.lower_bounds();
            s.upper = problem.upper_bounds();
        }
        else
        {
            constexpr double inf = std::numeric_limits<double>::infinity();
            s.lower = Eigen::Vector<double, N>::Constant(n, -inf);
            s.upper = Eigen::Vector<double, N>::Constant(n, inf);
        }

        // Slack initialization: s_0 = max(-c_ineq(x_0), 0) enforces
        // s_0 >= 0 and minimizes the initial equality residual
        // ||c_ineq(x_0) + s_0||.
        //
        // Reference: scipy/optimize/_trustregion_constr (public-API
        //            reference shape for slack-reformulated inequalities).
        s.s_slack = (-s.c_ineq).cwiseMax(0.0).eval();

        // Lagrangian-Hessian on the joint (x, s) space. dense_ldl_bfgs
        // initializes to identity; the slack block picks up curvature
        // through (s_k, y_k) pairs naturally on subsequent push() calls.
        s.hessian = detail::dense_ldl_bfgs<double, N>(n + s.n_ineq);

        // Multiplier vector sized on the un-reformulated constraint
        // count (cross-policy consistency with the line-search SQP
        // family; the slack-equality multipliers are policy-internal
        // and live inside bufs).
        s.lambda = Eigen::VectorXd::Zero(m);

        s.trust_radius = options.initial_trust_radius;
        s.iteration = 0;

        return s;
    }

    template <typename P>
    step_result<double> step(state_type<P>& s)
    {
        // Skeleton: returns a null-step result that satisfies the
        // nlp_solver concept and flows cleanly through the convergence
        // framework. The Byrd-Omojokun composite-step integration
        // (normal step + Steihaug-CG tangential step + ratio test +
        // radius update + BFGS curvature-pair update on the joint
        // (x, s) space + KKT-multiplier active-set re-estimation)
        // lands in the next commit.
        s.bufs.kkt_lambda_eq_buf.setZero(s.n_eq);
        s.bufs.kkt_mu_ineq_buf.setZero(s.n_ineq);
        auto r = argmin::detail::null_step_result<double, N,
                                                  Eigen::Dynamic,
                                                  Eigen::Dynamic>(
            s.objective_value, s.g, s.J_eq, s.J_ineq,
            s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf,
            s.c_eq, s.c_ineq, s.x.norm(), /*bfgs_reset_count=*/0);
        ++s.iteration;
        return r;
    }

    // Hot start -- preserves BFGS Hessian.
    template <typename P>
    void reset(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        const int n = static_cast<int>(x0.size());
        const int m = s.n_eq + s.n_ineq;
        s.x = x0;
        s.objective_value = s.problem->value(x0);
        s.problem->gradient(x0, s.g);

        Eigen::VectorXd c_eval(m);
        if(m > 0)
            s.problem->constraints(x0, c_eval);
        s.c_eq = c_eval.head(s.n_eq);
        s.c_ineq = c_eval.tail(s.n_ineq);

        Eigen::MatrixXd J_eval(m, n);
        if(m > 0)
            s.problem->constraint_jacobian(x0, J_eval);
        s.J_eq = J_eval.topRows(s.n_eq);
        s.J_ineq = J_eval.bottomRows(s.n_ineq);

        s.s_slack = (-s.c_ineq).cwiseMax(0.0).eval();
        s.trust_radius = options.initial_trust_radius;
        s.iteration = 0;
    }

    // Cold restart -- clears BFGS Hessian.
    template <typename P>
    void reset_clear(state_type<P>& s,
                     const Eigen::Vector<double, N>& x0)
    {
        reset(s, x0);
        s.hessian.reset();
        s.lambda.setZero();
    }
};

template <int N = dynamic_dimension>
using tr_sqp_policy_fast = tr_sqp_policy<N, sqp_mode::fast>;

template <int N = dynamic_dimension>
using tr_sqp_policy_accurate = tr_sqp_policy<N, sqp_mode::accurate>;

}

#endif
