// Instantiation + move-safety coverage for the augmented-Lagrangian /
// separable-approximation composition.
//
// The augmented Lagrangian wraps a constrained separable-approximation
// inner solver (moving asymptotes and its globally convergent /
// quadratic-penalty variants) by lifting ONLY the equalities into the
// augmented objective and passing the inequalities + box straight through
// to the inner solver, which handles them natively. Before this adapter
// the composition failed to compile: the augmented Lagrangian's synthetic
// subproblem satisfied only bound_constrained, while the inner solver
// static_asserts constrained. The equality-lifting subproblem view and the
// compile-time concept-probe fork resolve the clash.
//
// This test verifies (i) all three constrained inner solvers instantiate
// and take finite steps through the augmented Lagrangian, (ii) the
// concept-probe fork selects the equality-lifting view for the constrained
// inners and the classic all-constraint view for a bound-constrained inner,
// and (iii) the new equality-lifting view is move-safe (its raw pointers
// into outer state are re-seeded every step), which under -fsanitize=address
// turns any dangling-pointer regression into a hard failure.

#include "argmin/solver/augmented_lagrangian_policy.h"
#include "argmin/solver/ccsa_quadratic_policy.h"
#include "argmin/solver/gcmma_policy.h"
#include "argmin/solver/mma_policy.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>
#include <utility>

using namespace argmin;

namespace
{

// Small convex equality-constrained problem with a finite box.
//
//   min  x0^2 + x1^2   s.t.  x0 + x1 - 1 = 0,   x in [-10, 10]^2.
//
// Hand-computed KKT point: stationarity 2 x0 = lambda, 2 x1 = lambda plus
// the equality x0 + x1 = 1 give x* = (0.5, 0.5), f* = 0.5, lambda* = 1.
struct convex_eq
{
    static constexpr int problem_dimension = 2;
    static constexpr int constraint_count = 1;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 1; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] double value(const Eigen::Vector<double, 2>& x) const
    {
        return x[0] * x[0] + x[1] * x[1];
    }

    void gradient(const Eigen::Vector<double, 2>& x,
                  Eigen::Vector<double, 2>& g) const
    {
        g[0] = 2.0 * x[0];
        g[1] = 2.0 * x[1];
    }

    void constraints(const Eigen::Vector<double, 2>& x, Eigen::VectorXd& c) const
    {
        c[0] = x[0] + x[1] - 1.0;
    }

    void constraint_jacobian(const Eigen::Vector<double, 2>& /*x*/,
                             Eigen::MatrixXd& J) const
    {
        J(0, 0) = 1.0;
        J(0, 1) = 1.0;
    }

    [[nodiscard]] Eigen::Vector<double, 2> lower_bounds() const
    {
        return Eigen::Vector<double, 2>{-10.0, -10.0};
    }

    [[nodiscard]] Eigen::Vector<double, 2> upper_bounds() const
    {
        return Eigen::Vector<double, 2>{10.0, 10.0};
    }

    [[nodiscard]] Eigen::Vector<double, 2> initial_point() const
    {
        return Eigen::Vector<double, 2>{2.0, -3.0};
    }
};

constexpr int D = convex_eq::problem_dimension;

// Construct -> step -> move -> step -> destroy-intermediary -> step, the
// same shape as the move-and-step ASan instrument. A view that stores raw
// pointers into sibling state members without rebasing them on every step
// would dereference freed memory here.
template <typename Policy, typename Problem>
void construct_move_step(Policy policy, const Problem& problem,
                         const Eigen::VectorXd& x0,
                         const solver_options<>& opts)
{
    step_budget_solver probe{std::move(policy), problem, x0, opts};
    using solver_type = decltype(probe);

    auto heap_solver = std::make_unique<solver_type>(std::move(probe));
    heap_solver->step();

    solver_type survivor{std::move(*heap_solver)};
    heap_solver.reset();

    survivor.step();
    REQUIRE(survivor.state().x.size() == x0.size());
}

}

TEST_CASE("auglag_mma_instantiation_test: constrained inner solvers instantiate and step",
          "[auglag][mma][instantiation]")
{
    convex_eq problem;
    Eigen::VectorXd x0 = problem.initial_point();

    solver_options opts;
    opts.max_iterations = 40;
    opts.set_gradient_threshold(1e-6);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    SECTION("augmented Lagrangian over moving asymptotes")
    {
        step_budget_solver solver{augmented_lagrangian_policy<mma_policy<D>, D>{},
            problem, x0, opts};
        auto sr = solver.step();
        CHECK(std::isfinite(sr.objective_value));
        CHECK(std::isfinite(sr.gradient_norm));
        CHECK(sr.kkt_residual.has_value());
    }

    SECTION("augmented Lagrangian over globally convergent MMA")
    {
        step_budget_solver solver{augmented_lagrangian_policy<gcmma_policy<D>, D>{},
            problem, x0, opts};
        auto sr = solver.step();
        CHECK(std::isfinite(sr.objective_value));
        CHECK(std::isfinite(sr.gradient_norm));
        CHECK(sr.kkt_residual.has_value());
    }

    SECTION("augmented Lagrangian over quadratic-penalty CCSA")
    {
        step_budget_solver solver{augmented_lagrangian_policy<ccsa_quadratic_policy<D>, D>{},
            problem, x0, opts};
        auto sr = solver.step();
        CHECK(std::isfinite(sr.objective_value));
        CHECK(std::isfinite(sr.gradient_norm));
        CHECK(sr.kkt_residual.has_value());
    }
}

// The concept-probe fork must select the equality-lifting view for the
// constrained inner solvers and the classic all-constraint view for a
// bound-constrained inner solver. lift_equalities_only is the compile-time
// switch that drives that selection.
TEST_CASE("auglag_mma_instantiation_test: concept-probe fork selects the correct view",
          "[auglag][mma][instantiation]")
{
    using mma_al   = augmented_lagrangian_policy<mma_policy<D>, D>;
    using gcmma_al = augmented_lagrangian_policy<gcmma_policy<D>, D>;
    using ccsa_al  = augmented_lagrangian_policy<ccsa_quadratic_policy<D>, D>;
    using lbfgsb_al = augmented_lagrangian_policy<lbfgsb_policy<D>, D>;

    STATIC_REQUIRE(mma_al::state_type<convex_eq>::lift_equalities_only);
    STATIC_REQUIRE(gcmma_al::state_type<convex_eq>::lift_equalities_only);
    STATIC_REQUIRE(ccsa_al::state_type<convex_eq>::lift_equalities_only);
    STATIC_REQUIRE_FALSE(lbfgsb_al::state_type<convex_eq>::lift_equalities_only);

    // The equality-lifting view satisfies the constrained concept the inner
    // solver static_asserts; the classic view satisfies only bound_constrained.
    using mma_state = mma_al::state_type<convex_eq>;
    STATIC_REQUIRE(constrained<mma_state::eq_subproblem>);
    STATIC_REQUIRE(differentiable<mma_state::eq_subproblem>);
    STATIC_REQUIRE(bound_constrained<mma_state::subproblem>);
    STATIC_REQUIRE_FALSE(constrained<mma_state::subproblem>);
}

TEST_CASE("auglag_mma_instantiation_test: equality-lifting view survives move-then-step",
          "[auglag][mma][instantiation][move_asan]")
{
    convex_eq problem;
    Eigen::VectorXd x0 = problem.initial_point();
    solver_options<> opts;

    construct_move_step(augmented_lagrangian_policy<mma_policy<D>, D>{},
                        problem, x0, opts);
    construct_move_step(augmented_lagrangian_policy<gcmma_policy<D>, D>{},
                        problem, x0, opts);
    construct_move_step(augmented_lagrangian_policy<ccsa_quadratic_policy<D>, D>{},
                        problem, x0, opts);
}
