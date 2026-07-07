// Move-and-step instrument: construct, step, move, step again.
//
// Exercises every shipped policy through a construct -> step -> move ->
// step sequence with the pre-move object destroyed between the two
// moves (see construct_move_step below). A policy whose state_type
// stores raw pointers into sibling members of that same state_type
// (rather than rebasing them on every move) leaves those pointers
// referring to the destroyed intermediate object once moved --
// augmented_lagrangian_policy is exactly this shape: state_type::
// subproblem stores pointers into the enclosing state (lambda_eq,
// lambda_ineq, mu, the per-inner-iter buffers), and the persisted
// inner step_budget_solver's problem pointer is the address of that same
// subproblem. step_budget_solver's move constructor is unconditional
// noexcept memberwise move, so none of those pointers are rebased.
//
// This test is expected to abort under -fsanitize=address on the
// augmented_lagrangian case today; every other policy below is
// exercised identically to demonstrate it does not share the defect.

#include "argmin/solver/augmented_lagrangian_policy.h"
#include "argmin/solver/ccsa_quadratic_policy.h"
#include "argmin/solver/projected_gn_policy.h"
#include "argmin/solver/byrd_lbfgsb_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/cmaes_policy.h"
#include "argmin/solver/gcmma_policy.h"
#include "argmin/solver/isres_policy.h"
#include "argmin/solver/mma_policy.h"
#include "argmin/solver/lm_policy.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>
#include <utility>

using namespace argmin;

namespace
{

// Bound-constrained Rosenbrock. Used by every policy family that only
// needs box bounds (line-search SQP, L-BFGS-B/Byrd-L-BFGS-B, CMA-ES).
struct box_rosenbrock
{
    static constexpr int problem_dimension = dynamic_dimension;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        const double t1 = 1.0 - x[0];
        const double t2 = x[1] - x[0] * x[0];
        return t1 * t1 + 5.0 * t2 * t2;
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        g.resize(2);
        g[0] = -2.0 * (1.0 - x[0]) - 20.0 * (x[1] - x[0] * x[0]) * x[0];
        g[1] = 10.0 * (x[1] - x[0] * x[0]);
    }

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd::Constant(2, -5.0); }
    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd::Constant(2, 5.0); }
};

// Rosenbrock in least-squares residual form, fixed dimension. Used by
// lm_policy (unconstrained least squares).
struct rosenbrock_ls
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }
    int num_residuals() const { return 2; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        const double r0 = 1.0 - x(0);
        const double r1 = std::sqrt(5.0) * (x(1) - x(0) * x(0));
        return 0.5 * (r0 * r0 + r1 * r1);
    }

    void residuals(const Eigen::Vector<double, 2>& x, Eigen::VectorXd& r) const
    {
        r(0) = 1.0 - x(0);
        r(1) = std::sqrt(5.0) * (x(1) - x(0) * x(0));
    }

    void jacobian(const Eigen::Vector<double, 2>& x, Eigen::MatrixXd& J) const
    {
        J(0, 0) = -1.0;
        J(0, 1) = 0.0;
        J(1, 0) = -2.0 * std::sqrt(5.0) * x(0);
        J(1, 1) = std::sqrt(5.0);
    }
};

// Bounded least-squares Rosenbrock. Used by projected_gn_policy.
struct bounded_rosenbrock_ls
{
    static constexpr int problem_dimension = dynamic_dimension;

    int dimension() const { return 2; }
    int num_residuals() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        const double r0 = 1.0 - x(0);
        const double r1 = std::sqrt(5.0) * (x(1) - x(0) * x(0));
        return 0.5 * (r0 * r0 + r1 * r1);
    }

    void residuals(const Eigen::VectorXd& x, Eigen::VectorXd& r) const
    {
        r(0) = 1.0 - x(0);
        r(1) = std::sqrt(5.0) * (x(1) - x(0) * x(0));
    }

    void jacobian(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const
    {
        J(0, 0) = -1.0;
        J(0, 1) = 0.0;
        J(1, 0) = -2.0 * std::sqrt(5.0) * x(0);
        J(1, 1) = std::sqrt(5.0);
    }

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd{{-2.0, -2.0}}; }
    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd{{0.5, 2.0}}; }
};

// Bound-constrained, derivative-free Rosenbrock. Used by bobyqa_policy.
struct bobyqa_rosenbrock
{
    static constexpr int problem_dimension = dynamic_dimension;
    Eigen::VectorXd lb{Eigen::VectorXd::Constant(2, -5.0)};
    Eigen::VectorXd ub{Eigen::VectorXd::Constant(2, 5.0)};

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        const double t1 = 1.0 - x[0];
        const double t2 = x[1] - x[0] * x[0];
        return t1 * t1 + 5.0 * t2 * t2;
    }

    Eigen::VectorXd lower_bounds() const { return lb; }
    Eigen::VectorXd upper_bounds() const { return ub; }
};

// Bound + single-inequality constrained, derivative-free problem.
// Used by cobyla_policy and isres_policy.
struct simple_constrained
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        return x[0] * x[0] + x[1] * x[1];
    }

    void constraints(const Eigen::Vector<double, 2>& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = x[0] + x[1] - 1.0;
    }

    int num_equality() const { return 0; }
    int num_inequality() const { return 1; }

    Eigen::Vector<double, 2> lower_bounds() const
    {
        return Eigen::Vector<double, 2>{-10.0, -10.0};
    }

    Eigen::Vector<double, 2> upper_bounds() const
    {
        return Eigen::Vector<double, 2>{10.0, 10.0};
    }
};

// Constructs a step_budget_solver, steps it once (materializing whatever
// per-outer-iteration state the policy persists), moves it onto the
// heap, steps again, moves it back onto the stack, then destroys the
// heap intermediary before a final step().
//
// The double-hop shape with an explicit destruction in between is
// deliberate: a policy that stores pointers into sibling members of
// its own state_type rather than rebasing them on every move leaves
// those pointers referring to the destroyed intermediate object's
// memory, turning a merely-stale read into a genuine heap-use-after-
// free that ASan reliably reports -- rather than the test silently
// reading logically-wrong-but-still-allocated memory.
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

TEST_CASE("kraft_slsqp (line-search SQP) survives move-then-step", "[move_asan]")
{
    box_rosenbrock problem;
    Eigen::VectorXd x0{{-1.2, 1.0}};
    solver_options<> opts;

    construct_move_step(kraft_slsqp_policy<box_rosenbrock::problem_dimension>{},
                        problem, x0, opts);
}

TEST_CASE("tr_sqp survives move-then-step", "[move_asan]")
{
    hs076<> problem;
    auto x0 = problem.initial_point();
    solver_options<> opts;

    construct_move_step(tr_sqp_policy<hs076<>::problem_dimension>{}, problem, x0, opts);
}

TEST_CASE("lbfgsb survives move-then-step", "[move_asan]")
{
    box_rosenbrock problem;
    Eigen::VectorXd x0{{-1.2, 1.0}};
    solver_options<> opts;

    construct_move_step(lbfgsb_policy<box_rosenbrock::problem_dimension>{},
                        problem, x0, opts);
}

TEST_CASE("byrd_lbfgsb survives move-then-step", "[move_asan]")
{
    box_rosenbrock problem;
    Eigen::VectorXd x0{{-1.2, 1.0}};
    solver_options<> opts;

    construct_move_step(byrd_lbfgsb_policy<box_rosenbrock::problem_dimension>{},
                        problem, x0, opts);
}

TEST_CASE("lm survives move-then-step", "[move_asan]")
{
    rosenbrock_ls problem;
    Eigen::VectorXd x0{{-1.0, 1.0}};
    solver_options<> opts;

    construct_move_step(lm_policy{}, problem, x0, opts);
}

TEST_CASE("projected_gn survives move-then-step", "[move_asan]")
{
    bounded_rosenbrock_ls problem;
    Eigen::VectorXd x0{{-1.0, 1.0}};
    solver_options<> opts;

    construct_move_step(projected_gn_policy{}, problem, x0, opts);
}

TEST_CASE("bobyqa survives move-then-step", "[move_asan]")
{
    bobyqa_rosenbrock problem;
    Eigen::VectorXd x0{{-1.2, 1.0}};
    solver_options<> opts;

    construct_move_step(bobyqa_policy{}, problem, x0, opts);
}

TEST_CASE("cobyla survives move-then-step", "[move_asan]")
{
    simple_constrained problem;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options<> opts;

    construct_move_step(cobyla_policy{}, problem, x0, opts);
}

TEST_CASE("augmented_lagrangian survives move-then-step", "[move_asan]")
{
    // The self-referential case. state_type::subproblem references outer
    // state (lambda_eq, lambda_ineq, mu, bounds) and the per-inner-iter
    // buffers, and the persisted inner step_budget_solver caches the address of
    // that subproblem. state_type now boxes the subproblem, its buffers,
    // and the inner solver in one heap node reached through a single
    // owning pointer, and re-seeds the subproblem's outer-state pointers
    // at the top of every step -- so a memberwise move leaves every
    // internal address valid and the moved-to solver steps cleanly.
    hs076<> problem;
    auto x0 = problem.initial_point();
    solver_options<> opts;

    construct_move_step(
        augmented_lagrangian_policy<lbfgsb_policy<hs076<>::problem_dimension>>{},
        problem, x0, opts);
}

TEST_CASE("mma survives move-then-step", "[move_asan]")
{
    hs076<> problem;
    auto x0 = problem.initial_point();
    solver_options<> opts;

    construct_move_step(mma_policy<hs076<>::problem_dimension>{}, problem, x0, opts);
}

TEST_CASE("gcmma survives move-then-step", "[move_asan]")
{
    hs024<> problem;
    auto x0 = problem.initial_point();
    solver_options<> opts;

    construct_move_step(gcmma_policy<hs024<>::problem_dimension>{}, problem, x0, opts);
}

TEST_CASE("ccsa_quadratic survives move-then-step", "[move_asan]")
{
    hs076<> problem;
    auto x0 = problem.initial_point();
    solver_options<> opts;

    construct_move_step(ccsa_quadratic_policy<hs076<>::problem_dimension>{},
                        problem, x0, opts);
}

TEST_CASE("cmaes survives move-then-step", "[move_asan]")
{
    box_rosenbrock problem;
    Eigen::VectorXd x0{{-1.0, -1.0}};
    solver_options<> opts;

    construct_move_step(cmaes_policy<box_rosenbrock::problem_dimension>{},
                        problem, x0, opts);
}

TEST_CASE("isres survives move-then-step", "[move_asan]")
{
    simple_constrained problem;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options<> opts;

    construct_move_step(isres_policy<simple_constrained::problem_dimension>{},
                        problem, x0, opts);
}

// The mid-chunk variant of the augmented_lagrangian self-referential case.
// With a small inner_chunk the FIRST step() stops before the inner solve
// completes (a null step), so the subsequent move relocates the solver WHILE
// an inner solve is in flight. The chunking state machine must re-seed the
// subproblem's outer-state pointers at the top of the resume step -- a stale
// sp.pen / problem pointer there would be exactly the heap-use-after-free this
// double-hop-with-destruction shape turns into a hard ASan failure.
TEST_CASE("augmented_lagrangian survives a move mid-chunk", "[move_asan]")
{
    using policy_t =
        augmented_lagrangian_policy<lbfgsb_policy<hs076<>::problem_dimension>,
                                    hs076<>::problem_dimension>;

    hs076<> problem;
    auto x0 = problem.initial_point();
    solver_options<> opts;

    typename policy_t::options_type popts;
    popts.inner_chunk = 3;  // force the first step() to stop mid-inner-solve

    step_budget_solver probe{policy_t{}, problem, x0, opts, popts};
    using solver_type = decltype(probe);

    auto heap_solver = std::make_unique<solver_type>(std::move(probe));
    auto sr = heap_solver->step();      // partial chunk: inner solve in flight
    REQUIRE(sr.is_null_step);

    solver_type survivor{std::move(*heap_solver)};
    heap_solver.reset();

    survivor.step();                    // resume the SAME inner solve, re-seeded
    REQUIRE(survivor.state().x.size() == x0.size());
}
