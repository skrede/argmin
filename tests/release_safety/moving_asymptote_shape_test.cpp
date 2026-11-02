#include "eigen_trap.h"

#include "argmin/solver/options.h"
#include "argmin/solver/mma_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/ccsa_quadratic_policy.h"

#include "argmin/solver/alternative/gcmma/rho_wval_policy.h"
#include "argmin/solver/alternative/gcmma/raa_augmented_policy.h"
#include "argmin/solver/alternative/gcmma/move_limit_shrink_policy.h"

#include "argmin/result/status.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <tuple>
#include <cstdint>

using namespace argmin;

namespace
{

using vector2 = Eigen::Vector2d;

template <std::int32_t Cap>
struct declared
{
    static constexpr std::int32_t constraint_count = Cap;
};

struct undeclared
{
};

// f = 0.5 |x - (1, 2)|^2 on the [-5, 5]^2 box with `equalities` copies of x0 + x1 - 3 = 0 first, then the first `rows`
// of the inactive x0 + 10 >= 0, x1 + 10 >= 0 and 10 - x0 - x1 >= 0. A constraint call handed a buffer of any length
// other than equalities + rows is counted and writes nothing.
template <typename Count>
struct counted_rows : Count
{
    static constexpr std::int32_t problem_dimension = 2;

    std::int32_t rows;
    std::int32_t equalities;
    mutable std::int32_t evaluations;
    mutable std::int32_t wrong_buffers;

    std::int32_t dimension() const { return 2; }

    std::int32_t num_equality() const { return equalities; }

    std::int32_t num_inequality() const { return rows; }

    double value(const vector2& x) const
    {
        ++evaluations;
        return 0.5 * (x - target()).squaredNorm();
    }

    void gradient(const vector2& x, vector2& g) const
    {
        ++evaluations;
        g = x - target();
    }

    template <typename Values>
    void constraints(const vector2& x, Values& c) const
    {
        if(!counts(c.rows()))
            return;
        c.head(equalities).setConstant(x[0] + x[1] - 3.0);
        c.tail(rows) = normals().topRows(rows) * x + Eigen::Vector3d(10.0, 10.0, 20.0).head(rows);
    }

    template <typename Jacobian>
    void constraint_jacobian(const vector2&, Jacobian& J) const
    {
        if(!counts(J.rows()))
            return;
        J.topRows(equalities).setOnes();
        J.bottomRows(rows) = normals().topRows(rows);
    }

    vector2 lower_bounds() const { return vector2::Constant(-5.0); }

    vector2 upper_bounds() const { return vector2::Constant(5.0); }

    bool counts(Eigen::Index buffer_rows) const
    {
        ++evaluations;
        const bool fits = buffer_rows == equalities + rows;
        wrong_buffers += fits ? 0 : 1;
        return fits;
    }

    static vector2 target() { return vector2(6.0, 7.0); }

    static Eigen::Matrix<double, 3, 2> normals() { return (Eigen::Matrix<double, 3, 2>() << 1.0, 0.0, 0.0, 1.0, -1.0, -1.0).finished(); }
};

using moving_asymptote_policies = std::tuple<mma_policy<double, 2>,
                                             ccsa_quadratic_policy<double, 2>,
                                             alternative::gcmma::rho_wval_policy<double, 2>,
                                             alternative::gcmma::raa_augmented_policy<double, 2>,
                                             alternative::gcmma::move_limit_shrink_policy<double, 2>>;

const vector2 inside(0.5, 0.5);

template <typename Policy, typename Problem>
void check_rejected(const Problem& problem)
{
    step_budget_solver solver{Policy{}, problem, inside, solver_options<>{}};
    CHECK(solver.status() == solver_status::invalid_problem);
    const std::int32_t evaluations = problem.evaluations;
    CHECK(solver.step().policy_status == solver_status::invalid_problem);
    CHECK(problem.evaluations == evaluations);
    step_budget_solver fresh{Policy{}, problem, inside, solver_options<>{}};
    CHECK(fresh.solve(solver_options<>{}).status == solver_status::invalid_problem);
    CHECK(problem.wrong_buffers == 0);
}

// Plain MMA and move-limit shrinkage stall short of the tolerance on this problem, so acceptance is checked as a
// non-rejected solve that improves on the start, not as convergence.
template <typename Policy, typename Problem>
void check_accepted(const Problem& problem)
{
    step_budget_solver solver{Policy{}, problem, inside, solver_options<>{}};
    CHECK(solver.status() == solver_status::running);
    const auto result = solver.solve(solver_options<>{});
    CHECK(result.status != solver_status::invalid_problem);
    CHECK(result.objective_value < problem.value(inside));
    CHECK(problem.wrong_buffers == 0);
}

template <typename Solver, typename Problem>
void check_reset_rejects(Solver& solver, const Problem& problem)
{
    const std::int32_t evaluations = problem.evaluations;
    CHECK(solver.reset(inside) == solver_status::invalid_problem);
    CHECK(solver.reset_clear(inside) == solver_status::invalid_problem);
    CHECK(solver.status() == solver_status::invalid_problem);
    CHECK(problem.evaluations == evaluations);
}

}

TEMPLATE_LIST_TEST_CASE("a moving-asymptote policy rejects more inequality rows than its fixed constraint count", "[release-safety][moving-asymptote-shape]", moving_asymptote_policies)
{
    check_rejected<TestType>(counted_rows<declared<1>>{.rows = 3, .equalities = 0, .evaluations = 0, .wrong_buffers = 0});
}

TEMPLATE_LIST_TEST_CASE("a moving-asymptote policy rejects fewer inequality rows than its fixed constraint count", "[release-safety][moving-asymptote-shape]", moving_asymptote_policies)
{
    check_rejected<TestType>(counted_rows<declared<3>>{.rows = 1, .equalities = 0, .evaluations = 0, .wrong_buffers = 0});
}

TEMPLATE_LIST_TEST_CASE("a moving-asymptote policy rejects an equality row with a full-length constraint probe", "[release-safety][moving-asymptote-shape]", moving_asymptote_policies)
{
    check_rejected<TestType>(counted_rows<undeclared>{.rows = 1, .equalities = 1, .evaluations = 0, .wrong_buffers = 0});
}

TEMPLATE_LIST_TEST_CASE("a moving-asymptote policy accepts a matching or undeclared constraint count", "[release-safety][moving-asymptote-shape]", moving_asymptote_policies)
{
    SECTION("three rows under a fixed count of three")
    {
        check_accepted<TestType>(counted_rows<declared<3>>{.rows = 3, .equalities = 0, .evaluations = 0, .wrong_buffers = 0});
    }
    SECTION("three rows with no declared count")
    {
        check_accepted<TestType>(counted_rows<undeclared>{.rows = 3, .equalities = 0, .evaluations = 0, .wrong_buffers = 0});
    }
}

TEMPLATE_LIST_TEST_CASE("a moving-asymptote policy reset keeps rejecting a mismatched constraint count without evaluating", "[release-safety][moving-asymptote-shape]", moving_asymptote_policies)
{
    const counted_rows<declared<1>> problem{.rows = 3, .equalities = 0, .evaluations = 0, .wrong_buffers = 0};
    step_budget_solver solver{TestType{}, problem, inside, solver_options<>{}};
    check_reset_rejects(solver, problem);
    CHECK(problem.wrong_buffers == 0);
}

TEMPLATE_LIST_TEST_CASE("a moving-asymptote policy reset rejects a changed row shape until it is restored", "[release-safety][moving-asymptote-shape]", moving_asymptote_policies)
{
    SECTION("a fixed count of three")
    {
        counted_rows<declared<3>> problem{.rows = 3, .equalities = 0, .evaluations = 0, .wrong_buffers = 0};
        step_budget_solver solver{TestType{}, problem, inside, solver_options<>{}};
        problem.rows = 1;
        check_reset_rejects(solver, problem);
        problem.rows = 3;
        problem.equalities = 1;
        check_reset_rejects(solver, problem);
        problem.equalities = 0;
        CHECK(solver.reset_clear(inside) == solver_status::running);
        CHECK(solver.solve(solver_options<>{}).status != solver_status::invalid_problem);
        CHECK(problem.wrong_buffers == 0);
    }
    SECTION("no declared count")
    {
        counted_rows<undeclared> problem{.rows = 3, .equalities = 0, .evaluations = 0, .wrong_buffers = 0};
        step_budget_solver solver{TestType{}, problem, inside, solver_options<>{}};
        problem.rows = 2;
        check_reset_rejects(solver, problem);
        problem.rows = 3;
        CHECK(solver.reset(inside) == solver_status::running);
        CHECK(solver.solve(solver_options<>{}).status != solver_status::invalid_problem);
        CHECK(problem.wrong_buffers == 0);
    }
}
