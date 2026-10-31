#include "eigen_trap.h"

#include "argmin/solver/options.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/solver/alternative/isres/nlopt_faithful_policy.h"
#include "argmin/solver/alternative/isres/original_argmin_policy.h"
#include "argmin/solver/alternative/isres/runarsson_yao_paper_policy.h"

#include "argmin/result/status.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <tuple>
#include <cstdint>
#include <optional>

using namespace argmin;

namespace
{

// f = |x|^2 on the [-5, 5]^n box with `rows` copies of the inactive 10 - sum(x) >= 0. A constraint call handed a
// buffer of any length other than `rows` is counted and writes nothing.
template <std::int32_t N>
struct counted_box
{
    using vector = Eigen::Vector<double, N>;

    static constexpr std::int32_t problem_dimension = N;

    std::int32_t n;
    std::int32_t rows;
    mutable std::int32_t evaluations;
    mutable std::int32_t wrong_buffers;

    std::int32_t dimension() const { return n; }

    std::int32_t num_equality() const { return 0; }

    std::int32_t num_inequality() const { return rows; }

    double value(const vector& x) const
    {
        ++evaluations;
        return x.squaredNorm();
    }

    template <typename Values>
    void constraints(const vector& x, Values& c) const
    {
        ++evaluations;
        if(c.rows() != rows)
        {
            ++wrong_buffers;
            return;
        }
        c.setConstant(10.0 - x.sum());
    }

    vector lower_bounds() const { return vector::Constant(n, -5.0); }

    vector upper_bounds() const { return vector::Constant(n, 5.0); }
};

using original_argmin = alternative::isres::original_argmin_policy<double, 2>;
using nlopt_faithful = alternative::isres::nlopt_faithful_policy<double, 2>;
using runarsson_yao_paper = alternative::isres::runarsson_yao_paper_policy<double, 2>;

using bounded_variants = std::tuple<nlopt_faithful, runarsson_yao_paper>;
using isres_variants = std::tuple<original_argmin, nlopt_faithful, runarsson_yao_paper>;

constexpr std::uint32_t max_lambda = 1792;

const Eigen::Vector2d inside(0.5, 0.5);

counted_box<2> plane(std::int32_t rows)
{
    return {.n = 2, .rows = rows, .evaluations = 0, .wrong_buffers = 0};
}

template <typename Policy>
Policy sized(std::optional<std::uint32_t> population, std::optional<double> parent_fraction)
{
    Policy policy;
    policy.options.seed = 7u;
    policy.options.population_size = population;
    policy.options.parent_fraction = parent_fraction;
    return policy;
}

template <typename Policy, typename Problem>
void check_rejected(const Policy& policy, const Problem& problem, const typename Problem::vector& x0)
{
    step_budget_solver solver{policy, problem, x0, solver_options<>{}};
    CHECK(solver.status() == solver_status::invalid_problem);
    CHECK(solver.step().policy_status == solver_status::invalid_problem);
    CHECK(solver.reset(x0) == solver_status::invalid_problem);
    CHECK(solver.reset_clear(x0) == solver_status::invalid_problem);
    step_budget_solver fresh{policy, problem, x0, solver_options<>{}};
    CHECK(fresh.solve(solver_options<>{}).status == solver_status::invalid_problem);
    CHECK(problem.evaluations == 0);
}

template <typename Policy, typename Problem>
void check_accepted(const Policy& policy, const Problem& problem, const typename Problem::vector& x0)
{
    step_budget_solver solver{policy, problem, x0, solver_options<>{}};
    CHECK(solver.status() == solver_status::running);
    CHECK(solver.step().policy_status != solver_status::invalid_problem);
    CHECK(solver.reset(x0) == solver_status::running);
    CHECK(problem.wrong_buffers == 0);
}

}

TEMPLATE_LIST_TEST_CASE("a bounded-storage ISRES rejects a population above its column capacity", "[release-safety][isres]", bounded_variants)
{
    check_rejected(sized<TestType>(max_lambda + 1u, std::nullopt), plane(1), inside);
}

TEMPLATE_LIST_TEST_CASE("a bounded-storage ISRES rejects a parent count above its snapshot capacity", "[release-safety][isres]", bounded_variants)
{
    check_rejected(sized<TestType>(1000u, 0.5), plane(1), inside);
}

TEMPLATE_LIST_TEST_CASE("a bounded-storage ISRES rejects a default population its storage cannot hold", "[release-safety][isres]", bounded_variants)
{
    using dynamic_policy = typename TestType::template rebind<dynamic_dimension>;
    const counted_box<dynamic_dimension> problem{.n = 89, .rows = 1, .evaluations = 0, .wrong_buffers = 0};
    check_rejected(sized<dynamic_policy>(std::nullopt, std::nullopt), problem, Eigen::VectorXd::Zero(89));
}

TEMPLATE_LIST_TEST_CASE("a bounded-storage ISRES accepts a population at its storage capacity", "[release-safety][isres]", bounded_variants)
{
    SECTION("the population option at the column capacity")
    {
        check_accepted(sized<TestType>(max_lambda, std::nullopt), plane(1), inside);
    }
    SECTION("the largest dimension whose default population fits")
    {
        using dynamic_policy = typename TestType::template rebind<dynamic_dimension>;
        const counted_box<dynamic_dimension> problem{.n = 88, .rows = 1, .evaluations = 0, .wrong_buffers = 0};
        check_accepted(sized<dynamic_policy>(std::nullopt, std::nullopt), problem, Eigen::VectorXd::Zero(88));
    }
}

TEMPLATE_LIST_TEST_CASE("an ISRES reset rejects a changed inequality count until it is restored", "[release-safety][isres]", isres_variants)
{
    auto problem = plane(1);
    step_budget_solver solver{sized<TestType>(std::nullopt, std::nullopt), problem, inside, solver_options<>{}};
    REQUIRE(solver.status() == solver_status::running);
    for(const std::int32_t rows : {0, 3})
    {
        problem.rows = rows;
        const std::int32_t evaluations = problem.evaluations;
        CHECK(solver.reset(inside) == solver_status::invalid_problem);
        CHECK(solver.reset_clear(inside) == solver_status::invalid_problem);
        CHECK(solver.status() == solver_status::invalid_problem);
        CHECK(solver.step().policy_status == solver_status::invalid_problem);
        CHECK(problem.evaluations == evaluations);
    }
    problem.rows = 1;
    CHECK(solver.reset(inside) == solver_status::running);
    CHECK(solver.step().policy_status != solver_status::invalid_problem);
    CHECK(problem.wrong_buffers == 0);
}
