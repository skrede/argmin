#include "embedded_problems.h"

#include "argmin/solver/options.h"
#include "argmin/solver/convergence.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/time_budget_solver.h"
#include "argmin/solver/time_budget_options.h"

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdint>
#include <optional>

using namespace argmin;

namespace
{

// Every copy or move of it counts, so a solve path that duplicates the convergence object shows a nonzero count.
struct copy_counting_criterion
{
    using scalar_type = double;
    static inline std::uint32_t copies = 0;

    copy_counting_criterion() = default;

    copy_counting_criterion(const copy_counting_criterion&) { ++copies; }

    copy_counting_criterion& operator=(const copy_counting_criterion&)
    {
        ++copies;
        return *this;
    }

    std::optional<solver_status> check(const step_result<double>&, std::uint32_t) const { return std::nullopt; }
};

}

TEST_CASE("a time budget solve with default options takes at least one iteration", "[known-defect][!shouldfail]")
{
    const defects::coupled_quadratic<4, 0, 0> problem;
    time_budget_solver solver{lbfgsb_policy<double, 4>{}, problem, problem.initial_point(), time_budget_options<>{}};
    CHECK(solver.solve().iterations > 0);
}

TEST_CASE("a defaulted solve copies no convergence object", "[embedded]")
{
    const defects::coupled_quadratic<4, 0, 0> problem;
    solver_options<convergence_policy<copy_counting_criterion>> opts;
    opts.max_iterations = 20;
    step_budget_solver solver{lbfgsb_policy<double, 4>{}, problem, problem.initial_point(), opts};
    time_budget_solver timed{lbfgsb_policy<double, 4>{}, problem, problem.initial_point(), time_budget_options<convergence_policy<copy_counting_criterion>>{.core = opts, .max_time = std::chrono::hours(1)}};
    copy_counting_criterion::copies = 0;
    solver.step_n(5);
    solver.solve();
    timed.step_n(5);
    timed.solve();
    CHECK(copy_counting_criterion::copies == 0);
}
