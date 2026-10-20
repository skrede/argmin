#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/result/status.h"

#include "../cobyla_box_cases.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

using namespace argmin;
using namespace argmin::test;

namespace
{

template <typename P>
void check_float_case(const P& problem)
{
    box_record<float> record;
    const box_recorder<float, P> recorder{problem, record};
    step_budget_solver solver{cobyla_policy<float>{}, recorder, problem.initial_point(),
                              publication_options<float>()};
    const auto result = solver.solve();
    CHECK(record.value_calls > 0);
    CHECK(record.outside_calls == 0);
    CHECK(solver.state().clamped_evaluations == 0);
    CHECK(succeeded(result.status));
}

}

TEST_CASE("cobyla in float keeps every evaluation inside the box", "[float][cobyla][bounds]")
{
    SECTION("narrow-coordinate witness")
    {
        check_float_case(narrow_witness<float>());
    }
    SECTION("undershoot witness")
    {
        check_float_case(undershoot<float>());
    }
    SECTION("out-of-box start")
    {
        check_float_case(start_outside<float>());
    }
    SECTION("three-scale box")
    {
        check_float_case(three_scale<float>());
    }
}
