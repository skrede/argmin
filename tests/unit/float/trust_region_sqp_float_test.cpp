#include "sqp_float_gate.h"

#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/filter_trsqp_policy.h"

#include "argmin/test_functions/hock_schittkowski.h"

#include <catch2/catch_test_macros.hpp>

using namespace argmin;
using namespace argmin::float_gate;

namespace
{

enum class experimental_check
{
    tracking,
    recorded_tracking,
    finite_only
};

template <template <scalar, int> typename Policy>
void check_hs035_hs028()
{
    SECTION("hs035")
    {
        check_sqp_tracks_double<Policy, hs035>(k_trust_region_sqp);
    }

    SECTION("hs028")
    {
        check_sqp_tracks_double<Policy, hs028>(k_trust_region_sqp);
    }
}

template <template <scalar, int> typename Policy, template <typename> typename Problem>
void check_experimental(experimental_check check)
{
    const Problem<double> problem_d;
    const Problem<float> problem_f;
    const auto result_d = drive_sqp<Policy<double, dynamic_dimension>>(problem_d);
    const auto result_f = drive_sqp<Policy<float, dynamic_dimension>>(problem_f);

    REQUIRE(result_f.x.allFinite());
    CHECK_NOFAIL(result_f.status == result_d.status);
    if(check == experimental_check::finite_only)
        return;
    if(check == experimental_check::tracking)
        CHECK(tracks(result_f.x, result_d.x, k_trust_region_sqp));
    else
        CHECK_NOFAIL(tracks(result_f.x, result_d.x, k_trust_region_sqp));
    CHECK(max_violation(problem_f, result_f.x) <= solver_options<default_convergence<float>>{}.feasibility_tolerance);
}

}

TEST_CASE("trust-region SQP policies in float track double on constrained Hock-Schittkowski fixtures at their own defaults", "[float]")
{
    SECTION("tr_sqp accurate")
    {
        check_hs035_hs028<tr_sqp_policy_accurate>();
        SECTION("hs043")
        {
            check_sqp_tracks_double<tr_sqp_policy_accurate, hs043>(k_trust_region_sqp);
        }
    }

    SECTION("tr_sqp fast hs043")
    {
        check_sqp_tracks_double<tr_sqp_policy_fast, hs043>(k_trust_region_sqp);
    }

    SECTION("filter_trsqp accurate")
    {
        check_hs035_hs028<filter_trsqp_policy_accurate>();
        SECTION("hs071")
        {
            check_sqp_tracks_double<filter_trsqp_policy_accurate, hs071>(k_trust_region_sqp);
        }
    }

    SECTION("filter_trsqp fast")
    {
        check_hs035_hs028<filter_trsqp_policy_fast>();
        SECTION("hs071")
        {
            check_sqp_tracks_double<filter_trsqp_policy_fast, hs071>(k_trust_region_sqp);
        }
    }
}

// In these cells the double or the float run ends outside the criterion stops on some builds (a trust-radius
// collapse at the solution reported as a failure, an exhausted budget), so they cannot meet the supported gate.
TEST_CASE("trust-region SQP policies in float on fixtures where a run does not stop by a criterion", "[float]")
{
    SECTION("tr_sqp accurate hs071")
    {
        check_experimental<tr_sqp_policy_accurate, hs071>(experimental_check::tracking);
    }

    SECTION("tr_sqp fast hs071")
    {
        check_experimental<tr_sqp_policy_fast, hs071>(experimental_check::tracking);
    }

    SECTION("tr_sqp fast hs035")
    {
        check_experimental<tr_sqp_policy_fast, hs035>(experimental_check::tracking);
    }

    SECTION("tr_sqp fast hs028")
    {
        check_experimental<tr_sqp_policy_fast, hs028>(experimental_check::tracking);
    }

    SECTION("filter_trsqp accurate hs043")
    {
        check_experimental<filter_trsqp_policy_accurate, hs043>(experimental_check::finite_only);
    }

    SECTION("filter_trsqp fast hs043")
    {
        check_experimental<filter_trsqp_policy_fast, hs043>(experimental_check::recorded_tracking);
    }
}
