#include "sqp_float_gate.h"

#include "argmin/solver/mma_policy.h"
#include "argmin/solver/ccsa_quadratic_policy.h"
#include "argmin/solver/alternative/gcmma/rho_wval_policy.h"
#include "argmin/solver/alternative/gcmma/raa_augmented_policy.h"
#include "argmin/solver/alternative/gcmma/move_limit_shrink_policy.h"

#include "argmin/test_functions/hock_schittkowski.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <algorithm>

using namespace argmin;
using namespace argmin::float_gate;

namespace
{

template <scalar S, int N>
using mma = mma_policy<S, N>;

template <scalar S, int N>
using ccsa_quadratic = ccsa_quadratic_policy<S, N>;

template <scalar S, int N>
using gcmma_rho_wval = alternative::gcmma::rho_wval_policy<S, N>;

template <scalar S, int N>
using gcmma_raa_augmented = alternative::gcmma::raa_augmented_policy<S, N>;

template <scalar S, int N>
using gcmma_move_limit_shrink = alternative::gcmma::move_limit_shrink_policy<S, N>;

// The moving-asymptote family stops successfully only through the harness criteria; on hs076 the
// double run ends on the step-tolerance stop, which fires only within the feasibility tolerance.
constexpr std::array accepted_statuses{solver_status::converged, solver_status::stalled};

bool accepted(solver_status status)
{
    return std::ranges::find(accepted_statuses, status) != accepted_statuses.end();
}

template <template <typename, int> typename Policy, template <typename> typename Problem>
void check_tracks_double()
{
    const Problem<double> problem_d;
    const Problem<float> problem_f;
    const auto result_d = drive_sqp<Policy<double, Problem<double>::problem_dimension>>(problem_d);
    const auto result_f = drive_sqp<Policy<float, Problem<float>::problem_dimension>>(problem_f);

    REQUIRE(accepted(result_d.status));
    REQUIRE(accepted(result_f.status));
    CHECK(tracks(result_f.x, result_d.x, k_mma));
    CHECK(max_violation(problem_f, result_f.x) <= solver_options<default_convergence<float>>{}.feasibility_tolerance);
}

template <template <typename, int> typename Policy, template <typename> typename First, template <typename> typename Second>
void check_family(const char* first, const char* second)
{
    SECTION(first)
    {
        check_tracks_double<Policy, First>();
    }

    SECTION(second)
    {
        check_tracks_double<Policy, Second>();
    }
}

}

TEST_CASE("moving-asymptote policies in float track double on inequality-constrained fixtures at their own defaults", "[float]")
{
    SECTION("mma_policy")
    {
        check_family<mma, hs076, hs035>("hs076", "hs035");
    }

    SECTION("ccsa_quadratic_policy")
    {
        check_family<ccsa_quadratic, hs076, hs035>("hs076", "hs035");
    }

    SECTION("gcmma rho_wval_policy")
    {
        check_family<gcmma_rho_wval, hs024, hs076>("hs024", "hs076");
    }

    SECTION("gcmma raa_augmented_policy")
    {
        check_family<gcmma_raa_augmented, hs024, hs076>("hs024", "hs076");
    }

    SECTION("gcmma move_limit_shrink_policy")
    {
        SECTION("hs024")
        {
            check_tracks_double<gcmma_move_limit_shrink, hs024>();
        }
    }
}

// On hs076 the double run advances only sublinearly (f = -3.06 after 200 iterations, -4.58 after
// 20000, against -4.6818), so neither run's stop compares two converged solutions.
TEST_CASE("GCMMA move-limit shrinking in float on hs076 where the double run does not stop by a criterion", "[float]")
{
    const hs076<double> problem_d;
    const hs076<float> problem_f;
    const auto result_d = drive_sqp<gcmma_move_limit_shrink<double, hs076<double>::problem_dimension>>(problem_d);
    const auto result_f = drive_sqp<gcmma_move_limit_shrink<float, hs076<float>::problem_dimension>>(problem_f);

    REQUIRE(result_f.x.allFinite());
    CHECK_NOFAIL(result_f.status == result_d.status);
    CHECK_NOFAIL(tracks(result_f.x, result_d.x, k_mma));
    CHECK(max_violation(problem_f, result_f.x) <= solver_options<default_convergence<float>>{}.feasibility_tolerance);
    CHECK(result_d.objective_value < -3.0);
    CHECK(result_f.objective_value < -3.0f);
}
