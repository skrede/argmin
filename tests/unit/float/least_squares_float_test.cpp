#include "float_gate.h"
#include "float_fixtures.h"

#include "argmin/solver/options.h"
#include "argmin/solver/lm_policy.h"
#include "argmin/solver/convergence.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/projected_gn_policy.h"
#include "argmin/solver/projected_gradient_gn_policy.h"

#include "argmin/test_functions/small_dense.h"

#include "argmin/result/status.h"
#include "argmin/result/solve_result.h"

#include "argmin/types.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using namespace argmin;
using namespace argmin::float_gate;
using namespace argmin::float_fixtures;

namespace
{

template <scalar S>
using extended_rosenbrock_ls = sd_ls012<S>;

template <scalar S>
using unbounded_rosenbrock_ls = unbounded<rosenbrock_ls<S>, S>;

template <scalar S>
using unbounded_extended_rosenbrock_ls = unbounded<sd_ls012<S>, S>;

constexpr std::uint32_t max_iterations = 200;

template <typename Policy, typename Problem>
solve_result<typename Policy::scalar_type, Problem::problem_dimension> drive(const Problem& problem)
{
    solver_options<default_convergence<typename Policy::scalar_type>> opts{};
    opts.max_iterations = max_iterations;
    step_budget_solver solver{Policy{}, problem, problem.initial_point(), opts};
    return solver.solve();
}

bool stopped_by_criterion(solver_status status)
{
    return status == solver_status::converged || status == solver_status::stalled;
}

template <template <scalar, int> typename Policy, template <scalar> typename Problem, bool ActiveBound = false>
void check_tracks_double()
{
    const Problem<double> problem_d;
    const Problem<float> problem_f;
    const auto result_d = drive<Policy<double, dynamic_dimension>>(problem_d);
    const auto result_f = drive<Policy<float, dynamic_dimension>>(problem_f);

    REQUIRE(stopped_by_criterion(result_d.status));
    REQUIRE(stopped_by_criterion(result_f.status));
    CHECK(tracks(result_f.x, result_d.x, k_least_squares));
    if constexpr(ActiveBound)
    {
        CHECK(result_d.x[0] == problem_d.upper_bounds()[0]);
        CHECK(result_f.x[0] == problem_f.upper_bounds()[0]);
    }
}

template <template <scalar, int> typename Policy>
void check_projected()
{
    SECTION("rosenbrock_ls")
    {
        check_tracks_double<Policy, unbounded_rosenbrock_ls>();
    }

    SECTION("sd_ls012")
    {
        check_tracks_double<Policy, unbounded_extended_rosenbrock_ls>();
    }
}

}

TEST_CASE("least-squares policies in float track double on zero-residual and active-bound fixtures at their own defaults", "[float]")
{
    SECTION("lm_policy")
    {
        SECTION("rosenbrock_ls")
        {
            check_tracks_double<lm_policy, rosenbrock_ls>();
        }

        SECTION("sd_ls012")
        {
            check_tracks_double<lm_policy, extended_rosenbrock_ls>();
        }
    }

    SECTION("projected_gn_policy")
    {
        check_projected<projected_gn_policy>();

        SECTION("active_box_rosenbrock_ls")
        {
            check_tracks_double<projected_gn_policy, active_box_rosenbrock_ls, true>();
        }
    }

    SECTION("projected_gradient_gn_policy")
    {
        check_projected<projected_gradient_gn_policy>();
    }
}

// Along the active bound the full-space damped step, clipped by the projection, advances x1 only
// sublinearly, so the double run also ends on the iteration budget rather than a criterion.
TEST_CASE("projected-gradient Gauss-Newton in float on an active-bound fixture where a run does not stop by a criterion", "[float]")
{
    const active_box_rosenbrock_ls<double> problem_d;
    const active_box_rosenbrock_ls<float> problem_f;
    const auto result_d = drive<projected_gradient_gn_policy<double, dynamic_dimension>>(problem_d);
    const auto result_f = drive<projected_gradient_gn_policy<float, dynamic_dimension>>(problem_f);

    REQUIRE(result_f.x.allFinite());
    CHECK_NOFAIL(result_f.status == result_d.status);
    CHECK(tracks(result_f.x, result_d.x, k_least_squares));
    CHECK(result_d.x[0] == problem_d.upper_bounds()[0]);
    CHECK(result_f.x[0] == problem_f.upper_bounds()[0]);
}
