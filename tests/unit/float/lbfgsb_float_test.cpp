#include "float_gate.h"
#include "float_fixtures.h"

#include "argmin/solver/options.h"
#include "argmin/solver/convergence.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/byrd_lbfgsb_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/test_functions/rosenbrock.h"

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
using plain_rosenbrock = rosenbrock<S, 2>;

constexpr std::uint32_t max_iterations = 1000;

template <typename Policy, typename Problem>
solve_result<typename Policy::scalar_type, 2> drive(const Problem& problem)
{
    solver_options<default_convergence<typename Policy::scalar_type>> opts{};
    opts.max_iterations = max_iterations;
    step_budget_solver solver{Policy{}, problem, problem.initial_point(), opts};
    return solver.solve();
}

template <template <typename, int> typename Policy, template <typename> typename Problem>
void check_tracks_double()
{
    const Problem<double> problem_d;
    const Problem<float> problem_f;
    const auto result_d = drive<Policy<double, dynamic_dimension>>(problem_d);
    const auto result_f = drive<Policy<float, dynamic_dimension>>(problem_f);

    REQUIRE(result_d.status == solver_status::converged);
    REQUIRE(result_f.status == solver_status::converged);
    CHECK(tracks(result_f.x, result_d.x, k_lbfgsb));
}

template <template <typename, int> typename Policy>
void check_active_bound()
{
    const active_box_rosenbrock<double> problem_d;
    const active_box_rosenbrock<float> problem_f;
    const auto result_d = drive<Policy<double, dynamic_dimension>>(problem_d);
    const auto result_f = drive<Policy<float, dynamic_dimension>>(problem_f);

    CHECK(result_d.x[0] == problem_d.upper_bounds()[0]);
    CHECK(result_f.x[0] == problem_f.upper_bounds()[0]);
}

template <template <typename, int> typename Policy>
void check_family()
{
    check_tracks_double<Policy, plain_rosenbrock>();
    check_tracks_double<Policy, active_box_rosenbrock>();
    check_active_bound<Policy>();
}

}

TEST_CASE("L-BFGS-B in float tracks double on an unconstrained and an active-bound Rosenbrock at its own defaults", "[float]")
{
    SECTION("lbfgsb_policy")
    {
        check_family<lbfgsb_policy>();
    }

    SECTION("byrd_lbfgsb_policy")
    {
        check_family<byrd_lbfgsb_policy>();
    }
}
