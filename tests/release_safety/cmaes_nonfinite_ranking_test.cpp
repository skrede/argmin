#include "eigen_trap.h"

#include "argmin/solver/options.h"

#include "argmin/solver/alternative/cmaes/repair_l2_penalty_policy.h"
#include "argmin/solver/alternative/cmaes/pwq_reparameterization_policy.h"
#include "argmin/solver/alternative/cmaes/no_repair_adaptive_penalty_policy.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <cmath>
#include <limits>
#include <cstddef>
#include <cstdint>

using namespace argmin;

namespace
{

using pwq = alternative::cmaes::pwq_reparameterization_policy<double, 2>;
using repair_l2 = alternative::cmaes::repair_l2_penalty_policy<double, 2>;
using no_repair = alternative::cmaes::no_repair_adaptive_penalty_policy<double, 2>;

// |x|^2 on [-5, 5]^2 except on the half plane x0 > 0, which about half of every generation samples.
struct half_plane_box
{
    using vector = Eigen::Vector2d;

    static constexpr std::int32_t problem_dimension = 2;

    double off_plane;

    std::int32_t dimension() const { return 2; }

    double value(const vector& x) const { return x[0] > 0.0 ? off_plane : x.squaredNorm(); }

    vector lower_bounds() const { return vector::Constant(-5.0); }

    vector upper_bounds() const { return vector::Constant(5.0); }
};

constexpr std::int32_t generations = 10;

const Eigen::Vector2d start(-0.25, 0.5);

template <typename Policy>
Policy seeded()
{
    Policy policy;
    policy.options.seed = 11u;
    return policy;
}

template <typename State>
double ranked_fitness(const State& s, std::size_t rank)
{
    return s.fitnesses_buf[s.idx_buf[rank]];
}

template <typename State>
bool ranked_finite_first(const State& s)
{
    for(std::size_t r = 1; r < static_cast<std::size_t>(s.params.lambda); ++r)
    {
        const double ahead = ranked_fitness(s, r - 1);
        const double behind = ranked_fitness(s, r);
        if(std::isfinite(behind) && (!std::isfinite(ahead) || behind < ahead))
            return false;
    }
    return true;
}

template <typename State>
bool mixed(const State& s)
{
    const auto lambda = static_cast<std::size_t>(s.params.lambda);
    return std::isfinite(ranked_fitness(s, 0)) != std::isfinite(ranked_fitness(s, lambda - 1));
}

template <typename Policy>
void check_ranked_finite_first(double off_plane)
{
    const half_plane_box problem{off_plane};
    auto policy = seeded<Policy>();
    auto s = policy.init(problem, start, solver_options<>{});
    std::int32_t mixed_generations = 0;
    for(std::int32_t k = 0; k < generations; ++k)
    {
        policy.step(s);
        CHECK(ranked_finite_first(s));
        CHECK(s.mean.allFinite());
        mixed_generations += mixed(s) ? 1 : 0;
    }
    CHECK(mixed_generations > 0);
}

// A ranking change moves these by orders of magnitude more than the cross-platform libm and FMA drift the margin absorbs.
template <typename Policy>
void check_finite_trajectory(double mean0, double mean1, double sigma)
{
    const half_plane_box problem{25.0};
    auto policy = seeded<Policy>();
    auto s = policy.init(problem, start, solver_options<>{});
    for(std::int32_t k = 0; k < generations; ++k)
        policy.step(s);
    CHECK(s.mean[0] == Catch::Approx(mean0).epsilon(0).margin(1e-12));
    CHECK(s.mean[1] == Catch::Approx(mean1).epsilon(0).margin(1e-12));
    CHECK(s.sigma == Catch::Approx(sigma).epsilon(0).margin(1e-12));
}

}

TEMPLATE_TEST_CASE("a CMA-ES generation ranks NaN objective values after every finite one", "[release-safety][cmaes]", pwq, repair_l2, no_repair)
{
    check_ranked_finite_first<TestType>(std::numeric_limits<double>::quiet_NaN());
}

TEMPLATE_TEST_CASE("a CMA-ES generation ranks infinite objective values after every finite one", "[release-safety][cmaes]", pwq, repair_l2, no_repair)
{
    SECTION("positive infinity")
    {
        check_ranked_finite_first<TestType>(std::numeric_limits<double>::infinity());
    }
    SECTION("negative infinity")
    {
        check_ranked_finite_first<TestType>(-std::numeric_limits<double>::infinity());
    }
}

TEST_CASE("the pwq CMA-ES variant keeps its finite-only trajectory", "[release-safety][cmaes]")
{
    check_finite_trajectory<pwq>(-0x1.2dca7bd856c89p-1, -0x1.f695d912e2192p-1, 0x1.b5090867f3e3dp+0);
}

TEST_CASE("the repair-l2 CMA-ES variant keeps its finite-only trajectory", "[release-safety][cmaes]")
{
    check_finite_trajectory<repair_l2>(-0x1.dd96d47cc282bp-2, -0x1.c4f4172b4a716p-3, 0x1.69f2f1809ea5dp+0);
}

TEST_CASE("the no-repair CMA-ES variant keeps its finite-only trajectory", "[release-safety][cmaes]")
{
    check_finite_trajectory<no_repair>(-0x1.dd96d47cc282bp-2, -0x1.c4f4172b4a716p-3, 0x1.69f2f1809ea5dp+0);
}
