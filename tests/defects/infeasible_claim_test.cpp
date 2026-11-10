#include "oracle/kkt_measure.h"
#include "oracle/result_oracle.h"

#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/filter_slsqp_policy.h"

#include "argmin/solver/alternative/isres/nlopt_faithful_policy.h"
#include "argmin/solver/alternative/isres/original_argmin_policy.h"
#include "argmin/solver/alternative/isres/runarsson_yao_paper_policy.h"

#include "argmin/solver/alternative/cmaes/repair_l2_penalty_policy.h"
#include "argmin/solver/alternative/cmaes/pwq_reparameterization_policy.h"
#include "argmin/solver/alternative/cmaes/no_repair_adaptive_penalty_policy.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <cstdint>

using namespace argmin;

namespace
{

// x0 >= 1 and x0 <= -1 on [-5, 5]^2: no point is feasible, and the least violation is 1 at x0 = 0.
struct split_rows
{
    static constexpr std::int32_t problem_dimension = 2;

    std::int32_t dimension() const { return 2; }
    std::int32_t num_equality() const { return 0; }
    std::int32_t num_inequality() const { return 2; }

    double value(const Eigen::Vector2d& x) const { return x[0] * x[0] + (x[1] - 1.0) * (x[1] - 1.0); }

    void gradient(const Eigen::Vector2d& x, Eigen::Vector2d& g) const
    {
        g[0] = 2.0 * x[0];
        g[1] = 2.0 * (x[1] - 1.0);
    }

    void constraints(const Eigen::Vector2d& x, auto& c) const
    {
        c[0] = x[0] - 1.0;
        c[1] = -x[0] - 1.0;
    }

    void constraint_jacobian(const Eigen::Vector2d&, auto& J) const { J << 1.0, 0.0, -1.0, 0.0; }

    Eigen::Vector2d lower_bounds() const { return Eigen::Vector2d::Constant(-5.0); }
    Eigen::Vector2d upper_bounds() const { return Eigen::Vector2d::Constant(5.0); }
};

solver_options<> split_rows_options()
{
    solver_options<> opts;
    opts.max_iterations = 2000;
    return opts;
}

template <typename Policy>
step_budget_solver<Policy, 2, split_rows> make_solver(const Policy& policy)
{
    static const split_rows problem{};
    return {policy, problem, Eigen::Vector2d{0.5, 0.5}, split_rows_options()};
}

template <typename Policy>
Policy seeded()
{
    Policy policy;
    policy.options.seed = 20261028;
    return policy;
}

double violation(const Eigen::Vector2d& x)
{
    return oracle::measure_kkt(split_rows{}, x, 0.0).inequality_violation;
}

}

TEST_CASE("a policy whose own stopping test ends at a point of an infeasible problem makes no claim", "[sqp][derivative_free]")
{
    auto filter_slsqp = make_solver(filter_slsqp_policy<double, 2>{});
    CHECK(filter_slsqp.solve().status == solver_status::objective_stalled);
    auto cobyla = make_solver(cobyla_policy<double>{});
    CHECK(cobyla.solve().status == solver_status::objective_stalled);
}

TEMPLATE_TEST_CASE("a stochastic search reports the constraint violation of the point it holds",
                   "[stochastic]",
                   (alternative::isres::nlopt_faithful_policy<double, 2>),
                   (alternative::isres::original_argmin_policy<double, 2>),
                   (alternative::isres::runarsson_yao_paper_policy<double, 2>),
                   (alternative::cmaes::repair_l2_penalty_policy<double, 2>),
                   (alternative::cmaes::pwq_reparameterization_policy<double, 2>),
                   (alternative::cmaes::no_repair_adaptive_penalty_policy<double, 2>))
{
    auto solver = make_solver(seeded<TestType>());
    std::int32_t misreported = 0;
    for(std::int32_t k = 0; k < 50; ++k)
        misreported += solver.step().constraint_violation == violation(solver.state().x) ? 0 : 1;
    CHECK(misreported == 0);
    const auto result = solver.solve();
    CHECK(result.constraint_violation == violation(result.x));
    CHECK(oracle::claim_of(result.status) == oracle::claim::none);
}
