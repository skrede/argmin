#include "argmin/detail/kkt_residual.h"

#include "argmin/solver/options.h"
#include "argmin/solver/mma_policy.h"
#include "argmin/solver/convergence.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"
#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <optional>

using namespace argmin;

namespace
{

constexpr double threshold = 1e-8;

// min -1000 (x0 + x1) on the unit disk: the multiplier at the optimum is 1000 / sqrt(2), so an iterate whose
// violation is under the threshold can still be far from complementary.
struct scaled_disk
{
    static constexpr std::int32_t problem_dimension = 2;
    static constexpr std::int32_t constraint_count = 1;

    std::int32_t dimension() const { return 2; }
    std::int32_t num_equality() const { return 0; }
    std::int32_t num_inequality() const { return 1; }
    double value(const Eigen::Vector2d& x) const { return -1000.0 * (x[0] + x[1]); }
    void gradient(const Eigen::Vector2d&, Eigen::Vector2d& g) const { g.setConstant(-1000.0); }
    void constraints(const Eigen::Vector2d& x, auto& c) const { c[0] = 1.0 - x.squaredNorm(); }
    void constraint_jacobian(const Eigen::Vector2d& x, auto& J) const { J.row(0) = -2.0 * x.transpose(); }
    Eigen::Vector2d lower_bounds() const { return Eigen::Vector2d::Constant(-10.0); }
    Eigen::Vector2d upper_bounds() const { return Eigen::Vector2d::Constant(10.0); }
    Eigen::Vector2d initial_point() const { return {0.5, 0.2}; }
};

// The multiplier fitted to grad f = mu grad c at x, times c(x).
double complementarity_at(const Eigen::Vector2d& x)
{
    const Eigen::Vector2d grad_c = -2.0 * x;
    const double mu = grad_c.dot(Eigen::Vector2d::Constant(-1000.0)) / grad_c.squaredNorm();
    return std::abs(mu * (1.0 - x.squaredNorm()));
}

template <typename Policy>
solve_result<double, 2> solve_disk()
{
    solver_options<> opts{};
    opts.max_iterations = 200;
    opts.set_gradient_threshold(threshold);
    opts.set_objective_threshold(1e-16);
    opts.set_step_threshold(1e-16);
    const scaled_disk problem;
    step_budget_solver<Policy, 2, scaled_disk> solver(problem, problem.initial_point(), opts);
    return solver.solve(opts);
}

std::optional<solver_status> verdict_on(double mu, double c)
{
    const Eigen::Vector2d grad_f = Eigen::Vector2d::Zero();
    const Eigen::Matrix<double, 0, 2> J_eq;
    const Eigen::Matrix<double, 1, 2> J_ineq = Eigen::Matrix<double, 1, 2>::Zero();
    const Eigen::Vector<double, 0> none;
    const Eigen::Vector<double, 1> mu_ineq{{mu}};
    const Eigen::Vector<double, 1> c_ineq{{c}};
    step_result<double> r{};
    r.kkt_residual = detail::kkt_residual<double, 2, 0, 1>(grad_f, J_eq, J_ineq, none, mu_ineq, none, c_ineq);
    gradient_tolerance_criterion<double> criterion{};
    criterion.threshold = threshold;
    return criterion.check(r, 1);
}

// The box-only reading of a point `slack` below the upper bound with a multiplier of 1000 pushing out of the box.
std::optional<solver_status> bound_verdict_on(double slack)
{
    const Eigen::Vector2d x{1.0 - slack, 0.0};
    const Eigen::Vector2d g{-1000.0, 0.0};
    step_result<double> r{};
    r.kkt_residual = detail::kkt_residual_bound<double, 2>(x, g, Eigen::Vector2d::Constant(-1.0), Eigen::Vector2d::Constant(1.0));
    gradient_tolerance_criterion<double> criterion{};
    criterion.threshold = threshold;
    return criterion.check(r, 1);
}

// min -1e11 x on [-1, 1]: the bound's multiplier is 1e11, so a start two ulps below the bound, a step L-BFGS-B's
// zero-step guard refuses, is 2.2e-5 from complementary while its projected gradient reads 2.2e-16.
struct steep_ramp
{
    static constexpr std::int32_t problem_dimension = 1;

    std::int32_t dimension() const { return 1; }
    double value(const Eigen::Vector<double, 1>& x) const { return -1e11 * x[0]; }
    void gradient(const Eigen::Vector<double, 1>&, Eigen::Vector<double, 1>& g) const { g[0] = -1e11; }
    Eigen::Vector<double, 1> lower_bounds() const { return Eigen::Vector<double, 1>::Constant(-1.0); }
    Eigen::Vector<double, 1> upper_bounds() const { return Eigen::Vector<double, 1>::Constant(1.0); }
};

}

TEST_CASE("the complementarity leg is the product of the multiplier and the constraint value", "[kkt_residual]")
{
    const Eigen::Vector<double, 2> mu{{1000.0, 0.5}};
    const Eigen::Vector<double, 2> c{{1e-10, -0.25}};
    CHECK(detail::complementarity_leg(mu, c) == 0.125);
    CHECK(detail::complementarity_leg(Eigen::Vector2d(1000.0, 0.0), Eigen::Vector2d(1e-10, 0.25)) == 1000.0 * 1e-10);
}

TEST_CASE("a point whose multiplier-constraint product exceeds the threshold makes no optimality claim", "[kkt_residual]")
{
    SECTION("min(|mu|, |c|) under the threshold, |mu c| above it")
    {
        CHECK_FALSE(verdict_on(1000.0, 1e-10).has_value());
    }
    SECTION("control: |mu c| under the threshold")
    {
        CHECK(verdict_on(1000.0, 1e-12) == solver_status::converged);
    }
}

TEST_CASE("a converged claim on the scaled disk is complementary within the threshold", "[kkt_residual]")
{
    SECTION("mma")
    {
        const auto r = solve_disk<mma_policy<double, 2>>();
        CHECK((r.status != solver_status::converged || complementarity_at(r.x) <= threshold));
    }
    SECTION("tr_sqp accurate")
    {
        const auto r = solve_disk<tr_sqp_policy_accurate<double, 2>>();
        CHECK((r.status != solver_status::converged || complementarity_at(r.x) <= threshold));
    }
    SECTION("control: nw_sqp claims converged at a complementary point")
    {
        const auto r = solve_disk<nw_sqp_policy<double, 2>>();
        CHECK(r.status == solver_status::converged);
        CHECK(complementarity_at(r.x) <= threshold);
    }
}

TEST_CASE("a box-only point whose bound multiplier-slack product exceeds the threshold makes no optimality claim", "[kkt_residual]")
{
    SECTION("projected gradient under the threshold, multiplier-slack product above it")
    {
        CHECK_FALSE(bound_verdict_on(1e-10).has_value());
    }
    SECTION("control: the product under the threshold")
    {
        CHECK(bound_verdict_on(1e-12) == solver_status::converged);
    }
    SECTION("L-BFGS-B at a point its zero-step guard does not move")
    {
        const steep_ramp problem;
        const Eigen::Vector<double, 1> x0{{std::nextafter(std::nextafter(1.0, 0.0), 0.0)}};
        solver_options<> opts{};
        opts.max_iterations = 3;
        step_budget_solver<lbfgsb_policy<double, 1>, 1, steep_ramp> solver(problem, x0, opts);
        const auto r = solver.solve();
        CHECK(r.status != solver_status::converged);
        CHECK(r.status != solver_status::ftol_reached);
    }
}
