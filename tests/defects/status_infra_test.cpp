#include "argmin/solver/stepper.h"
#include "argmin/solver/lm_policy.h"

#include "argmin/result/status.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>

using namespace argmin;

namespace
{

using vector1 = Eigen::Vector<double, 1>;

// The first Levenberg-Marquardt trial from x = 1 lands at x < 0, where the residual is an exact NaN.
struct sqrt_residual
{
    static constexpr std::int32_t problem_dimension = 1;

    std::int32_t dimension() const { return 1; }

    std::int32_t num_residuals() const { return 1; }

    double value(const vector1& x) const
    {
        const double r = std::sqrt(x[0]) - 0.1;
        return 0.5 * r * r;
    }

    void residuals(const vector1& x, Eigen::VectorXd& r) const
    {
        r.resize(1);
        r[0] = std::sqrt(x[0]) - 0.1;
    }

    void jacobian(const vector1& x, Eigen::MatrixXd& J) const
    {
        J.resize(1, 1);
        J(0, 0) = 0.5 / std::sqrt(x[0]);
    }
};

using lm_stepper = stepper<lm_policy<double, 1>, 1, sqrt_residual>;

lm_stepper stepped_into_nan(const sqrt_residual& problem)
{
    lm_stepper st{lm_policy<double, 1>{}, problem, vector1{1.0}, solver_options<>{}, lm_policy<double, 1>::options_type{}};
    st.step();
    return st;
}

}

TEST_CASE("stepper ends diverged after a step into a non-finite residual", "[status]")
{
    const sqrt_residual problem{};
    CHECK(stepped_into_nan(problem).status() == solver_status::diverged);
}

TEST_CASE("stepper converged() is false after a solve that ended diverged", "[known-defect][!shouldfail]")
{
    const sqrt_residual problem{};
    CHECK_FALSE(stepped_into_nan(problem).converged());
}
