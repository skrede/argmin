#include "oracle_problems.h"

#include "oracle/result_oracle.h"

#include "argmin/result/status.h"
#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>

using namespace argmin;
using namespace argmin::oracle::fixtures;

namespace
{

constexpr double quiet_nan = std::numeric_limits<double>::quiet_NaN();

oracle::tolerances<double> zero_activity_tolerances()
{
    return oracle::tolerances<double>(1e-6, 0.0, 1e-8);
}

solve_result<double> dynamic_result(solver_status status, const Eigen::VectorXd& x)
{
    solve_result<double> result;
    result.status = status;
    result.x = x;
    return result;
}

}

TEST_CASE("a constraint exactly at zero is active under a zero activity tolerance", "[oracle]")
{
    const oracle::verdict<double> v = oracle::judge(linear_halfline{-1.0}, planted(solver_status::converged, vec1(1.0)), zero_activity_tolerances());
    CHECK(v.legs.dual_sign_breach == 1.0);
    CHECK(v.legs.stationarity == 0.0);
}

TEST_CASE("a point exactly on a bound is active under a zero activity tolerance", "[oracle]")
{
    const oracle::verdict<double> v = oracle::judge(linear_box{-1.0}, planted(solver_status::converged, vec1(1.0)), zero_activity_tolerances());
    CHECK(v.legs.dual_sign_breach == 1.0);
    CHECK(v.legs.bound_violation == 0.0);
}

TEST_CASE("a stationarity leg exactly at k times its tolerance passes on an n = 1 problem", "[oracle]")
{
    const double tol = std::ldexp(1.0, -20);
    const double at_limit = oracle::safety_factor * tol;
    const oracle::tolerances<double> tols(tol, 1e-6, 1e-8);
    const oracle::verdict<double> v = oracle::judge(linear_free{at_limit}, planted(solver_status::converged, vec1(0.0)), tols);
    CHECK(v.legs.stationarity == at_limit);
    CHECK(v.ok);
}

TEST_CASE("a stationarity leg one double above k times its tolerance fails", "[oracle]")
{
    const double tol = std::ldexp(1.0, -20);
    const double above = std::nextafter(oracle::safety_factor * tol, std::numeric_limits<double>::infinity());
    const oracle::tolerances<double> tols(tol, 1e-6, 1e-8);
    const oracle::verdict<double> v = oracle::judge(linear_free{above}, planted(solver_status::converged, vec1(0.0)), tols);
    CHECK(v.failing == "stationarity");
}

TEST_CASE("ftol_reached is judged against the objective criterion's stationarity tolerance", "[oracle]")
{
    const oracle::tolerances<double> tols(1e-2, 1e-6, 1e-8);
    const solve_result<double, 1> result = planted(solver_status::ftol_reached, vec1(0.0));
    CHECK(oracle::judge(linear_free{5e-8}, result, tols).ok);
    CHECK(oracle::judge(linear_free{1e-6}, result, tols).failing == "stationarity");
}

TEST_CASE("a feasibility claim ignores stationarity but not a bound violation", "[oracle]")
{
    CHECK(oracle::judge(linear_box{1.0}, planted(solver_status::stalled, vec1(3.0)), unit_tolerances()).ok);
    CHECK(oracle::judge(linear_box{1.0}, planted(solver_status::stalled, vec1(0.5)), unit_tolerances()).failing == "bound_violation");
}

TEST_CASE("an unconstrained problem reports zero feasibility legs", "[oracle]")
{
    const oracle::verdict<double> v = oracle::judge(shifted_quadratic{}, planted(solver_status::converged, shifted_quadratic{}.minimizer()), unit_tolerances());
    CHECK(v.legs.equality_violation == 0.0);
    CHECK(v.legs.inequality_violation == 0.0);
    CHECK(v.legs.bound_violation == 0.0);
    CHECK(v.ok);
}

TEST_CASE("a least-squares problem without a gradient is judged through J^T r", "[oracle]")
{
    CHECK(oracle::judge(residual_pair{}, planted(solver_status::converged, vec2(1.0, 2.0)), unit_tolerances()).ok);
    CHECK(oracle::judge(residual_pair{}, planted(solver_status::converged, vec2(1.1, 2.0)), unit_tolerances()).failing == "stationarity");
}

TEST_CASE("an empty x is rejected without evaluating the problem", "[oracle]")
{
    const oracle::verdict<double> v = oracle::judge(shifted_quadratic{}, dynamic_result(solver_status::converged, Eigen::VectorXd()), unit_tolerances());
    CHECK(v.failing == "dimension");
}

TEST_CASE("an x of the wrong length is rejected", "[oracle]")
{
    const oracle::verdict<double> v = oracle::judge(shifted_quadratic{}, dynamic_result(solver_status::max_iterations, Eigen::VectorXd::Zero(3)), unit_tolerances());
    CHECK(v.failing == "dimension");
}

TEST_CASE("a non-finite x is rejected under every claim", "[oracle]")
{
    const oracle::verdict<double> v = oracle::judge(shifted_quadratic{}, planted(solver_status::aborted, vec2(1.0, quiet_nan)), unit_tolerances());
    CHECK(v.failing == "finite");
}

TEST_CASE("a running status is rejected as unfinished", "[oracle]")
{
    const oracle::verdict<double> v = oracle::judge(shifted_quadratic{}, planted(solver_status::running, shifted_quadratic{}.minimizer()), unit_tolerances());
    CHECK(v.failing == "status");
}
