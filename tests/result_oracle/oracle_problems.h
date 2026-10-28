#ifndef HPP_GUARD_ARGMIN_TESTS_RESULT_ORACLE_ORACLE_PROBLEMS_H
#define HPP_GUARD_ARGMIN_TESTS_RESULT_ORACLE_ORACLE_PROBLEMS_H

#include "oracle/kkt_measure.h"

#include "argmin/result/status.h"
#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <limits>
#include <cstdint>

namespace argmin::oracle::fixtures
{

using vec1 = Eigen::Vector<double, 1>;
using vec2 = Eigen::Vector<double, 2>;

struct linear_free
{
    static constexpr std::int32_t problem_dimension = 1;

    double slope;

    std::int32_t dimension() const { return 1; }
    double value(const vec1& x) const { return slope * x[0]; }
    void gradient(const vec1&, vec1& g) const { g[0] = slope; }
};

struct linear_halfline
{
    static constexpr std::int32_t problem_dimension = 1;

    double slope;

    std::int32_t dimension() const { return 1; }
    std::int32_t num_equality() const { return 0; }
    std::int32_t num_inequality() const { return 1; }
    double value(const vec1& x) const { return slope * x[0]; }
    void gradient(const vec1&, vec1& g) const { g[0] = slope; }
    void constraints(const vec1& x, Eigen::VectorXd& c) const { c[0] = x[0] - 1.0; }
    void constraint_jacobian(const vec1&, Eigen::MatrixXd& J) const { J(0, 0) = 1.0; }
};

struct linear_box
{
    static constexpr std::int32_t problem_dimension = 1;

    double slope;

    std::int32_t dimension() const { return 1; }
    double value(const vec1& x) const { return slope * x[0]; }
    void gradient(const vec1&, vec1& g) const { g[0] = slope; }
    vec1 lower_bounds() const { return vec1::Constant(1.0); }
    vec1 upper_bounds() const { return vec1::Constant(std::numeric_limits<double>::infinity()); }
};

struct shifted_quadratic
{
    static constexpr std::int32_t problem_dimension = 2;

    std::int32_t dimension() const { return 2; }
    double value(const vec2& x) const { return (x - minimizer()).squaredNorm(); }
    void gradient(const vec2& x, vec2& g) const { g = 2.0 * (x - minimizer()); }
    vec2 minimizer() const { return vec2(1.0, -2.0); }
};

struct equality_quadratic
{
    static constexpr std::int32_t problem_dimension = 2;

    std::int32_t dimension() const { return 2; }
    std::int32_t num_equality() const { return 1; }
    std::int32_t num_inequality() const { return 0; }
    double value(const vec2& x) const { return x.squaredNorm(); }
    void gradient(const vec2& x, vec2& g) const { g = 2.0 * x; }
    void constraints(const vec2& x, Eigen::VectorXd& c) const { c[0] = x[0] + x[1] - 1.0; }
    void constraint_jacobian(const vec2&, Eigen::MatrixXd& J) const { J.row(0).setOnes(); }
};

struct residual_pair
{
    static constexpr std::int32_t problem_dimension = 2;

    std::int32_t dimension() const { return 2; }
    std::int32_t num_residuals() const { return 2; }
    double value(const vec2& x) const { return 0.5 * (x - vec2(1.0, 2.0)).squaredNorm(); }
    void residuals(const vec2& x, Eigen::VectorXd& r) const { r = x - vec2(1.0, 2.0); }
    void jacobian(const vec2&, Eigen::MatrixXd& J) const { J.setIdentity(); }
};

template <int N>
solve_result<double, N> planted(solver_status status, const Eigen::Vector<double, N>& x)
{
    solve_result<double, N> result;
    result.status = status;
    result.x = x;
    return result;
}

inline tolerances<double> unit_tolerances()
{
    return tolerances<double>(1e-6, 1e-6, 1e-8);
}

}

#endif
