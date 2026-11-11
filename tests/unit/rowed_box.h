#ifndef HPP_GUARD_ARGMIN_TESTS_UNIT_ROWED_BOX_H
#define HPP_GUARD_ARGMIN_TESTS_UNIT_ROWED_BOX_H

#include <Eigen/Core>

#include <cstdint>

namespace argmin::rowed
{

// 0.5 |x|^2 as the residuals r = x on the [-5, 5]^2 box, with n_eq equality and n_ineq inequality copies of the row
// 10 - x0 - x1, counting every call into the problem.
struct rowed_box
{
    static constexpr std::int32_t problem_dimension = 2;

    std::int32_t n_eq;
    std::int32_t n_ineq;
    mutable std::int32_t evaluations;

    std::int32_t dimension() const { return 2; }

    std::int32_t num_equality() const { return n_eq; }

    std::int32_t num_inequality() const { return n_ineq; }

    std::int32_t num_residuals() const { return 2; }

    double value(const Eigen::Vector2d& x) const
    {
        ++evaluations;
        return 0.5 * x.squaredNorm();
    }

    void gradient(const Eigen::Vector2d& x, Eigen::Vector2d& g) const
    {
        ++evaluations;
        g = x;
    }

    void residuals(const Eigen::Vector2d& x, Eigen::VectorXd& r) const
    {
        ++evaluations;
        r = x;
    }

    void jacobian(const Eigen::Vector2d&, Eigen::MatrixXd& J) const
    {
        ++evaluations;
        J.setIdentity(2, 2);
    }

    template <typename Values>
    void constraints(const Eigen::Vector2d& x, Values& c) const
    {
        ++evaluations;
        c.setConstant(10.0 - x.sum());
    }

    void constraint_jacobian(const Eigen::Vector2d&, Eigen::MatrixXd& J) const
    {
        ++evaluations;
        J.setConstant(-1.0);
    }

    Eigen::Vector2d lower_bounds() const { return Eigen::Vector2d::Constant(-5.0); }

    Eigen::Vector2d upper_bounds() const { return Eigen::Vector2d::Constant(5.0); }
};

inline const Eigen::Vector2d start(0.5, 0.5);

template <typename Policy>
Policy seeded()
{
    Policy policy;
    if constexpr(requires { policy.options.seed; })
        policy.options.seed = 7u;
    else if constexpr(requires { policy.options.inner.seed; })
        policy.options.inner.seed = 7u;
    return policy;
}

}

#endif
