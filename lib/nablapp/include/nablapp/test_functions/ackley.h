#ifndef HPP_GUARD_NABLAPP_TEST_FUNCTIONS_ACKLEY_H
#define HPP_GUARD_NABLAPP_TEST_FUNCTIONS_ACKLEY_H

#include "nablapp/test_functions/problem_class.h"

#include <Eigen/Core>

#include <cmath>
#include <numbers>

namespace nablapp
{

// n-dimensional Ackley function.
//
// K&W Appendix B.1, Eq. B.1.
//
//   f(x) = -a * exp(-b * sqrt(sum(x_i^2) / d))
//         - exp(sum(cos(c * x_i)) / d)
//         + a + exp(1)
//
// Default parameters: a = 20, b = 0.2, c = 2*pi (K&W convention).
// Global minimum: x* = 0, f(x*) = 0.

template <typename Scalar = double>
struct ackley
{
    Scalar a{20};
    Scalar b{Scalar(0.2)};
    Scalar c{Scalar(2) * std::numbers::pi_v<Scalar>};
    int n{2};

    static constexpr problem_class pclass = problem_class::global | problem_class::bound_constrained;

    [[nodiscard]] int dimension() const { return n; }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }

    [[nodiscard]] Eigen::VectorX<Scalar> initial_point() const
    {
        return Eigen::VectorX<Scalar>::Ones(n);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> lower_bounds() const
    {
        return Eigen::VectorX<Scalar>::Constant(n, Scalar(-32.768));
    }

    [[nodiscard]] Eigen::VectorX<Scalar> upper_bounds() const
    {
        return Eigen::VectorX<Scalar>::Constant(n, Scalar(32.768));
    }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        Scalar d = Scalar(n);
        Scalar sum_sq{0};
        Scalar sum_cos{0};
        for(int i = 0; i < n; ++i)
        {
            sum_sq += x[i] * x[i];
            sum_cos += std::cos(c * x[i]);
        }
        return -a * std::exp(-b * std::sqrt(sum_sq / d))
             - std::exp(sum_cos / d)
             + a + std::exp(Scalar(1));
    }

    void gradient(const Eigen::VectorX<Scalar>& x, Eigen::VectorX<Scalar>& g) const
    {
        Scalar d = Scalar(n);
        Scalar sum_sq{0};
        Scalar sum_cos{0};
        for(int i = 0; i < n; ++i)
        {
            sum_sq += x[i] * x[i];
            sum_cos += std::cos(c * x[i]);
        }

        Scalar norm = std::sqrt(sum_sq / d);
        // d/dx_i of first term: -a * exp(-b*norm) * (-b) * x_i / (d * norm)
        //                      = a * b * exp(-b*norm) * x_i / (d * norm)
        // When norm == 0, gradient of first term is 0 (L'Hopital).
        Scalar exp_term1 = (norm > Scalar(0))
            ? a * b * std::exp(-b * norm) / (d * norm)
            : Scalar(0);
        // d/dx_i of second term: exp(sum_cos/d) * sin(c*x_i) * c / d
        Scalar exp_term2 = std::exp(sum_cos / d) / d;

        g.resize(n);
        for(int i = 0; i < n; ++i)
        {
            g[i] = exp_term1 * x[i] + exp_term2 * c * std::sin(c * x[i]);
        }
    }
};

}

#endif
