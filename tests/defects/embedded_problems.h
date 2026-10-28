#ifndef HPP_GUARD_ARGMIN_TESTS_DEFECTS_EMBEDDED_PROBLEMS_H
#define HPP_GUARD_ARGMIN_TESTS_DEFECTS_EMBEDDED_PROBLEMS_H

#include <Eigen/Core>

#include <cstdint>

namespace argmin::defects
{

// A convex quadratic with a cyclic coupling term, ME linear equality rows x_k + 0.5 x_{k+1} = 1.2 and MI rows
// 3 - x_k^2 >= 0.
template <std::int32_t N, std::int32_t ME, std::int32_t MI>
struct coupled_quadratic
{
    using vector_type = Eigen::Vector<double, N>;

    static constexpr std::int32_t problem_dimension = N;
    static constexpr std::int32_t constraint_count = ME + MI;

    std::int32_t dimension() const { return N; }

    std::int32_t num_equality() const { return ME; }

    std::int32_t num_inequality() const { return MI; }

    double value(const vector_type& x) const
    {
        double f = 0.0;
        for(std::int32_t i = 0; i < N; ++i)
            f += (x[i] - 1.0 - 0.01 * i) * (x[i] - 1.0 - 0.01 * i) + 0.1 * x[i] * x[(i + 1) % N];
        return f;
    }

    void gradient(const vector_type& x, vector_type& g) const
    {
        for(std::int32_t i = 0; i < N; ++i)
            g[i] = 2.0 * (x[i] - 1.0 - 0.01 * i) + 0.1 * x[(i + 1) % N] + 0.1 * x[(i + N - 1) % N];
    }

    template <typename Values>
    void constraints(const vector_type& x, Values& c) const
    {
        for(std::int32_t k = 0; k < ME; ++k)
            c[k] = x[k] + 0.5 * x[(k + 1) % N] - 1.2;
        for(std::int32_t k = 0; k < MI; ++k)
            c[ME + k] = 3.0 - x[k] * x[k];
    }

    template <typename Jacobian>
    void constraint_jacobian(const vector_type& x, Jacobian& J) const
    {
        J.setZero();
        for(std::int32_t k = 0; k < ME; ++k)
        {
            J(k, k) = 1.0;
            J(k, (k + 1) % N) += 0.5;
        }
        for(std::int32_t k = 0; k < MI; ++k)
            J(ME + k, k) = -2.0 * x[k];
    }

    vector_type initial_point() const { return vector_type::Constant(0.3); }
};

}

#endif
