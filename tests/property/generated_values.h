#ifndef HPP_GUARD_ARGMIN_TESTS_PROPERTY_GENERATED_VALUES_H
#define HPP_GUARD_ARGMIN_TESTS_PROPERTY_GENERATED_VALUES_H

#include <Eigen/Core>

#include <rapidcheck.h>

#include <cstdint>

namespace argmin::property
{

// Values on a grid of step 1e-3 in [lo, hi] keep every generated problem well scaled.
inline double pick_value(std::int32_t lo, std::int32_t hi)
{
    return static_cast<double>(*rc::gen::inRange<std::int32_t>(1000 * lo, 1000 * hi + 1)) / 1000.0;
}

inline Eigen::Index pick_size(std::int32_t lo, std::int32_t hi)
{
    return static_cast<Eigen::Index>(*rc::gen::inRange<std::int32_t>(lo, hi + 1));
}

inline Eigen::VectorXd pick_vector(Eigen::Index size)
{
    Eigen::VectorXd v(size);
    for(Eigen::Index i = 0; i < size; ++i)
        v[i] = pick_value(-1, 1);
    return v;
}

inline Eigen::MatrixXd pick_matrix(Eigen::Index rows, Eigen::Index cols)
{
    Eigen::MatrixXd m(rows, cols);
    for(Eigen::Index j = 0; j < cols; ++j)
        m.col(j) = pick_vector(rows);
    return m;
}

// B B^T + 0.1 I has its smallest eigenvalue at least 0.1, so every generated objective is strongly convex.
inline Eigen::MatrixXd pick_spd(Eigen::Index n)
{
    const Eigen::MatrixXd b = pick_matrix(n, n);
    return b * b.transpose() + 0.1 * Eigen::MatrixXd::Identity(n, n);
}

}

#endif
