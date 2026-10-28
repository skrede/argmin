#ifndef HPP_GUARD_ARGMIN_TESTS_FUZZ_QP_INPUT_H
#define HPP_GUARD_ARGMIN_TESTS_FUZZ_QP_INPUT_H

#include "harness.h"

#include "argmin/qp/qp_types.h"

#include <Eigen/Core>

#include <array>
#include <cmath>
#include <limits>
#include <cstddef>
#include <cstdint>
#include <algorithm>

namespace argmin::fuzzing
{

enum class hessian_shape : std::uint8_t
{
    definite,
    semidefinite,
    symmetric
};

struct qp_problem
{
    Eigen::MatrixXd P;
    Eigen::VectorXd q;
    Eigen::MatrixXd A;
    Eigen::VectorXd l;
    Eigen::VectorXd u;
    Eigen::VectorXd x_f;
    hessian_shape shape;
};

class error_set
{
public:
    error_set() : m_mandatory(false), m_allowed(0) {}

    void demand(qp_error error)
    {
        allow(error);
        m_mandatory = true;
    }

    void allow(qp_error error) { m_allowed |= std::uint32_t(1) << static_cast<std::uint32_t>(error); }

    bool admits(qp_error error) const { return (m_allowed & (std::uint32_t(1) << static_cast<std::uint32_t>(error))) != 0; }

    bool mandatory() const { return m_mandatory; }

private:
    bool m_mandatory;
    std::uint32_t m_allowed;
};

// A raw eight-byte double is almost always of extreme magnitude, so data is decoded as a signed mantissa over a
// bounded binary exponent; non-finite values enter only through inject().
inline double bounded(field_reader& fields)
{
    const auto mantissa = static_cast<std::int8_t>(fields.byte());
    const std::int32_t exponent = fields.byte() % 13 - 6;
    return std::ldexp(mantissa / 16.0, exponent);
}

inline Eigen::MatrixXd bounded_matrix(field_reader& fields, std::int32_t rows, std::int32_t cols)
{
    Eigen::MatrixXd values(rows, cols);
    for(Eigen::Index i = 0; i < values.size(); ++i)
        values.data()[i] = bounded(fields);
    return values;
}

inline Eigen::MatrixXd hessian(field_reader& fields, std::int32_t n, hessian_shape shape, std::int32_t rank)
{
    Eigen::MatrixXd factor = bounded_matrix(fields, n, n);
    if(shape == hessian_shape::symmetric)
        return factor.selfadjointView<Eigen::Lower>();
    if(shape == hessian_shape::semidefinite)
        factor.rightCols(n - rank).setZero();
    const Eigen::MatrixXd gram = factor * factor.transpose();
    Eigen::MatrixXd P = 0.5 * (gram + gram.transpose());
    if(shape == hessian_shape::definite && n > 0)
        P.diagonal().array() += 1e-2 * (1.0 + P.diagonal().maxCoeff());
    return P;
}

// Every shape but one holds the centre A x_f; the exception decodes its interval apart from it, so the rows can be
// jointly infeasible.
inline void bound_row(std::uint8_t row_shape, double centre, double w1, double w2, double& l, double& u)
{
    constexpr double inf = std::numeric_limits<double>::infinity();
    const std::array<double, 6> lower{centre, -inf, centre - std::abs(w1), -inf, w1 - std::abs(w2), centre - std::abs(w1)};
    const std::array<double, 6> upper{centre, centre + std::abs(w2), inf, inf, w1 + std::abs(w2), centre + std::abs(w2)};
    const std::size_t shape = std::min<std::size_t>(row_shape % 8, lower.size() - 1);
    l = lower[shape];
    u = upper[shape];
}

inline void decode_bounds(field_reader& fields, const Eigen::VectorXd& centre, Eigen::VectorXd& l, Eigen::VectorXd& u)
{
    l.resize(centre.size());
    u.resize(centre.size());
    for(Eigen::Index i = 0; i < centre.size(); ++i)
    {
        const std::uint8_t row_shape = fields.byte();
        const double w1 = bounded(fields);
        bound_row(row_shape, centre[i], w1, bounded(fields), l[i], u[i]);
    }
}

// The two selector bytes are read whether or not the fault is enabled, so the flag never shifts the fields
// behind it.
template <typename Derived>
void inject(field_reader& fields, bool enabled, Eigen::DenseBase<Derived>& target)
{
    constexpr std::array<double, 3> specials{std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
    const std::uint8_t position = fields.byte();
    const std::uint8_t kind = fields.byte();
    if(enabled && target.size() > 0)
        target.coeffRef(position % target.size()) = specials[kind % specials.size()];
}

inline void inject_vector_faults(field_reader& fields, Eigen::VectorXd& q, Eigen::VectorXd& l, Eigen::VectorXd& u)
{
    const std::uint8_t faults = fields.byte();
    inject(fields, (faults & 1) != 0, q);
    inject(fields, (faults & 2) != 0, l);
    inject(fields, (faults & 4) != 0, u);
    const std::uint8_t crossed = fields.byte();
    if((faults & 8) != 0 && l.size() > 0)
    {
        l[crossed % l.size()] = 1.0;
        u[crossed % u.size()] = -1.0;
    }
}

inline qp_problem decode_problem(field_reader& fields, std::uint8_t shape_count)
{
    const std::int32_t n = fields.byte() % 9;
    const std::int32_t m = fields.byte() % 9;
    const std::uint8_t kind = fields.byte();
    const auto shape = static_cast<hessian_shape>(kind % shape_count);
    qp_problem p{hessian(fields, n, shape, kind / 3 % (n + 1)), bounded_matrix(fields, n, 1), bounded_matrix(fields, m, n), {}, {}, bounded_matrix(fields, n, 1), shape};
    decode_bounds(fields, p.A * p.x_f, p.l, p.u);
    const std::uint8_t faults = fields.byte();
    inject(fields, (faults & 1) != 0, p.P);
    inject(fields, (faults & 2) != 0, p.A);
    inject_vector_faults(fields, p.q, p.l, p.u);
    return p;
}

inline void vector_faults(const Eigen::VectorXd& q, const Eigen::VectorXd& l, const Eigen::VectorXd& u, error_set& justified)
{
    if(!q.allFinite() || l.hasNaN() || u.hasNaN())
        justified.demand(qp_error::non_finite_input);
    if((l.array() > u.array()).any())
        justified.demand(qp_error::invalid_bounds);
}

// A singular or indefinite P may be refused, or may fail the factorization the method relies on.
inline void hessian_faults(const qp_problem& p, error_set& justified)
{
    if(!p.P.allFinite() || !p.A.allFinite())
        justified.demand(qp_error::non_finite_input);
    if(p.shape != hessian_shape::definite)
    {
        justified.allow(qp_error::invalid_problem);
        justified.allow(qp_error::factorization_failed);
    }
}

}

#endif
