#ifndef HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_VALIDATE_H
#define HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_VALIDATE_H

// Every check below is an ordinary runtime conditional, and none of them may be
// rewritten as the standard debug-assertion macro. That macro's whole behavior
// is to compile to nothing once the release preprocessor symbol is defined, and
// that erasure is exactly how a sibling binding shipped an out-of-bounds read
// that only interpreter callers could reach: a C++ caller was stopped at compile
// time by the concept layer, so the elided checks were the only thing standing
// between a dynamically typed caller and undefined behavior.

#include "bindings/detail/errors.h"
#include "bindings/detail/format_number.h"

#include "argmin/types.h"

#include <Eigen/SparseCore>

#include <cmath>
#include <string>
#include <string_view>

namespace argmin::python
{

inline void check_vector_length(Eigen::Ref<const vector<double>> values, int expected,
                                std::string_view name)
{
    const int actual = static_cast<int>(values.size());
    if(actual == expected)
        return;
    raise_argmin_error(error_kind::dimension_mismatch,
                       std::string(name) + " has length " + format_number(actual)
                           + ", expected " + format_number(expected));
}

inline void check_matrix_shape(Eigen::Ref<const matrix<double>> values, int rows, int cols,
                               std::string_view name)
{
    const int actual_rows = static_cast<int>(values.rows());
    const int actual_cols = static_cast<int>(values.cols());
    if(actual_rows == rows && actual_cols == cols)
        return;
    raise_argmin_error(error_kind::dimension_mismatch,
                       std::string(name) + " has shape " + format_shape(actual_rows, actual_cols)
                           + ", expected " + format_shape(rows, cols));
}

inline void check_square(Eigen::Ref<const matrix<double>> values, std::string_view name)
{
    const int rows = static_cast<int>(values.rows());
    const int cols = static_cast<int>(values.cols());
    if(rows == cols)
        return;
    raise_argmin_error(error_kind::dimension_mismatch,
                       std::string(name) + " has shape " + format_shape(rows, cols)
                           + ", expected a square matrix");
}

inline void check_square_shape(int rows, int cols, std::string_view name)
{
    if(rows == cols)
        return;
    raise_argmin_error(error_kind::dimension_mismatch,
                       std::string(name) + " has shape " + format_shape(rows, cols)
                           + ", expected a square matrix");
}

inline void check_matrix_columns(int actual, int expected, std::string_view name)
{
    if(actual == expected)
        return;
    raise_argmin_error(error_kind::dimension_mismatch,
                       std::string(name) + " has " + format_number(actual)
                           + " columns, expected " + format_number(expected));
}

// Separate from check_all_finite because an infinite bound is how a one-sided
// constraint is spelled, while a bound that is not a number silently defeats
// every ordering comparison downstream.
inline void check_no_nan(Eigen::Ref<const vector<double>> values, std::string_view name)
{
    for(int index = 0; index < static_cast<int>(values.size()); ++index)
    {
        if(!std::isnan(values[index]))
            continue;
        raise_argmin_error(error_kind::non_finite_input,
                           std::string(name) + " is not a number at index "
                               + format_number(index) + " of length "
                               + format_number(static_cast<int>(values.size())));
    }
}

template <typename Derived>
inline void check_all_finite(const Eigen::DenseBase<Derived>& values, std::string_view name)
{
    const int rows = static_cast<int>(values.rows());
    const int cols = static_cast<int>(values.cols());
    for(int col = 0; col < cols; ++col)
    {
        for(int row = 0; row < rows; ++row)
        {
            if(std::isfinite(values.coeff(row, col)))
                continue;
            if(rows == 1 || cols == 1)
            {
                const int index = rows == 1 ? col : row;
                raise_argmin_error(error_kind::non_finite_input,
                                   std::string(name) + " has a non-finite entry at index "
                                       + format_number(index) + " of length "
                                       + format_number(rows == 1 ? cols : rows));
            }
            raise_argmin_error(error_kind::non_finite_input,
                               std::string(name) + " has a non-finite entry at row "
                                   + format_number(row) + ", column " + format_number(col)
                                   + " of shape " + format_shape(rows, cols));
        }
    }
}

template <typename Derived>
inline void check_all_finite(const Eigen::SparseMatrixBase<Derived>& values, std::string_view name)
{
    const Derived& stored = values.derived();
    for(int outer = 0; outer < static_cast<int>(stored.outerSize()); ++outer)
    {
        for(typename Derived::InnerIterator entry(stored, outer); entry; ++entry)
        {
            if(std::isfinite(entry.value()))
                continue;
            raise_argmin_error(error_kind::non_finite_input,
                               std::string(name) + " has a non-finite stored entry at row "
                                   + format_number(static_cast<int>(entry.row())) + ", column "
                                   + format_number(static_cast<int>(entry.col())) + " of shape "
                                   + format_shape(static_cast<int>(stored.rows()),
                                                  static_cast<int>(stored.cols())));
        }
    }
}

inline void check_ordered_bounds(Eigen::Ref<const vector<double>> lower,
                                 Eigen::Ref<const vector<double>> upper)
{
    if(lower.size() != upper.size())
        raise_argmin_error(error_kind::dimension_mismatch,
                           std::string("lower has length ")
                               + format_number(static_cast<int>(lower.size()))
                               + ", upper has length "
                               + format_number(static_cast<int>(upper.size())));
    for(int index = 0; index < static_cast<int>(lower.size()); ++index)
    {
        if(!(lower[index] > upper[index]))
            continue;
        raise_argmin_error(error_kind::invalid_bounds,
                           std::string("lower bound ") + format_number(lower[index])
                               + " exceeds upper bound " + format_number(upper[index])
                               + " at index " + format_number(index));
    }
}

inline void check_positive_dimension(int value, std::string_view name)
{
    if(value > 0)
        return;
    raise_argmin_error(error_kind::dimension_mismatch,
                       std::string(name) + " must be positive, got " + format_number(value));
}

inline void check_finite(double value, std::string_view name)
{
    if(std::isfinite(value))
        return;
    raise_argmin_error(error_kind::non_finite_input, std::string(name) + " is not finite");
}

inline void check_non_negative(double value, std::string_view name)
{
    check_finite(value, name);
    if(value >= 0.0)
        return;
    raise_argmin_error(error_kind::invalid_problem,
                       std::string(name) + " must be non-negative, got " + format_number(value));
}

}

#endif
