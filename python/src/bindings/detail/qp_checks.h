#ifndef HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_QP_CHECKS_H
#define HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_QP_CHECKS_H

#include "bindings/detail/errors.h"
#include "bindings/detail/validate.h"

#include "argmin/types.h"
#include "argmin/qp/qp_types.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/eigen/sparse.h>

#include <Eigen/SparseCore>

#include <string>
#include <string_view>

namespace argmin::python
{

using dense_matrix = matrix<double>;
using dense_vector = vector<double>;
using sparse_matrix = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using qp_result_type = qp_result<double>;

inline void check_problem_data(const dense_matrix& P, const dense_vector& q,
                               const dense_matrix& A, const dense_vector& l,
                               const dense_vector& u)
{
    const int n = static_cast<int>(P.rows());
    const int m = static_cast<int>(A.rows());
    check_square(P, "P");
    check_matrix_columns(static_cast<int>(A.cols()), n, "A");
    check_vector_length(q, n, "q");
    check_vector_length(l, m, "l");
    check_vector_length(u, m, "u");
    check_all_finite(P, "P");
    check_all_finite(q, "q");
    check_all_finite(A, "A");
    check_no_nan(l, "l");
    check_no_nan(u, "u");
    check_ordered_bounds(l, u);
}

inline void check_update_data(const dense_vector& q, const dense_vector& l,
                              const dense_vector& u, int n, int m)
{
    check_vector_length(q, n, "q");
    check_vector_length(l, m, "l");
    check_vector_length(u, m, "u");
    check_all_finite(q, "q");
    check_no_nan(l, "l");
    check_no_nan(u, "u");
    check_ordered_bounds(l, u);
}

inline sparse_matrix as_compressed_column(nanobind::handle value, std::string_view name)
{
    const bool is_compressed_column =
        nanobind::hasattr(value, "format")
        && nanobind::isinstance<nanobind::str>(nanobind::getattr(value, "format"))
        && nanobind::cast<std::string>(nanobind::getattr(value, "format")) == "csc";
    if(!is_compressed_column)
        raise_argmin_error(error_kind::invalid_array,
                           std::string(name)
                               + " must be a compressed sparse column matrix; convert it with "
                                 ".tocsc()");
    return nanobind::cast<sparse_matrix>(value);
}

}

#endif
