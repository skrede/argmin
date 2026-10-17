#ifndef HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_SPARSE_QP_FACADE_H
#define HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_SPARSE_QP_FACADE_H

#include "bindings/detail/errors.h"
#include "bindings/detail/validate.h"
#include "bindings/detail/qp_checks.h"
#include "bindings/detail/qp_posed_state.h"

#include "argmin/expected.h"
#include "argmin/qp/qp_types.h"
#include "argmin/qp/sparse_admm_qp.h"
#include "argmin/options/sparse_qp_options.h"

#include <nanobind/nanobind.h>

#include <atomic>
#include <utility>

namespace argmin::python
{

class sparse_qp_facade
{
    class busy_guard
    {
    public:
        explicit busy_guard(sparse_qp_facade& owner) : owner_(owner)
        {
            bool idle = false;
            if(!owner_.busy_.compare_exchange_strong(idle, true))
                raise_argmin_error(error_kind::invalid_state,
                                   "this solver is already running on another thread");
        }

        ~busy_guard() { owner_.busy_.store(false); }

        busy_guard(const busy_guard&) = delete;
        busy_guard& operator=(const busy_guard&) = delete;

    private:
        sparse_qp_facade& owner_;
    };

public:
    sparse_qp_facade() : busy_(false) {}

    qp_result_type solve(nanobind::handle P, const dense_vector& q, nanobind::handle A,
                         const dense_vector& l, const dense_vector& u,
                         const sparse_qp_options<>& opts)
    {
        const busy_guard guard(*this);
        const sparse_matrix P_csc = as_compressed_column(P, "P");
        const sparse_matrix A_csc = as_compressed_column(A, "A");

        const int n = static_cast<int>(P_csc.rows());
        const int m = static_cast<int>(A_csc.rows());
        check_square_shape(n, static_cast<int>(P_csc.cols()), "P");
        check_matrix_columns(static_cast<int>(A_csc.cols()), n, "A");
        check_vector_length(q, n, "q");
        check_vector_length(l, m, "l");
        check_vector_length(u, m, "u");
        check_all_finite(P_csc, "P");
        check_all_finite(q, "q");
        check_all_finite(A_csc, "A");
        check_no_nan(l, "l");
        check_no_nan(u, "u");
        check_ordered_bounds(l, u);

        auto outcome = run_solve(P_csc, q, A_csc, l, u, opts);
        if(!outcome)
            raise_qp_error(outcome.error(), "solve");
        state_.pose(n, m);
        return std::move(*outcome);
    }

    qp_result_type resolve(const dense_vector& q, const dense_vector& l, const dense_vector& u,
                           const sparse_qp_options<>& opts)
    {
        const busy_guard guard(*this);
        state_.require_posed("resolve");
        check_update_data(q, l, u, state_.posed_rows(), state_.posed_constraints());

        auto outcome = run_resolve(q, l, u, opts);
        if(!outcome)
            raise_qp_error(outcome.error(), "resolve");
        return std::move(*outcome);
    }

    void warm_start(const dense_vector& x, const dense_vector& y)
    {
        const busy_guard guard(*this);
        state_.require_posed("warm_start");
        check_vector_length(x, state_.posed_rows(), "x");
        check_vector_length(y, state_.posed_constraints(), "y");
        check_all_finite(x, "x");
        check_all_finite(y, "y");
        solver_.warm_start(x, y);
    }

    void reset()
    {
        const busy_guard guard(*this);
        solver_.reset();
    }

private:
    expected<qp_result_type, qp_error> run_solve(const sparse_matrix& P, const dense_vector& q,
                                                 const sparse_matrix& A, const dense_vector& l,
                                                 const dense_vector& u,
                                                 const sparse_qp_options<>& opts)
    {
        nanobind::gil_scoped_release released;
        return solver_.solve(P, q, A, l, u, opts);
    }

    expected<qp_result_type, qp_error> run_resolve(const dense_vector& q, const dense_vector& l,
                                                   const dense_vector& u,
                                                   const sparse_qp_options<>& opts)
    {
        nanobind::gil_scoped_release released;
        return solver_.resolve(q, l, u, opts);
    }

    sparse_admm_qp_solver<double> solver_;
    posed_state state_;
    std::atomic<bool> busy_;
};

}

#endif
