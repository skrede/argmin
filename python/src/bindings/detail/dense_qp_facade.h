#ifndef HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_DENSE_QP_FACADE_H
#define HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_DENSE_QP_FACADE_H

#include "bindings/detail/errors.h"
#include "bindings/detail/validate.h"
#include "bindings/detail/qp_checks.h"
#include "bindings/detail/format_number.h"
#include "bindings/detail/qp_posed_state.h"

#include "argmin/expected.h"
#include "argmin/qp/qp_types.h"

#include <nanobind/nanobind.h>

#include <atomic>
#include <utility>

namespace argmin::python
{

template <typename Solver, typename Options>
class dense_qp_facade
{
    // Every public method takes the guard, the mutators included: a solve runs
    // with the interpreter lock released, so an unguarded warm_start or reset
    // from another thread would write into state that solve is reading.
    class busy_guard
    {
    public:
        explicit busy_guard(dense_qp_facade& owner) : owner_(owner)
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
        dense_qp_facade& owner_;
    };

public:
    dense_qp_facade(int n, int m) : n_cap_(n), m_cap_(m), solver_(n, m), busy_(false) {}

    qp_result_type solve(const dense_matrix& P, const dense_vector& q, const dense_matrix& A,
                         const dense_vector& l, const dense_vector& u,
                         const Options& opts)
    {
        const busy_guard guard(*this);
        check_problem_data(P, q, A, l, u);
        const int n = static_cast<int>(P.rows());
        const int m = static_cast<int>(A.rows());
        check_capacity(n, m);

        auto outcome = run_solve(P, q, A, l, u, opts);
        if(!outcome)
            raise_qp_error(outcome.error(), "solve");
        state_.pose(n, m);
        // The result carries its own vectors, and the property that hands them
        // to the caller copies again. Neither may become a view: the solver
        // reuses its buffers across calls, so a view would change under a
        // caller that merely held on to a previous answer.
        return std::move(*outcome);
    }

    qp_result_type resolve(const dense_vector& q, const dense_vector& l, const dense_vector& u,
                           const Options& opts)
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
        check_vector_length(x, n_cap_, "x");
        check_all_finite(x, "x");
        check_all_finite(y, "y");
        if(const auto rejected = solver_.warm_start(x, y))
            raise_qp_error(*rejected, "warm_start");
    }

    void reset()
    {
        const busy_guard guard(*this);
        solver_.reset();
    }

    [[nodiscard]] int decision_capacity() const { return n_cap_; }
    [[nodiscard]] int constraint_capacity() const { return m_cap_; }

private:
    void check_capacity(int n, int m) const
    {
        if(n != n_cap_)
            raise_argmin_error(error_kind::capacity_exceeded,
                               "P has " + format_number(n)
                                   + " rows, but this solver was constructed for exactly "
                                   + format_number(n_cap_));
        if(m > m_cap_)
            raise_argmin_error(error_kind::capacity_exceeded,
                               "A has " + format_number(m)
                                   + " rows, exceeding the constraint capacity "
                                   + format_number(m_cap_));
    }

    // A declaration-level call guard would release before the function body
    // runs, which would put the validation above and the raise below outside
    // the lock; the release is therefore scoped to the solver call alone.
    expected<qp_result_type, qp_error> run_solve(const dense_matrix& P, const dense_vector& q,
                                                 const dense_matrix& A, const dense_vector& l,
                                                 const dense_vector& u,
                                                 const Options& opts)
    {
        nanobind::gil_scoped_release released;
        return solver_.solve(P, q, A, l, u, opts);
    }

    expected<qp_result_type, qp_error> run_resolve(const dense_vector& q, const dense_vector& l,
                                                   const dense_vector& u,
                                                   const Options& opts)
    {
        nanobind::gil_scoped_release released;
        return solver_.resolve(q, l, u, opts);
    }

    int n_cap_;
    int m_cap_;
    Solver solver_;
    posed_state state_;
    std::atomic<bool> busy_;
};

}

#endif
