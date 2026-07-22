#include "bindings/detail/errors.h"
#include "bindings/registrations.h"
#include "bindings/detail/validate.h"
#include "bindings/detail/format_number.h"

#include "argmin/qp/qp_types.h"
#include "argmin/qp/dense_admm_qp.h"
#include "argmin/qp/sparse_admm_qp.h"
#include "argmin/qp/dense_active_set_qp.h"

#include "argmin/options/dense_qp_options.h"
#include "argmin/options/sparse_qp_options.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>

#include <Eigen/SparseCore>

#include <string>
#include <utility>
#include <string_view>

namespace nb = nanobind;

namespace argmin::python
{

namespace
{

using dense_matrix = matrix<double>;
using dense_vector = vector<double>;
using sparse_matrix = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using qp_result_type = qp_result<double>;

template <typename Options>
std::string describe_options(std::string_view name, const Options& opts)
{
    return std::string(name) + "(rho=" + format_number(opts.rho) + ", sigma="
           + format_number(opts.sigma) + ", alpha=" + format_number(opts.alpha) + ", eps_abs="
           + format_number(opts.eps_abs) + ", eps_rel=" + format_number(opts.eps_rel)
           + ", eps_prim_inf=" + format_number(opts.eps_prim_inf) + ", eps_dual_inf="
           + format_number(opts.eps_dual_inf) + ", max_iterations="
           + format_number(opts.max_iterations) + ", check_termination="
           + format_number(opts.check_termination) + ", adaptive_rho="
           + (opts.adaptive_rho ? "True" : "False") + ", adaptive_rho_interval="
           + format_number(opts.adaptive_rho_interval) + ", adaptive_rho_tolerance="
           + format_number(opts.adaptive_rho_tolerance) + ", scaling="
           + format_number(opts.scaling) + ", polish=" + (opts.polish ? "True" : "False")
           + ", polish_refine_iter=" + format_number(opts.polish_refine_iter) + ", delta="
           + format_number(opts.delta) + ", warm_start=" + (opts.warm_start ? "True" : "False")
           + ")";
}

// The class is bound over the library aggregate itself rather than over a
// mirror struct: an exposed default is then the library's own brace
// initializer, and there is no second copy of a tolerance to drift out of step
// with the solver that reads it.
template <typename Options>
void bind_options(nb::module_& m, const char* name)
{
    nb::class_<Options>(m, name)
        .def(nb::init<>())
        .def_rw("rho", &Options::rho)
        .def_rw("sigma", &Options::sigma)
        .def_rw("alpha", &Options::alpha)
        .def_rw("eps_abs", &Options::eps_abs)
        .def_rw("eps_rel", &Options::eps_rel)
        .def_rw("eps_prim_inf", &Options::eps_prim_inf)
        .def_rw("eps_dual_inf", &Options::eps_dual_inf)
        .def_rw("max_iterations", &Options::max_iterations)
        .def_rw("check_termination", &Options::check_termination)
        .def_rw("adaptive_rho", &Options::adaptive_rho)
        .def_rw("adaptive_rho_interval", &Options::adaptive_rho_interval)
        .def_rw("adaptive_rho_tolerance", &Options::adaptive_rho_tolerance)
        .def_rw("scaling", &Options::scaling)
        .def_rw("polish", &Options::polish)
        .def_rw("polish_refine_iter", &Options::polish_refine_iter)
        .def_rw("delta", &Options::delta)
        .def_rw("warm_start", &Options::warm_start)
        .def("__repr__",
             [name](const Options& opts) { return describe_options(name, opts); });
}

void check_problem_data(const dense_matrix& P, const dense_vector& q, const dense_matrix& A,
                        const dense_vector& l, const dense_vector& u)
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

void check_update_data(const dense_vector& q, const dense_vector& l, const dense_vector& u, int n,
                       int m)
{
    check_vector_length(q, n, "q");
    check_vector_length(l, m, "l");
    check_vector_length(u, m, "u");
    check_all_finite(q, "q");
    check_no_nan(l, "l");
    check_no_nan(u, "u");
    check_ordered_bounds(l, u);
}

// The library reports an unposed resolve through the same generic value it uses
// for several other precondition failures, so the distinction is not
// recoverable from the return value and has to be tracked here.
class posed_state
{
public:
    void pose(int n, int m)
    {
        n_ = n;
        m_ = m;
        posed_ = true;
    }

    void require_posed() const
    {
        if(posed_)
            return;
        raise_argmin_error(error_kind::invalid_state,
                           "resolve requires a preceding solve to pose the problem");
    }

    [[nodiscard]] int posed_rows() const { return n_; }
    [[nodiscard]] int posed_constraints() const { return m_; }

private:
    int n_{0};
    int m_{0};
    bool posed_{false};
};

template <typename Solver>
class dense_qp_facade
{
public:
    dense_qp_facade(int n, int m) : n_cap_(n), m_cap_(m), solver_(n, m) {}

    qp_result_type solve(const dense_matrix& P, const dense_vector& q, const dense_matrix& A,
                         const dense_vector& l, const dense_vector& u,
                         const dense_qp_options& opts)
    {
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
                           const dense_qp_options& opts)
    {
        state_.require_posed();
        check_update_data(q, l, u, state_.posed_rows(), state_.posed_constraints());

        auto outcome = run_resolve(q, l, u, opts);
        if(!outcome)
            raise_qp_error(outcome.error(), "resolve");
        return std::move(*outcome);
    }

    void warm_start(const dense_vector& x, const dense_vector& y)
    {
        check_vector_length(x, n_cap_, "x");
        check_all_finite(x, "x");
        if(y.size() != 0)
        {
            check_vector_length(y, m_cap_, "y");
            check_all_finite(y, "y");
        }
        solver_.warm_start(x, y);
    }

    void reset() { solver_.reset(); }

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
                                                 const dense_qp_options& opts)
    {
        nb::gil_scoped_release released;
        return solver_.solve(P, q, A, l, u, opts);
    }

    expected<qp_result_type, qp_error> run_resolve(const dense_vector& q, const dense_vector& l,
                                                   const dense_vector& u,
                                                   const dense_qp_options& opts)
    {
        nb::gil_scoped_release released;
        return solver_.resolve(q, l, u, opts);
    }

    int n_cap_;
    int m_cap_;
    Solver solver_;
    posed_state state_;
};

sparse_matrix as_compressed_column(nb::handle value, std::string_view name)
{
    const bool is_compressed_column =
        nb::hasattr(value, "format") && nb::isinstance<nb::str>(nb::getattr(value, "format"))
        && nb::cast<std::string>(nb::getattr(value, "format")) == "csc";
    if(!is_compressed_column)
        raise_argmin_error(error_kind::invalid_array,
                           std::string(name)
                               + " must be a compressed sparse column matrix; convert it with "
                                 ".tocsc()");
    return nb::cast<sparse_matrix>(value);
}

class sparse_qp_facade
{
public:
    qp_result_type solve(nb::handle P, const dense_vector& q, nb::handle A, const dense_vector& l,
                         const dense_vector& u, const sparse_qp_options& opts)
    {
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
                           const sparse_qp_options& opts)
    {
        state_.require_posed();
        check_update_data(q, l, u, state_.posed_rows(), state_.posed_constraints());

        auto outcome = run_resolve(q, l, u, opts);
        if(!outcome)
            raise_qp_error(outcome.error(), "resolve");
        return std::move(*outcome);
    }

    void warm_start(const dense_vector& x, const dense_vector& y)
    {
        check_all_finite(x, "x");
        check_all_finite(y, "y");
        solver_.warm_start(x, y);
    }

    void reset() { solver_.reset(); }

private:
    expected<qp_result_type, qp_error> run_solve(const sparse_matrix& P, const dense_vector& q,
                                                 const sparse_matrix& A, const dense_vector& l,
                                                 const dense_vector& u,
                                                 const sparse_qp_options& opts)
    {
        nb::gil_scoped_release released;
        return solver_.solve(P, q, A, l, u, opts);
    }

    expected<qp_result_type, qp_error> run_resolve(const dense_vector& q, const dense_vector& l,
                                                   const dense_vector& u,
                                                   const sparse_qp_options& opts)
    {
        nb::gil_scoped_release released;
        return solver_.resolve(q, l, u, opts);
    }

    sparse_admm_qp_solver<double> solver_;
    posed_state state_;
};

template <typename Solver>
void bind_dense_solver(nb::module_& m, const char* name)
{
    using facade = dense_qp_facade<Solver>;

    nb::class_<facade>(m, name)
        .def(
            "__init__",
            [](facade* self, int n, int m_rows)
            {
                check_positive_dimension(n, "n");
                check_positive_dimension(m_rows, "m");
                new(self) facade(n, m_rows);
            },
            nb::arg("n"), nb::arg("m"))
        .def("solve", &facade::solve, nb::arg("P"), nb::arg("q"), nb::arg("A"), nb::arg("l"),
             nb::arg("u"), nb::arg("options") = dense_qp_options())
        .def("resolve", &facade::resolve, nb::arg("q"), nb::arg("l"), nb::arg("u"),
             nb::arg("options") = dense_qp_options())
        .def("warm_start", &facade::warm_start, nb::arg("x"), nb::arg("y") = dense_vector())
        .def("reset", &facade::reset)
        .def_prop_ro("n", &facade::decision_capacity)
        .def_prop_ro("m", &facade::constraint_capacity)
        .def("__repr__",
             [name](const facade& solver)
             {
                 return std::string(name) + "(n=" + format_number(solver.decision_capacity())
                        + ", m=" + format_number(solver.constraint_capacity()) + ")";
             });
}

void bind_sparse_solver(nb::module_& m)
{
    nb::class_<sparse_qp_facade>(m, "SparseAdmmQpSolver")
        .def(nb::init<>())
        .def("solve", &sparse_qp_facade::solve, nb::arg("P"), nb::arg("q"), nb::arg("A"),
             nb::arg("l"), nb::arg("u"), nb::arg("options") = sparse_qp_options())
        .def("resolve", &sparse_qp_facade::resolve, nb::arg("q"), nb::arg("l"), nb::arg("u"),
             nb::arg("options") = sparse_qp_options())
        .def("warm_start", &sparse_qp_facade::warm_start, nb::arg("x"),
             nb::arg("y") = dense_vector())
        .def("reset", &sparse_qp_facade::reset)
        .def("__repr__", [](const sparse_qp_facade&) { return std::string("SparseAdmmQpSolver()"); });
}

}

void register_qp(nb::module_& m)
{
    bind_options<dense_qp_options>(m, "DenseQpOptions");
    bind_options<sparse_qp_options>(m, "SparseQpOptions");

    bind_dense_solver<dense_admm_qp_solver<double>>(m, "DenseAdmmQpSolver");
    bind_dense_solver<dense_active_set_qp_solver<double>>(m, "DenseActiveSetQpSolver");
    bind_sparse_solver(m);
}

}
