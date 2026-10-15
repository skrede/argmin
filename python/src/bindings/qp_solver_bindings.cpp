#include "bindings/registrations.h"
#include "bindings/detail/validate.h"
#include "bindings/detail/format_number.h"
#include "bindings/detail/dense_qp_facade.h"
#include "bindings/detail/sparse_qp_facade.h"

#include "argmin/qp/dense_admm_qp.h"
#include "argmin/qp/dense_active_set_qp.h"
#include "argmin/options/dense_qp_options.h"
#include "argmin/options/sparse_qp_options.h"
#include "argmin/options/dense_active_set_qp_options.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/eigen/dense.h>

#include <new>
#include <string>

namespace nb = nanobind;

namespace argmin::python
{

namespace
{

template <typename Solver, typename Options>
void bind_dense_solver(nb::module_& m, const char* name, const char* doc)
{
    using facade = dense_qp_facade<Solver, Options>;

    nb::class_<facade>(m, name, doc)
        .def(
            "__init__",
            [](facade* self, int n, int m_rows)
            {
                check_positive_dimension(n, "n");
                check_non_negative_dimension(m_rows, "m");
                new(self) facade(n, m_rows);
            },
            nb::arg("n"), nb::arg("m"))
        .def("solve", &facade::solve, nb::arg("P"), nb::arg("q"), nb::arg("A"), nb::arg("l"),
             nb::arg("u"), nb::arg("options") = Options())
        .def("resolve", &facade::resolve, nb::arg("q"), nb::arg("l"), nb::arg("u"),
             nb::arg("options") = Options())
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
    nb::class_<sparse_qp_facade>(
        m, "SparseAdmmQpSolver",
        "Operator-splitting solver for a sparse quadratic program. Matrices arrive in "
        "compressed sparse column form; any other layout is rejected rather than "
        "silently converted. The problem is posed by solve() and may then be updated "
        "through resolve() without refactorizing.")
        .def(nb::init<>())
        .def("solve", &sparse_qp_facade::solve, nb::arg("P"), nb::arg("q"), nb::arg("A"),
             nb::arg("l"), nb::arg("u"), nb::arg("options") = sparse_qp_options<>())
        .def("resolve", &sparse_qp_facade::resolve, nb::arg("q"), nb::arg("l"), nb::arg("u"),
             nb::arg("options") = sparse_qp_options<>())
        .def("warm_start", &sparse_qp_facade::warm_start, nb::arg("x"),
             nb::arg("y") = dense_vector())
        .def("reset", &sparse_qp_facade::reset)
        .def("__repr__", [](const sparse_qp_facade&) { return std::string("SparseAdmmQpSolver()"); });
}

}

void register_qp_solvers(nb::module_& m)
{
    bind_dense_solver<dense_admm_qp_solver<double>, dense_qp_options<>>(
        m, "DenseAdmmQpSolver",
        "Operator-splitting solver for a dense quadratic program, constructed for a fixed "
        "decision length and a maximum constraint-row count. A constraint count of zero is "
        "the unconstrained program.");
    bind_dense_solver<dense_active_set_qp_solver<double>, dense_active_set_qp_options<>>(
        m, "DenseActiveSetQpSolver",
        "Active-set solver for a dense quadratic program. It requires a feasible starting "
        "point: seed one with warm_start() before solving a problem whose equality rows the "
        "origin does not satisfy.");
    bind_sparse_solver(m);
}

}
