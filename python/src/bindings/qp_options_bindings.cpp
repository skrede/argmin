#include "bindings/registrations.h"
#include "bindings/detail/format_number.h"

#include "argmin/options/dense_qp_options.h"
#include "argmin/options/sparse_qp_options.h"
#include "argmin/options/dense_active_set_qp_options.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <string>
#include <string_view>

namespace nb = nanobind;

namespace argmin::python
{

namespace
{

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
           + ", validate_convexity=" + (opts.validate_convexity ? "True" : "False") + ")";
}

// The class is bound over the library aggregate itself rather than over a
// mirror struct: an exposed default is then the library's own brace
// initializer, and there is no second copy of a tolerance to drift out of step
// with the solver that reads it.
template <typename Options>
void bind_options(nb::module_& m, const char* name, const char* doc)
{
    nb::class_<Options>(m, name, doc)
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
        .def_rw("validate_convexity", &Options::validate_convexity)
        .def("__repr__",
             [name](const Options& opts) { return describe_options(name, opts); });
}

void bind_active_set_options(nb::module_& m)
{
    nb::class_<dense_active_set_qp_options<>>(
        m, "DenseActiveSetQpOptions",
        "Configuration for the dense active-set quadratic-program solver. It carries only the "
        "settings that solver honors; the exposed defaults are the library's own.")
        .def(nb::init<>())
        .def_rw("max_iterations", &dense_active_set_qp_options<>::max_iterations)
        .def_rw("tolerance", &dense_active_set_qp_options<>::tolerance)
        .def_rw("equality_tolerance", &dense_active_set_qp_options<>::equality_tolerance)
        .def("__repr__",
             [](const dense_active_set_qp_options<>& opts)
             {
                 return "DenseActiveSetQpOptions(max_iterations=" + format_number(opts.max_iterations)
                        + ", tolerance=" + format_number(opts.tolerance)
                        + ", equality_tolerance=" + format_number(opts.equality_tolerance) + ")";
             });
}

}

void register_qp_options(nb::module_& m)
{
    bind_options<dense_qp_options<>>(
        m, "DenseQpOptions",
        "Configuration for the dense operator-splitting quadratic-program solver. The exposed "
        "defaults are the library's own; every field is readable and writable.");
    bind_options<sparse_qp_options<>>(
        m, "SparseQpOptions",
        "Configuration for the sparse quadratic-program solver, with the same fields and "
        "the same library defaults as the dense form.");
    bind_active_set_options(m);
}

}
