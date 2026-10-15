#include "bindings/detail/errors.h"
#include "bindings/registrations.h"

#include <nanobind/nanobind.h>

#include <exception>

namespace nb = nanobind;

namespace argmin::python
{

namespace
{

// Held for the process lifetime rather than in a destructible object: a static
// handle released during interpreter teardown would drop a reference after the
// runtime that owns it is gone.
PyObject* python_error_type = nullptr;

void translate_argmin_error(const std::exception_ptr& raised, void*)
{
    try
    {
        std::rethrow_exception(raised);
    }
    catch(const argmin_error& error)
    {
        nb::object instance = nb::borrow<nb::object>(python_error_type)(error.what());
        instance.attr("kind") = error.kind();
        PyErr_SetObject(python_error_type, instance.ptr());
    }
}

}

void register_error_channel(nb::module_& m)
{
    nb::enum_<error_kind>(m, "ErrorKind",
                     "The machine-readable category carried by ArgminError.kind. The first eight "
                     "mirror the library's own error channel value for value, including an option "
                     "outside its domain and a failed factorization; the last three exist only at "
                     "the interpreter boundary.")
        .value("dimension_mismatch", error_kind::dimension_mismatch)
        .value("invalid_bounds", error_kind::invalid_bounds)
        .value("non_finite_input", error_kind::non_finite_input)
        .value("capacity_exceeded", error_kind::capacity_exceeded)
        .value("infeasible_start", error_kind::infeasible_start)
        .value("invalid_problem", error_kind::invalid_problem)
        .value("invalid_options", error_kind::invalid_options)
        .value("factorization_failed", error_kind::factorization_failed)
        .value("invalid_array", error_kind::invalid_array)
        .value("invalid_callback", error_kind::invalid_callback)
        .value("invalid_state", error_kind::invalid_state);

    if(python_error_type == nullptr)
        python_error_type = PyErr_NewExceptionWithDoc(
            "argmin.ArgminError",
            "Raised when an argument or a precondition is violated. Carries a .kind "
            "attribute naming the category. Solve outcomes never arrive this way.",
            PyExc_RuntimeError, nullptr);
    m.attr("ArgminError") = nb::borrow<nb::object>(python_error_type);

    nb::register_exception_translator(translate_argmin_error, nullptr);
}

}
