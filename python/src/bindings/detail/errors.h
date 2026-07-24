#ifndef HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_ERRORS_H
#define HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_ERRORS_H

// The single failure channel of the binding tree. The library itself is
// exception-free and stays that way: everything here throws, nothing under lib/
// does. Solve outcomes are not failures -- reaching an iteration cap or
// certifying infeasibility is an answer and travels as an enumeration attribute
// on the returned result, never through this type.

#include "argmin/qp/qp_types.h"

#include <string>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace argmin::python
{

// The first six mirror argmin::qp_error value for value; a test asserts that
// correspondence by name, so a value added to the library and forgotten here
// fails the suite instead of degrading to a generic message.
enum class error_kind : std::uint8_t
{
    dimension_mismatch,
    invalid_bounds,
    non_finite_input,
    capacity_exceeded,
    infeasible_start,
    invalid_problem,
    invalid_array,
    invalid_callback,
    invalid_state
};

class argmin_error : public std::runtime_error
{
public:
    argmin_error(error_kind kind, const std::string& message)
        : std::runtime_error(message), kind_(kind)
    {
    }

    [[nodiscard]] error_kind kind() const noexcept
    {
        return kind_;
    }

private:
    error_kind kind_;
};

[[noreturn]] inline void raise_argmin_error(error_kind kind, const std::string& message)
{
    throw argmin_error(kind, message);
}

[[nodiscard]] inline error_kind kind_of(qp_error error) noexcept
{
    switch(error)
    {
        case qp_error::dimension_mismatch: return error_kind::dimension_mismatch;
        case qp_error::invalid_bounds: return error_kind::invalid_bounds;
        case qp_error::non_finite_input: return error_kind::non_finite_input;
        case qp_error::capacity_exceeded: return error_kind::capacity_exceeded;
        case qp_error::infeasible_start: return error_kind::infeasible_start;
        case qp_error::invalid_problem: return error_kind::invalid_problem;
    }
    return error_kind::invalid_problem;
}

[[nodiscard]] inline std::string describe(qp_error error)
{
    switch(error)
    {
        case qp_error::dimension_mismatch:
            return "the problem data do not agree on a common dimension";
        case qp_error::invalid_bounds:
            return "a lower bound exceeds its upper bound";
        case qp_error::non_finite_input:
            return "the problem data contain a non-finite entry";
        case qp_error::capacity_exceeded:
            return "the problem exceeds the capacity this solver was posed for";
        case qp_error::infeasible_start:
            return "the supplied starting point is not usable";
        case qp_error::invalid_problem:
            return "the problem as posed cannot be solved by this method";
    }
    return "the problem as posed cannot be solved by this method";
}

[[noreturn]] inline void raise_qp_error(qp_error error, std::string_view context)
{
    std::string message(context);
    message += ": ";
    message += describe(error);
    raise_argmin_error(kind_of(error), message);
}

}

#endif
