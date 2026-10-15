#ifndef HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_CALLBACK_BOUNDARY_H
#define HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_CALLBACK_BOUNDARY_H

#include "bindings/detail/format_number.h"
#include "bindings/detail/callback_latch.h"
#include "bindings/detail/callback_checks.h"

#include "argmin/types.h"

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>

#include <cmath>
#include <string>
#include <string_view>

namespace argmin::python
{

class callback_boundary : public callback_latch
{
public:
    using callback_latch::callback_latch;

protected:
    // The rejection values a short-circuited callback hands back once the latch
    // is set. The loop consults the abort flag only at the top of an iteration,
    // so the step already in flight runs to completion on these: an infinite
    // objective is a rejection every line search and every trust-region
    // acceptance test terminates on immediately, whereas a fabricated zero
    // could read as a stationary point and a not-a-number could stall a line
    // search inside that final step. The step's result is discarded -- the
    // latched exception is re-raised before any result is built -- so the only
    // requirement on these values is that they cannot make the in-flight step
    // loop or diverge.
    double invoke_value(const vector<double>& x) const
    {
        if(latched_)
            return infinity();
        try
        {
            nanobind::gil_scoped_acquire acquired;
            nanobind::object produced = calls_.objective(copy_out(x));
            double produced_value = 0.0;
            if(!nanobind::try_cast<double>(produced, produced_value))
                reject("the objective returned a value that is not a real number");
            if(std::isnan(produced_value))
                reject("the objective returned a value that is not a number");
            return produced_value;
        }
        catch(...)
        {
            latch_and_abort();
            return infinity();
        }
    }

    void invoke_gradient(const vector<double>& x, vector<double>& g) const
    {
        if(latched_)
        {
            g.setZero(n_);
            return;
        }
        try
        {
            nanobind::gil_scoped_acquire acquired;
            const vector<double> produced =
                as_vector(calls_.gradient(copy_out(x)), "the gradient");
            require_length(static_cast<int>(produced.size()), n_, "the gradient");
            require_finite(produced, "the gradient");
            g = produced;
        }
        catch(...)
        {
            latch_and_abort();
            g.setZero(n_);
        }
    }

    void invoke_constraints(const vector<double>& x, vector<double>& c) const
    {
        const int rows = calls_.num_equality + calls_.num_inequality;
        if(latched_)
        {
            c.setZero(rows);
            return;
        }
        try
        {
            nanobind::gil_scoped_acquire acquired;
            const vector<double> produced =
                as_vector(calls_.constraints(copy_out(x)), "the constraints");
            require_length(static_cast<int>(produced.size()), rows, "the constraints");
            require_finite(produced, "the constraints");
            c = produced;
        }
        catch(...)
        {
            latch_and_abort();
            c.setZero(rows);
        }
    }

    void invoke_jacobian(const vector<double>& x, matrix<double>& J) const
    {
        const int rows = calls_.num_equality + calls_.num_inequality;
        if(latched_)
        {
            J.setZero(rows, n_);
            return;
        }
        try
        {
            nanobind::gil_scoped_acquire acquired;
            nanobind::object produced = calls_.constraint_jacobian(copy_out(x));
            matrix<double> values;
            if(!nanobind::try_cast<matrix<double>>(produced, values))
                reject("the constraint Jacobian did not convert to a real matrix");
            if(static_cast<int>(values.rows()) != rows
               || static_cast<int>(values.cols()) != n_)
            {
                reject("the constraint Jacobian has shape "
                       + format_shape(static_cast<int>(values.rows()),
                                      static_cast<int>(values.cols()))
                       + ", expected " + format_shape(rows, n_));
            }
            require_finite(values, "the constraint Jacobian");
            J = values;
        }
        catch(...)
        {
            latch_and_abort();
            J.setZero(rows, n_);
        }
    }

    vector<double> invoke_bound(const nanobind::object& source, bool source_is_callable,
                                const vector<double>& cache,
                                std::string_view name) const
    {
        if(!source_is_callable || latched_)
            return cache;
        try
        {
            nanobind::gil_scoped_acquire acquired;
            const vector<double> produced = as_vector(source(), name);
            require_length(static_cast<int>(produced.size()), n_, name);
            require_no_nan(produced, name);
            return produced;
        }
        catch(...)
        {
            latch_and_abort();
            return cache;
        }
    }
};

}

#endif
