#ifndef HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_PROBLEM_ADAPTER_H
#define HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_PROBLEM_ADAPTER_H

#include "bindings/detail/errors.h"
#include "bindings/detail/validate.h"
#include "bindings/detail/format_number.h"

#include "argmin/types.h"
#include "argmin/derivative/finite_difference.h"

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>

#include <Eigen/Core>

#include <array>
#include <cmath>
#include <string>
#include <limits>
#include <utility>
#include <exception>
#include <string_view>

namespace argmin::python
{

struct callback_set
{
    nanobind::object objective{};
    nanobind::object gradient{};
    nanobind::object lower_bounds{};
    nanobind::object upper_bounds{};
    nanobind::object constraints{};
    nanobind::object constraint_jacobian{};
    int num_equality{0};
    int num_inequality{0};
};

class callback_boundary
{
public:
    using abort_hook = void (*)(void*) noexcept;

    callback_boundary(int n, callback_set callbacks)
        : n_(n), calls_(std::move(callbacks))
    {
        gradient_is_callable_ = is_callable(calls_.gradient);
        lower_is_callable_ = is_callable(calls_.lower_bounds);
        upper_is_callable_ = is_callable(calls_.upper_bounds);
        lower_cache_ = cache_bound(calls_.lower_bounds, lower_is_callable_, -infinity(), "lower");
        upper_cache_ = cache_bound(calls_.upper_bounds, upper_is_callable_, infinity(), "upper");
    }

    callback_boundary(const callback_boundary&) = delete;
    callback_boundary& operator=(const callback_boundary&) = delete;

    void set_abort_hook(abort_hook hook, void* owner) noexcept
    {
        hook_ = hook;
        owner_ = owner;
    }

    [[nodiscard]] int dimension() const noexcept { return n_; }

    [[nodiscard]] bool latched() const noexcept { return static_cast<bool>(latched_); }

    // The garbage collector cannot see inside a C++ object, so a callable that
    // reaches this adapter again -- through a closure, a bound method's self,
    // or merely its defining module's globals -- forms a cycle nothing can
    // break. These two expose the held references to the collector's traverse
    // and clear slots.
    [[nodiscard]] std::array<PyObject*, 6> callable_handles() const noexcept
    {
        return {calls_.objective.ptr(),   calls_.gradient.ptr(),
                calls_.lower_bounds.ptr(), calls_.upper_bounds.ptr(),
                calls_.constraints.ptr(),  calls_.constraint_jacobian.ptr()};
    }

    void release_callables() noexcept
    {
        calls_.objective.reset();
        calls_.gradient.reset();
        calls_.lower_bounds.reset();
        calls_.upper_bounds.reset();
        calls_.constraints.reset();
        calls_.constraint_jacobian.reset();
    }

    void rethrow_if_latched() const
    {
        if(!latched_)
            return;
        std::rethrow_exception(std::exchange(latched_, {}));
    }

protected:
    static constexpr double infinity() noexcept
    {
        return std::numeric_limits<double>::infinity();
    }

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

    [[nodiscard]] bool has_gradient() const noexcept { return gradient_is_callable_; }

    [[nodiscard]] const callback_set& callbacks() const noexcept { return calls_; }
    [[nodiscard]] bool lower_is_callable() const noexcept { return lower_is_callable_; }
    [[nodiscard]] bool upper_is_callable() const noexcept { return upper_is_callable_; }
    [[nodiscard]] const vector<double>& lower_cache() const noexcept { return lower_cache_; }
    [[nodiscard]] const vector<double>& upper_cache() const noexcept { return upper_cache_; }

    int n_;

private:
    static bool is_callable(const nanobind::object& value) noexcept
    {
        return value.is_valid() && !value.is_none() && PyCallable_Check(value.ptr()) != 0;
    }

    static nanobind::object copy_out(const vector<double>& x)
    {
        return nanobind::cast(x, nanobind::rv_policy::copy);
    }

    [[noreturn]] static void reject(const std::string& message)
    {
        raise_argmin_error(error_kind::invalid_callback, message);
    }

    static vector<double> as_vector(const nanobind::object& produced, std::string_view name)
    {
        vector<double> values;
        if(!nanobind::try_cast<vector<double>>(produced, values))
            reject(std::string(name) + " did not convert to a real vector");
        return values;
    }

    static void require_length(int actual, int expected, std::string_view name)
    {
        if(actual == expected)
            return;
        reject(std::string(name) + " has length " + format_number(actual) + ", expected "
               + format_number(expected));
    }

    template <typename Derived>
    static void require_finite(const Eigen::DenseBase<Derived>& values, std::string_view name)
    {
        if(values.allFinite())
            return;
        reject(std::string(name) + " has a non-finite entry");
    }

    static void require_no_nan(const vector<double>& values, std::string_view name)
    {
        if(!values.hasNaN())
            return;
        reject(std::string(name) + " has an entry that is not a number");
    }

    // An absent bound is filled with an infinity rather than a large finite
    // sentinel: the bound-constrained policy fills its own bound vectors with
    // exactly this value when the problem carries none, so an infinity is the
    // spelling that policy already treats as unbounded.
    vector<double> cache_bound(const nanobind::object& source, bool source_is_callable,
                               double unbounded, std::string_view name) const
    {
        if(!source.is_valid() || source.is_none() || source_is_callable)
            return vector<double>::Constant(n_, unbounded);
        vector<double> values;
        if(!nanobind::try_cast<vector<double>>(source, values))
            raise_argmin_error(error_kind::invalid_array,
                               std::string(name)
                                   + " must be an array of real numbers or a callable");
        check_vector_length(values, n_, name);
        check_no_nan(values, name);
        return values;
    }

    void latch_and_abort() const noexcept
    {
        latched_ = std::current_exception();
        if(hook_ != nullptr)
            hook_(owner_);
    }

    callback_set calls_;
    vector<double> lower_cache_;
    vector<double> upper_cache_;
    mutable std::exception_ptr latched_{};
    abort_hook hook_{nullptr};
    void* owner_{nullptr};
    bool gradient_is_callable_{false};
    bool lower_is_callable_{false};
    bool upper_is_callable_{false};
};

// Value only: the gradient is supplied by the library's own central-difference
// helper, so the tier is differentiable without a caller-supplied derivative.
class value_problem : public callback_boundary
{
public:
    static constexpr int problem_dimension = dynamic_dimension;

    using callback_boundary::callback_boundary;

    double value(const vector<double>& x) const { return invoke_value(x); }

    void gradient(const vector<double>& x, vector<double>& g) const
    {
        fd_gradient(*this, x, g);
    }
};

class gradient_problem : public callback_boundary
{
public:
    static constexpr int problem_dimension = dynamic_dimension;

    using callback_boundary::callback_boundary;

    double value(const vector<double>& x) const { return invoke_value(x); }

    void gradient(const vector<double>& x, vector<double>& g) const { invoke_gradient(x, g); }
};

class bounded_problem : public callback_boundary
{
public:
    static constexpr int problem_dimension = dynamic_dimension;

    using callback_boundary::callback_boundary;

    double value(const vector<double>& x) const { return invoke_value(x); }

    void gradient(const vector<double>& x, vector<double>& g) const
    {
        if(has_gradient())
            invoke_gradient(x, g);
        else
            fd_gradient(*this, x, g);
    }

    vector<double> lower_bounds() const
    {
        return invoke_bound(callbacks().lower_bounds, lower_is_callable(), lower_cache(), "lower");
    }

    vector<double> upper_bounds() const
    {
        return invoke_bound(callbacks().upper_bounds, upper_is_callable(), upper_cache(), "upper");
    }
};

class constrained_values_problem : public callback_boundary
{
public:
    static constexpr int problem_dimension = dynamic_dimension;

    using callback_boundary::callback_boundary;

    double value(const vector<double>& x) const { return invoke_value(x); }

    void gradient(const vector<double>& x, vector<double>& g) const
    {
        if(has_gradient())
            invoke_gradient(x, g);
        else
            fd_gradient(*this, x, g);
    }

    void constraints(const vector<double>& x, vector<double>& c) const
    {
        invoke_constraints(x, c);
    }

    int num_equality() const { return callbacks().num_equality; }
    int num_inequality() const { return callbacks().num_inequality; }

    // The derivative-free constrained policy paired with this tier constrains
    // its problem parameter on bound_constrained as well, so the tier carries
    // bounds even though nothing about a constraint-value formulation requires
    // them; a caller who supplies none gets the infinite fill the policy
    // already treats as unbounded.
    vector<double> lower_bounds() const
    {
        return invoke_bound(callbacks().lower_bounds, lower_is_callable(), lower_cache(), "lower");
    }

    vector<double> upper_bounds() const
    {
        return invoke_bound(callbacks().upper_bounds, upper_is_callable(), upper_cache(), "upper");
    }
};

class constrained_problem : public callback_boundary
{
public:
    static constexpr int problem_dimension = dynamic_dimension;

    using callback_boundary::callback_boundary;

    double value(const vector<double>& x) const { return invoke_value(x); }

    void gradient(const vector<double>& x, vector<double>& g) const
    {
        if(has_gradient())
            invoke_gradient(x, g);
        else
            fd_gradient(*this, x, g);
    }

    void constraints(const vector<double>& x, vector<double>& c) const
    {
        invoke_constraints(x, c);
    }

    int num_equality() const { return callbacks().num_equality; }
    int num_inequality() const { return callbacks().num_inequality; }

    void constraint_jacobian(const vector<double>& x, matrix<double>& J) const
    {
        invoke_jacobian(x, J);
    }
};

// Each adapter must satisfy exactly the tier its policy needs and no more.
// Concept satisfaction here is duck-typed, so an adapter that grows a member
// silently starts advertising a capability the caller never supplied -- an
// unconstrained problem claiming constraints, say. The negative half is the
// load-bearing one; the positive half only records intent.
static_assert(argmin::differentiable<value_problem>);
static_assert(!argmin::bound_constrained<value_problem>);

static_assert(argmin::differentiable<gradient_problem>);
static_assert(!argmin::bound_constrained<gradient_problem>);
static_assert(!argmin::constrained_values<gradient_problem>);

static_assert(argmin::bound_constrained<bounded_problem>);
static_assert(!argmin::constrained_values<bounded_problem>);

static_assert(argmin::constrained_values<constrained_values_problem>);
static_assert(!argmin::constrained<constrained_values_problem>);

static_assert(argmin::constrained<constrained_problem>);

}

#endif
