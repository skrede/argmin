#ifndef HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_CALLBACK_LATCH_H
#define HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_CALLBACK_LATCH_H

#include "bindings/detail/errors.h"
#include "bindings/detail/validate.h"
#include "bindings/detail/callback_set.h"

#include "argmin/types.h"

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>

#include <array>
#include <limits>
#include <string>
#include <utility>
#include <exception>
#include <string_view>

namespace argmin::python
{

class callback_latch
{
public:
    using abort_hook = void (*)(void*) noexcept;

    callback_latch(int n, callback_set callbacks)
        : n_(n), calls_(std::move(callbacks))
    {
        gradient_is_callable_ = is_callable(calls_.gradient);
        lower_is_callable_ = is_callable(calls_.lower_bounds);
        upper_is_callable_ = is_callable(calls_.upper_bounds);
        lower_cache_ = cache_bound(calls_.lower_bounds, lower_is_callable_, -infinity(), "lower");
        upper_cache_ = cache_bound(calls_.upper_bounds, upper_is_callable_, infinity(), "upper");
    }

    callback_latch(const callback_latch&) = delete;
    callback_latch& operator=(const callback_latch&) = delete;

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

    [[nodiscard]] bool has_gradient() const noexcept { return gradient_is_callable_; }

    [[nodiscard]] const callback_set& callbacks() const noexcept { return calls_; }
    [[nodiscard]] bool lower_is_callable() const noexcept { return lower_is_callable_; }
    [[nodiscard]] bool upper_is_callable() const noexcept { return upper_is_callable_; }
    [[nodiscard]] const vector<double>& lower_cache() const noexcept { return lower_cache_; }
    [[nodiscard]] const vector<double>& upper_cache() const noexcept { return upper_cache_; }

    void latch_and_abort() const noexcept
    {
        latched_ = std::current_exception();
        if(hook_ != nullptr)
            hook_(owner_);
    }

    int n_;
    callback_set calls_;
    mutable std::exception_ptr latched_{};

private:
    static bool is_callable(const nanobind::object& value) noexcept
    {
        return value.is_valid() && !value.is_none() && PyCallable_Check(value.ptr()) != 0;
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

    vector<double> lower_cache_;
    vector<double> upper_cache_;
    abort_hook hook_{nullptr};
    void* owner_{nullptr};
    bool gradient_is_callable_{false};
    bool lower_is_callable_{false};
    bool upper_is_callable_{false};
};

}

#endif
