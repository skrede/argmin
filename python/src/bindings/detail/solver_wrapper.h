#ifndef HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_SOLVER_WRAPPER_H
#define HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_SOLVER_WRAPPER_H

#include "bindings/detail/errors.h"
#include "bindings/detail/validate.h"
#include "bindings/detail/problem_adapter.h"

#include "argmin/types.h"
#include "argmin/result/status.h"
#include "argmin/solver/options.h"
#include "argmin/detail/solver_core.h"
#include "argmin/result/solve_result.h"
#include "argmin/solver/step_budget_solver.h"

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>

#include <Eigen/Core>

#include <atomic>
#include <memory>
#include <cstdint>
#include <utility>
#include <optional>

namespace argmin::python
{

template <typename Policy, typename Adapter>
class solver_wrapper
{
public:
    using driver_type = step_budget_solver<Policy, dynamic_dimension, Adapter>;
    using policy_options_type = policy_options_t<Policy, double>;
    using result_type = solve_result<double>;

    solver_wrapper(std::unique_ptr<Adapter> adapter, const vector<double>& x0,
                   const solver_options<>& opts, const policy_options_type& policy_opts)
        : adapter_(std::move(adapter)), options_(opts), policy_options_(policy_opts)
    {
        adapter_->set_abort_hook(&abort_from_adapter, this);
        driver_.emplace(*adapter_, x0, options_, policy_options_);
        adapter_->rethrow_if_latched();
    }

    solver_wrapper(const solver_wrapper&) = delete;
    solver_wrapper& operator=(const solver_wrapper&) = delete;

    result_type solve()
    {
        return run([](driver_type& driver) { return driver.solve(); });
    }

    result_type step_n(int budget)
    {
        check_positive_dimension(budget, "budget");
        const auto span = static_cast<std::uint32_t>(budget);
        return run([span](driver_type& driver) { return driver.step_n(span); });
    }

    solver_status step()
    {
        const busy_guard guard(*this);
        {
            const nanobind::gil_scoped_release released;
            driver_->step();
        }
        adapter_->rethrow_if_latched();
        return driver_->status();
    }

    void reset(const vector<double>& x0)
    {
        restart(x0, [](driver_type& driver, const vector<double>& start) { driver.reset(start); });
    }

    void reset_clear(const vector<double>& x0)
    {
        restart(x0,
                [](driver_type& driver, const vector<double>& start)
                { driver.reset_clear(start); });
    }

    void abort() { driver_->abort(); }

    [[nodiscard]] solver_status status() const { return driver_->status(); }
    [[nodiscard]] vector<double> x() const { return driver_->state().x; }
    [[nodiscard]] double gradient_norm() const { return driver_->gradient_norm(); }
    [[nodiscard]] double constraint_violation() const { return driver_->constraint_violation(); }
    [[nodiscard]] int dimension() const { return adapter_->dimension(); }

    // The effective configuration read back as the library's own aggregates:
    // the nonlinear surface takes its options as keyword arguments with no
    // exposed type, so without this an unspecified keyword's default is not
    // observable from the interpreter at all.
    [[nodiscard]] solver_options<> options() const
    {
        solver_options<> snapshot = options_;
        snapshot.convergence = driver_->convergence();
        return snapshot;
    }

    [[nodiscard]] policy_options_type policy_options() const { return driver_->policy().options; }

    [[nodiscard]] auto callable_handles() const noexcept { return adapter_->callable_handles(); }
    void release_callables() noexcept { adapter_->release_callables(); }

private:
    class busy_guard
    {
    public:
        explicit busy_guard(solver_wrapper& owner) : owner_(owner)
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
        solver_wrapper& owner_;
    };

    static void abort_from_adapter(void* owner) noexcept
    {
        auto* self = static_cast<solver_wrapper*>(owner);
        if(self->driver_)
            self->driver_->abort();
    }

    template <typename Body>
    result_type run(Body&& body)
    {
        const busy_guard guard(*this);
        result_type produced;
        {
            const nanobind::gil_scoped_release released;
            produced = body(*driver_);
        }
        // Before the result object is built, so a caller never has to check two
        // channels: a latched failure raises instead of returning an answer.
        adapter_->rethrow_if_latched();
        return produced;
    }

    template <typename Body>
    void restart(const vector<double>& x0, Body&& body)
    {
        const busy_guard guard(*this);
        check_vector_length(x0, adapter_->dimension(), "x0");
        check_all_finite(x0, "x0");
        {
            const nanobind::gil_scoped_release released;
            body(*driver_, x0);
        }
        adapter_->rethrow_if_latched();
    }

    // Declaration order is the lifetime guarantee: the policy stores a raw
    // pointer to the problem and the library requires the problem to outlive
    // the solver. Members are destroyed in reverse declaration order, so the
    // adapter must be declared first to be destroyed last; swapping these two
    // is a use-after-free during destruction that no test of normal operation
    // would reveal.
    std::unique_ptr<Adapter> adapter_;
    solver_options<> options_;
    policy_options_type policy_options_;
    std::atomic<bool> busy_{false};
    std::optional<driver_type> driver_;
};

template <typename Wrapper>
int wrapper_traverse(PyObject* self, visitproc visit, void* arg)
{
    Py_VISIT(Py_TYPE(self));
    if(!nanobind::inst_ready(self))
        return 0;
    for(PyObject* held : nanobind::inst_ptr<Wrapper>(self)->callable_handles())
        Py_VISIT(held);
    return 0;
}

template <typename Wrapper>
int wrapper_clear(PyObject* self)
{
    nanobind::inst_ptr<Wrapper>(self)->release_callables();
    return 0;
}

// Without these two the collector cannot break a cycle that runs through a
// held callable, and every such solver leaks for the interpreter's lifetime.
template <typename Wrapper>
inline PyType_Slot wrapper_slots[]{
    {Py_tp_traverse, reinterpret_cast<void*>(&wrapper_traverse<Wrapper>)},
    {Py_tp_clear, reinterpret_cast<void*>(&wrapper_clear<Wrapper>)},
    {0, nullptr}};

}

#endif
