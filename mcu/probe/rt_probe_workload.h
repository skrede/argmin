#ifndef HPP_GUARD_ARGMIN_MCU_PROBE_RT_PROBE_WORKLOAD_H
#define HPP_GUARD_ARGMIN_MCU_PROBE_RT_PROBE_WORKLOAD_H

// Shared fixed-N real-time probe workload for the on-device allocation proof.
//
// One header, two consumers:
//
//   * link_gate_main.cpp (the -fno-exceptions cross-compile-link CI gate) --
//     compiled WITHOUT ARGMIN_BENCH_TRACE_ALLOC, so every arm/reset/read/gate
//     call in alloc_counter.h is an inline no-op returning zero. The gate
//     exercises pure instantiation + execution of the four RT-claimed policies
//     under arm-none-eabi + newlib-nano; a surviving throw or an unresolved
//     symbol fails the link.
//
//   * the target images (NUCLEO nucleo_main.cpp / ESP32 probe) -- compiled
//     WITH ARGMIN_BENCH_TRACE_ALLOC and a target sensor (wrap_malloc.cpp on
//     NUCLEO; heap_caps/heap_trace on ESP32) that defines the counters
//     alloc_counter.h declares extern. The same arm/reset/read/gate calls are
//     then live and report the on-device allocs/step.
//
// The four RT-claimed policies exercised here mirror the host micro alloc-gate
// suite (the steady-state driver in argmin/detail/diagnostics/steady_state_driver.h): a full warmup solve()
// walks the descent trajectory unarmed so every one-time / lazy allocation is
// warm, reset() returns to the start OUTSIDE any armed window, a short unarmed
// transient re-enters steady descent, and only then does the armed window
// measure the pure per-step traffic. The host fixed-N contract is 0.00
// allocs/step on these windows (see docs/rt-safety-matrix.md).
//
// The consuming TU must define EIGEN_RUNTIME_NO_MALLOC and the counting
// eigen_assert (and include alloc_counter.h) BEFORE this header so the
// Eigen-native sensor is armed; this header includes alloc_counter.h again
// (guarded, no-op) purely to name the bench API.

#include "argmin/detail/diagnostics/alloc_counter.h"

#include "argmin/solver/lm_policy.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/filter_nw_sqp_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <cmath>

// The zero-allocation-per-step claim these windows measure depends on Eigen
// routing its internal kernel temporaries (product/solve blocking buffers,
// Householder-sequence applies) to the stack. Eigen only enables that path when
// EIGEN_ALLOCA is defined, and its auto-detection
//   EIGEN_OS_LINUX || EIGEN_OS_MAC || (defined alloca)
// finds nothing on a newlib target built at strict -std=c++20: __STRICT_ANSI__
// hides newlib's alloca macro. The resulting failure is silent -- every such
// temporary becomes a per-call heap allocation, with no diagnostic and no
// effect from EIGEN_STACK_ALLOCATION_LIMIT (which is not consulted on that
// branch). It measured as kraft_slsqp at 2.80 allocs/step on the NUCLEO while
// the host, immune via its OS branch, read 0.00.
//
// No board runs in CI, so this compile-time guard is the only mechanical check
// that the flag is still in force: it fails the board-free cross-compile-link
// gate, which does run per-commit. ARGMIN_EIGEN_MCU_DEFS supplies the define
// for every consumer of this header; see docs/embedded.md for the mechanism and
// the stack-budgeting rule that bounds the temporaries it puts back on the stack.
#ifndef EIGEN_ALLOCA
#error "EIGEN_ALLOCA is not defined: Eigen will heap-allocate its internal kernel temporaries on every call, silently breaking the zero-allocation-per-step contract this probe measures. Define EIGEN_ALLOCA=__builtin_alloca (see docs/embedded.md)."
#endif

// Pass/fail verdict over the sensor primitives, owned here because this probe
// is its sole consumer: the library ships the allocation sensor mechanism
// (arm/reset/read, the eigen_assert hook, the max-of-two-sensors combinator)
// and never the assertion. The witness-band machinery travels with the gate.
//
// min_per_step encodes the reviewer-measured pre-hoist witness band: a policy
// expected to allocate must be seen to allocate at least this many times per
// armed step, proving the sensor is not blind. A value of 0 marks a policy
// expected to already be allocation-free, held to the zero-allocation gate
// immediately. Defining ARGMIN_ALLOC_GATE_EXPECT_ZERO flips every probe to the
// zero-allocation gate -- the post-hoist acceptance mode.
#ifdef ARGMIN_BENCH_TRACE_ALLOC

#include <cstdio>
#include <cstddef>
#include <algorithm>

namespace argmin::detail::bench
{

inline int evaluate_gate(const char* label, std::size_t armed_steps,
                         std::size_t min_per_step) noexcept
{
    const std::size_t eigen_c = read_eigen_malloc_count();
    const std::size_t c_alloc = read_c_alloc_count();
    const std::size_t observed = std::max(eigen_c, c_alloc);
    const double per_step = armed_steps
        ? static_cast<double>(observed) / static_cast<double>(armed_steps)
        : static_cast<double>(observed);

    // %lu with an explicit unsigned-long cast rather than %zu: some embedded
    // libcs (e.g. newlib-nano) do not implement the %z length modifier, and
    // this prints identically on a hosted libc.
    std::printf("  [alloc-gate] %-18s eigen_malloc=%lu c_alloc=%lu "
                "armed_steps=%lu per_step=%.2f\n",
                label, static_cast<unsigned long>(eigen_c),
                static_cast<unsigned long>(c_alloc),
                static_cast<unsigned long>(armed_steps), per_step);

#ifdef ARGMIN_ALLOC_GATE_EXPECT_ZERO
    if(observed != 0)
    {
        std::fprintf(stderr,
            "  [alloc-gate] %s FAIL: zero-alloc gate saw %lu allocations\n",
            label, static_cast<unsigned long>(observed));
        return 1;
    }
    std::printf("  [alloc-gate] %s PASS (zero-alloc gate)\n", label);
    return 0;
#else
    if(min_per_step == 0)
    {
        if(observed != 0)
        {
            std::fprintf(stderr,
                "  [alloc-gate] %s FAIL: policy expected allocation-free, "
                "saw %lu\n", label, static_cast<unsigned long>(observed));
            return 1;
        }
        std::printf("  [alloc-gate] %s PASS (allocation-free)\n", label);
        return 0;
    }
    if(per_step < static_cast<double>(min_per_step))
    {
        std::fprintf(stderr,
            "  [alloc-gate] %s FAIL: pre-fix witness expected >= %lu/step, "
            "saw %.2f/step -- gate may be blind\n",
            label, static_cast<unsigned long>(min_per_step), per_step);
        return 1;
    }
    std::printf("  [alloc-gate] %s PASS (pre-fix witness %.2f/step >= "
                "%lu/step)\n", label, per_step,
                static_cast<unsigned long>(min_per_step));
    return 0;
#endif
}

}

#else

namespace argmin::detail::bench
{

// No-op twin, so the -fno-exceptions link gate (built without
// ARGMIN_BENCH_TRACE_ALLOC) compiles unchanged. With no sensor there is
// nothing to evaluate, so the gate trivially passes.
inline int evaluate_gate(const char*, std::size_t, std::size_t) noexcept
{
    return 0;
}

}

#endif

namespace argmin::mcu
{

// Fixed-2 least-squares Rosenbrock for lm_policy, with an initial_point() so it
// drives the same steady-state warmup path as the constrained fixtures.
struct fixed_rosenbrock_ls
{
    static constexpr int problem_dimension = 2;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_residuals() const { return 2; }

    [[nodiscard]] Eigen::Vector<double, 2> initial_point() const
    {
        return Eigen::Vector<double, 2>(-1.0, 1.0);
    }

    [[nodiscard]] double value(const Eigen::Vector<double, 2>& x) const
    {
        const double r0 = 1.0 - x(0);
        const double r1 = std::sqrt(5.0) * (x(1) - x(0) * x(0));
        return 0.5 * (r0 * r0 + r1 * r1);
    }

    void residuals(const Eigen::Vector<double, 2>& x, Eigen::VectorXd& r) const
    {
        r(0) = 1.0 - x(0);
        r(1) = std::sqrt(5.0) * (x(1) - x(0) * x(0));
    }

    void jacobian(const Eigen::Vector<double, 2>& x, Eigen::MatrixXd& J) const
    {
        J(0, 0) = -1.0;
        J(0, 1) = 0.0;
        J(1, 0) = -2.0 * std::sqrt(5.0) * x(0);
        J(1, 1) = std::sqrt(5.0);
    }
};

// One steady-state window for a single fixed-N policy on one problem. Mirrors
// the host steady-state driver (argmin/detail/diagnostics/steady_state_driver.h). Returns a process-style rc
// (0 pass). min_per_step is 0 for the RT-claimed policies (held to the
// zero-allocation gate immediately). label feeds the on-device report line.
template <typename Policy, typename Problem>
int measure_steady_window(const char* label, Policy policy,
                          const Problem& problem, std::size_t min_per_step)
{
    auto x0 = problem.initial_point();
    argmin::solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-10);

    argmin::step_budget_solver solver{policy, problem, x0, opts};

    solver.solve();       // full unarmed warmup: warms every one-time alloc
    solver.reset(x0);     // return to start OUTSIDE any armed region
    solver.step();        // short unarmed transient back into steady descent
    solver.step();

    constexpr std::size_t hot_steps = 10;
    argmin::detail::bench::reset_alloc_count();
    argmin::detail::bench::arm_alloc_trace();
    bool terminated = false;
    for(std::size_t i = 0; i < hot_steps; ++i)
    {
        const auto r = solver.step();
        if(r.policy_status.has_value())
            terminated = true;
    }
    argmin::detail::bench::disarm_alloc_trace();

    if(terminated)
        return 1;         // window was not pre-convergence steady state

    return argmin::detail::bench::evaluate_gate(label, hot_steps, min_per_step);
}

// Exercise all four RT-claimed policies (line-search SQP, LM, NW-SQP,
// filter-NW-SQP) on fixed-N problems. Returns 0 iff every window passes its
// gate. Under the link-gate build (no ARGMIN_BENCH_TRACE_ALLOC) the gate calls
// are no-ops, so this returns 0 after simply instantiating + running each
// policy -- which is exactly the -fno-exceptions link surface the CI gate
// needs. Under a target build it reports the real on-device allocs/step.
inline int run_all_rt_policies()
{
    using argmin::hs071;

    int rc = 0;

    // Line-search SQP (kraft_slsqp) on fixed-4 HS071.
    rc |= measure_steady_window(
        "kraft_slsqp hs071",
        argmin::kraft_slsqp_policy<hs071<>::problem_dimension>{}, hs071<>{}, 0);

    // NW-SQP on fixed-4 HS071.
    rc |= measure_steady_window(
        "nw_sqp hs071",
        argmin::nw_sqp_policy<hs071<>::problem_dimension>{}, hs071<>{}, 0);

    // filter-NW-SQP on fixed-4 HS071.
    rc |= measure_steady_window(
        "filter_nw_sqp hs071",
        argmin::filter_nw_sqp_policy<hs071<>::problem_dimension>{}, hs071<>{}, 0);

    // Levenberg-Marquardt on fixed-2 least-squares Rosenbrock.
    rc |= measure_steady_window(
        "lm rosenbrock_ls",
        argmin::lm_policy<fixed_rosenbrock_ls::problem_dimension>{},
        fixed_rosenbrock_ls{}, 0);

    return rc;
}

}

#endif
