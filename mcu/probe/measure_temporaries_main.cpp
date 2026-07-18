// Host measurement TU: the largest SINGLE Eigen stack temporary each of the four
// RT-claimed policies requests at its fixed N.
//
// Mechanism -- an EIGEN_ALLOCA recording shim (max-watermark per call). Eigen
// uses EIGEN_ALLOCA(size) inside ei_declare_aligned_stack_constructed_variable
// (Eigen 3.4.0, Eigen/src/Core/util/Memory.h) as a pointer-returning expression
// whose alloca MUST land in the CALLER's frame. So the shim is a macro
// comma-expression, never a function (a function would move the alloca into its
// own frame and free it on return). It records max(hwm, n) then alloca(n).
//
// Host is sufficient and target-accurate: the requested byte count is a pure
// function of matrix dimensions and the alignment constant. Compiling with the
// image's Eigen macro set (EIGEN_DONT_VECTORIZE, EIGEN_MAX_ALIGN_BYTES=8) makes
// the alloca padding match the arm image dimension-for-dimension. This is the
// same shim mechanism that produced kraft's already-known 79 + 223 B pair.
//
// Do NOT pass -DEIGEN_ALLOCA on the command line -- the shim below provides it
// (this also satisfies the #error guard in rt_probe_workload.h:67-69).
//
// Reproducible build command (run from the repo root):
//
//   c++ -std=c++20 -O2 -DEIGEN_DONT_VECTORIZE -DEIGEN_MAX_ALIGN_BYTES=8 \
//       -I lib/argmin/include -I mcu -I build-arm-nucleo/_deps/eigen3-src \
//       mcu/probe/measure_temporaries_main.cpp -o /tmp/measure_temps
//   /tmp/measure_temps
//
// (build-arm-nucleo/_deps/eigen3-src is the FetchContent Eigen 3.4.0 the image
//  uses; a system Eigen 3.4.0 works identically.)

#include <algorithm>
#include <cstddef>
#include <cstdio>

// The recording shim -- defined BEFORE any Eigen include so Memory.h sees it.
inline std::size_t& argmin_alloca_hwm()
{
    static std::size_t m = 0;
    return m;
}
#define EIGEN_ALLOCA(n)                                                         \
    (argmin_alloca_hwm()                                                        \
         = std::max(argmin_alloca_hwm(), static_cast<std::size_t>(n)),          \
     __builtin_alloca(static_cast<std::size_t>(n)))

// Pulls in Eigen (with the shim active), the four policies, step_budget_solver,
// hs071, and the fixed_rosenbrock_ls fixture. Compiled WITHOUT
// ARGMIN_BENCH_TRACE_ALLOC, so the bench arm/reset/read calls are inline no-ops:
// we read our own high-water mark, not the alloc counter.
#include "probe/rt_probe_workload.h"

namespace
{

// Reset the high-water, then construct + warm-solve + a few steady steps for one
// fixed-N policy, and return the largest single temporary observed over the whole
// lifecycle. Mirrors the unarmed warmup path of measure_steady_window, minus the
// (no-op) alloc gate.
template <typename Policy, typename Problem>
std::size_t measure_policy_temporary(Policy policy, const Problem& problem)
{
    auto x0 = problem.initial_point();
    argmin::solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-10);

    argmin_alloca_hwm() = 0; // reset immediately before this policy's work

    argmin::step_budget_solver solver{policy, problem, x0, opts};
    solver.solve();   // full warmup solve
    solver.reset(x0); // return to start
    for(int i = 0; i < 5; ++i)
        solver.step(); // a few steady steps

    return argmin_alloca_hwm();
}

} // namespace

int main()
{
    using argmin::hs071;

    // The exact four-policy set from rt_probe_workload.h:156-184.
    const std::size_t kraft = measure_policy_temporary(
        argmin::kraft_slsqp_policy<hs071<>::problem_dimension>{}, hs071<>{});
    std::printf("[kraft_slsqp hs071] largest single temporary = %zu B\n", kraft);

    const std::size_t nw = measure_policy_temporary(
        argmin::nw_sqp_policy<hs071<>::problem_dimension>{}, hs071<>{});
    std::printf("[nw_sqp hs071] largest single temporary = %zu B\n", nw);

    const std::size_t filt = measure_policy_temporary(
        argmin::filter_nw_sqp_policy<hs071<>::problem_dimension>{}, hs071<>{});
    std::printf("[filter_nw_sqp hs071] largest single temporary = %zu B\n", filt);

    const std::size_t lm = measure_policy_temporary(
        argmin::lm_policy<argmin::mcu::fixed_rosenbrock_ls::problem_dimension>{},
        argmin::mcu::fixed_rosenbrock_ls{});
    std::printf("[lm rosenbrock_ls] largest single temporary = %zu B\n", lm);

    const std::size_t l_max = std::max(std::max(kraft, nw), std::max(filt, lm));
    std::printf("L_max=%zu B\n", l_max);

    return 0;
}
