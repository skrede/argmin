# argmin

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)
![Status](https://img.shields.io/badge/status-public%20preview-orange.svg)

Header-only nonlinear optimization library with a C++20 language floor (using C++23 features, such as `std::expected`, when the toolchain provides them). Every solver is a policy that plugs into `step_budget_solver<Policy>` and exposes a uniform `step()` / `solve()` / `step_n(budget)` interface. Algorithms are textbook-cited (Kochenderfer & Wheeler 2e, Nocedal & Wright 2e, plus original papers).

The library is built around three properties that matter when the optimizer sits inside a larger system: compile-time dimensions that propagate into Eigen types, header-only distribution with no compiled artifacts, and step-level execution control suitable for control loops, IK chains, or MPC ticks.

## Status

**Public preview.** API surface and policy lineup are stabilizing; expect breaking changes through the v0.3.x line. Test suite is 719 cases / 10846 assertions. A cross-library benchmark harness lives under `benchmarks/` and is documented separately.

What is solid:
- L-BFGS-B (`lbfgsb`, `byrd_lbfgsb`) — unconstrained and box-constrained smooth problems
- BOBYQA (`bobyqa`, plus `multistart_policy` wrapping it) — derivative-free, small `n`
- Levenberg-Marquardt (`lm`) and Projected Gauss-Newton (`projected_gn`, `projected_gradient_gn`) — least-squares
- Kraft SLSQP (`kraft_slsqp`, `filter_slsqp`) — equality + inequality constrained
- MMA / GCMMA / CCSA-quadratic — separable convex approximation
- Augmented Lagrangian (`augmented_lagrangian`) — meta-policy wrapping any unconstrained inner
- CMA-ES and restarting CMA-ES — bound-constrained global

Known limitations at the current HEAD:
- `nw_sqp` and `filter_nw_sqp` can accept an infeasible iter-0 step on HS071-style problems and park there. Use `kraft_slsqp` or `filter_slsqp` instead until this is closed.
- COBYLA was hotfixed for HS024 but is unreliable on HS048/050/051-class problems; a full Powell-faithfulness rewrite is scheduled for v0.3.x.
- CMA-ES converges to local minima on Ackley and Schwefel; an active-CMA variant is pending for landscapes with very flat basins.

See [docs/choosing-a-solver.md](docs/choosing-a-solver.md) for a problem-class-to-policy map.

## Requirements

- C++20 compiler (floor): GCC 10+, Clang 13+, MSVC 17.0+; a C++23 toolchain (GCC 14+, Clang 18+, MSVC 17.10+) is used when available
- CMake 3.28+
- Eigen 3.4+ (auto-fetched via FetchContent if not provided)

## Install

### CMake FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(
    argmin
    GIT_REPOSITORY https://github.com/skrede/argmin.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(argmin)

target_link_libraries(my_app PRIVATE argmin::argmin)
```

### find_package

After `cmake --install`:

```cmake
find_package(argmin CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE argmin::argmin)
```

## Quickstart

Minimise the 2-D Rosenbrock function with L-BFGS-B:

```cpp
#include <argmin/argmin.h>
#include <argmin/solver/lbfgsb_policy.h>
#include <argmin/test_functions/rosenbrock.h>

#include <Eigen/Core>
#include <iostream>

int main()
{
    using namespace argmin;

    rosenbrock<> problem{.n = 2};
    Eigen::VectorXd x0{{-1.2, 1.0}};

    solver_options opts;
    opts.set_gradient_threshold(1e-6);
    opts.max_iterations = 500;

    step_budget_solver solver{lbfgsb_policy<>{}, problem, x0, opts};
    auto result = solver.solve(opts);

    std::cout << "status: " << static_cast<int>(result.status) << '\n'
              << "x: "      << result.x.transpose()           << '\n'
              << "f(x): "   << result.objective_value         << '\n';
}
```

Bring your own problem by satisfying the `differentiable` concept (and `bound_constrained`, `constrained_values`, `constrained` as appropriate). See `lib/argmin/include/argmin/formulation/concepts.h`.

## Step-level execution

`step_budget_solver` exposes the same algorithms in three execution modes:

```cpp
solver.step();             // single iteration
solver.step_n(budget);     // bounded run
solver.solve(opts);        // run to convergence / max_iterations
```

This matters when the optimizer is one stage of a larger loop — IK with a frame budget, MPC ticks bounded by the control period, or any setting that cannot afford a blocking `solve()`.

## Embeddability and real-time use

argmin is a passive kernel that owns no thread, scheduler, timer, or loop — the caller owns
scheduling. The budget is a compile-time property of the driver you pick, over a shared
steppable policy kernel:

- `step_budget_solver` — iteration cap + convergence, **does not include `<chrono>`**, so it
  is wall-clock-free by construction.
- `time_budget_solver` / `step_and_time_budget_solver` — add a wall-clock deadline (the only
  drivers that read a clock).
- `stepper` — the purest primitive: no internal loop; the caller's executor owns the budget
  and calls `step()` plus a converged-query. Best fit for a deadline-scheduled control loop.

**Allocation-freedom at fixed `N`.** Under the allocation-counting gate (a global
`operator new`/`delete` counter plus `malloc` interposition, armed on the warm-started
steady-state step), the following policies are certified allocation-free at a fixed
compile-time dimension in **zero mode** (the gate fails the build if a single allocation
fires on an armed step): `kraft_slsqp`, `filter_slsqp`, `tr_sqp`, `filter_trsqp`, `lbfgsb`,
`byrd_lbfgsb`, `augmented_lagrangian`, `projected_gn`, and `projected_gradient_gn`. The SQP
gates certify the warm-started pre-convergence operating regime; two characterized
off-hot-loop residuals (post-convergence feasibility restoration; box/inequality free-set
restart) are documented in the matrix.

**Not yet allocation-free** (their gates stay in witness mode, so no zero-allocation claim is
made): `nw_sqp` and `filter_nw_sqp` carry a ~32 allocations/step residual on the hot loop, and
`lm` measures ~9 allocations/step at HEAD. The nw-family residual is not zero-mode this cycle,
but it is reachable via a bounded-dimension plumbing fix (deferred solver work) rather than
being walled off by the linear-algebra backend — bounded, fixed-shape dense factorizations are
themselves allocation-free. The remaining derivative-free and stochastic policies (`bobyqa`,
`cobyla`, `cmaes`, `isres`, `mma`, `gcmma`, `ccsa_quadratic`) are not RT-claimed.

**Exception-free / RTTI-free.** The library has zero `throw` sites and uses no
`dynamic_cast`/`typeid`. A dedicated CI job builds and *runs* an instantiation probe under
`-fno-exceptions -fno-rtti` at the C++20 floor, so a surviving throw on any RT-claimed policy
path fails the build rather than passing a header parse.

The full per-solver breakdown — allocation-free, bounded-iterations, wall-clock-free,
exceptions-off-clean, deterministic(seeded), each cell citing its proving artifact — is in
[docs/rt-safety-matrix.md](docs/rt-safety-matrix.md).

## Project layout

```
lib/argmin/include/argmin/   public headers (header-only library)
  solver/                      policies (lbfgsb, kraft_slsqp, cmaes, ...)
  detail/                      implementation details (BFGS variants, QR, QP, ...)
  test_functions/              Hock-Schittkowski, More-Garbow-Hillstrom, K&W test set
  formulation/                 problem concepts
tests/                         unit and compile-time tests
benchmarks/                    argmin_bench (cross-library) and publish_bench (publication-grade)
```

## Building tests and benchmarks

```bash
cmake --preset dev            # tests
cmake --build build/dev
ctest --test-dir build/dev

cmake --preset bench          # argmin_bench + publish_bench
cmake --build build/bench --target argmin_bench publish_bench
```

Publication benchmark methodology, provenance requirements, and baseline
eligibility rules are documented in
[docs/benchmarking/publish-bench.md](docs/benchmarking/publish-bench.md).

## Citations

Every algorithm header cites its source. Primary references:

- Kochenderfer & Wheeler, *Algorithms for Optimization*, 2nd ed. (MIT Press, 2025)
- Nocedal & Wright, *Numerical Optimization*, 2nd ed. (Springer, 2006)
- Powell, *The BOBYQA algorithm for bound constrained optimization without derivatives* (DAMTP 2009/NA06)
- Svanberg, *A class of globally convergent optimization methods based on conservative convex separable approximations* (SIAM J. Optim. 12(2), 2002)
- Hansen, *The CMA Evolution Strategy: A Tutorial* (arXiv:1604.00772, 2023 revision)
- Kraft, *A software package for sequential quadratic programming* (DFVLR-FB 88-28, 1988)
- Byrd, Lu, Nocedal, Zhu, *A Limited Memory Algorithm for Bound Constrained Optimization* (SIAM J. Sci. Comput. 16(5), 1995)
- Fletcher & Leyffer, *Nonlinear programming without a penalty function* (Math. Programming 91, 2002)
- Runarsson & Yao, *Search biases in constrained evolutionary optimization* (IEEE Trans. SMC-C 35(2), 2005)

## License

Apache 2.0. See [LICENSE](LICENSE).
