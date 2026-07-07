# argmin

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)
![Status](https://img.shields.io/badge/status-public%20preview-orange.svg)

Header-only nonlinear optimization library with a C++20 language floor (using C++23 features, such as `std::expected`, when the toolchain provides them). Every solver is a policy that plugs into `basic_solver<Policy>` and exposes a uniform `step()` / `solve()` / `step_n(budget)` interface. Algorithms are textbook-cited (Kochenderfer & Wheeler 2e, Nocedal & Wright 2e, plus original papers).

The library is built around three properties that matter when the optimiser sits inside a larger system: compile-time dimensions that propagate into Eigen types, header-only distribution with no compiled artifacts, and step-level execution control suitable for control loops, IK chains, or MPC ticks.

## Status

**Public preview.** API surface and policy lineup are stabilising; expect breaking changes through the v0.2.x line. Test suite is 368 cases / 2852 assertions. A cross-library benchmark harness lives under `benchmarks/` and is documented separately.

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

    basic_solver solver{lbfgsb_policy<>{}, problem, x0, opts};
    auto result = solver.solve(opts);

    std::cout << "status: " << static_cast<int>(result.status) << '\n'
              << "x: "      << result.x.transpose()           << '\n'
              << "f(x): "   << result.objective_value         << '\n';
}
```

Bring your own problem by satisfying the `differentiable` concept (and `bound_constrained`, `constrained_values`, `constrained` as appropriate). See `lib/argmin/include/argmin/formulation/concepts.h`.

## Step-level execution

`basic_solver` exposes the same algorithms in three execution modes:

```cpp
solver.step();             // single iteration
solver.step_n(budget);     // bounded run
solver.solve(opts);        // run to convergence / max_iterations
```

This matters when the optimiser is one stage of a larger loop — IK with a frame budget, MPC ticks bounded by the control period, or any setting that cannot afford a blocking `solve()`.

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

## Citations

Every algorithm header cites its source. Primary references:

- Kochenderfer & Wheeler, *Algorithms for Optimization*, 2nd ed. (MIT Press, 2024)
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
