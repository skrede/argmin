# argmin

![Linux](https://github.com/skrede/argmin/actions/workflows/linux.yml/badge.svg)
![macOS](https://github.com/skrede/argmin/actions/workflows/macos.yml/badge.svg)
![Windows](https://github.com/skrede/argmin/actions/workflows/windows.yml/badge.svg)
![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)
![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)
![Status](https://img.shields.io/badge/status-public%20preview-orange.svg)

A header-only C++20 nonlinear optimization library for embedded, real-time, and control applications.

Every solver is a policy that plugs into `step_budget_solver<Policy>` and exposes a uniform `step()` / `step_n(budget)` / `solve()` interface. The library is built around three properties that matter when the optimizer sits *inside* a larger system: compile-time dimensions that propagate into Eigen types, header-only distribution with no compiled artifacts, and step-level execution control suitable for control loops, IK chains, or MPC ticks.

## Status

**Public preview.** The API surface and policy lineup are stabilizing; expect breaking changes through the `v0.3.x` line. The library builds and tests on Linux, macOS, and Windows and is consumed via CMake `FetchContent` (see below). Known limitations at the current HEAD are listed in [Choosing a Solver](docs/choosing-a-solver.md).

## Features

- **Policy-based solvers:** every algorithm is a policy behind one uniform interface -- swap `lbfgsb_policy` for `kraft_slsqp_policy` without touching the calling code.
- **Broad algorithm set:** L-BFGS-B, SLSQP (Kraft and filter variants), MMA / GCMMA / CCSA, Levenberg-Marquardt, Projected Gauss-Newton, BOBYQA, CMA-ES, and an augmented-Lagrangian meta-policy -- covering unconstrained, box, equality/inequality, least-squares, derivative-free, and global problems.
- **Compile-time dimensions:** problem sizes propagate into Eigen types for stack allocation and zero-overhead fixed-`N` solves.
- **Step-level execution control:** drive a solver one iteration at a time -- for a control loop, an IK chain, or an MPC tick that cannot afford a blocking `solve()`.
- **Embeddable / real-time-minded:** a passive kernel that owns no thread, timer, or loop; several policies are certified allocation-free at a fixed dimension under a CI allocation gate (see the [RT safety matrix](docs/rt-safety-matrix.md)).
- **Exception-free / RTTI-free:** builds `-fno-exceptions -fno-rtti`; a CI job runs an instantiation probe to keep it that way.
- **Header-only:** no compiled artifacts; Eigen is the sole dependency.
- **Textbook-cited:** every algorithm header cites its source (Kochenderfer & Wheeler 2e, Nocedal & Wright 2e, and the original papers).

## Scope

argmin is a nonlinear optimization kernel -- and deliberately only that. It
stays a small, composable library rather than a modeling framework, so the
following are **non-goals**, each better served by a purpose-built tool:

- **Automatic differentiation** -- bring your own gradients and Jacobians (satisfy the `differentiable` concept); pair with an AD tool if you need them generated.
- **A modeling / DSL layer** (JuMP, CasADi, Pyomo) -- argmin optimizes a problem you express directly in C++, not a symbolic model.
- **Sparse, large-scale interior-point NLP** (Ipopt) -- argmin targets small-to-medium *dense* problems that fit inside a control loop, not thousands of variables.
- **Linear algebra** -- Eigen only; argmin reinvents no matrix math.

The guiding principle is *library, not framework*: argmin owns the optimization
step and stays out of everything else.

## Requirements

- C++20 compiler (floor): GCC 10+, Clang 13+, MSVC 17.0+; a C++23 toolchain (GCC 14+, Clang 18+, MSVC 17.10+) is used when available.
- CMake 3.28+
- Eigen 3.4+ (auto-fetched via `FetchContent` if not provided)

## Quick Install

### CMake FetchContent (recommended)

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

This pulls argmin and its Eigen dependency automatically. No manual installation required.

### find_package

After `cmake --install`:

```cmake
find_package(argmin CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE argmin::argmin)
```

## A first solve

Minimize the 2-D Rosenbrock function with L-BFGS-B:

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
              << "x:      " << result.x.transpose()            << '\n'
              << "f(x):   " << result.objective_value          << '\n';
}
```

Bring your own problem by satisfying the `differentiable` concept (and `bound_constrained`, `constrained_values`, `constrained` as appropriate); see `lib/argmin/include/argmin/formulation/concepts.h`. More runnable programs -- constrained, least-squares, derivative-free, and step-level control-loop usage -- live in [examples/](examples/).

## Documentation

- [Getting Started](docs/getting-started.md) -- install, a first solve, and defining your own problem.
- [Documentation Index](docs/README.md) -- guides and references.
- [Choosing a Solver](docs/choosing-a-solver.md) -- a problem-class-to-policy map.
- [Real-Time Safety Matrix](docs/rt-safety-matrix.md) -- per-solver allocation-freedom, bounded iterations, and determinism.
- [Examples](examples/) -- runnable programs for each problem class.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow, coding conventions, and commit-message format.

## License

Apache License 2.0 -- see [LICENSE](LICENSE) for the full text.

Copyright 2026 Aleksander Skrede.

## Declaration of AI use

This library has been -- and will be -- developed with extensive use of Claude Code (Sonnet, Opus, and Fable).
