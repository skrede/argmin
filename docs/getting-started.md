# Getting Started

Zero to a first solve in a few minutes: add argmin to a CMake project, solve a
built-in problem, then wire up your own.

## Prerequisites

- A C++20 compiler (GCC 10+, Clang 13+, or MSVC 17.0+).
- CMake 3.28 or newer.
- Eigen 3.4+ -- auto-fetched by argmin when you enable `ARGMIN_CMAKE_FETCH_DEPS`, so you do not need to install it yourself.

argmin is header-only: there is no library to build or link, only headers to include.

## Add argmin to your project

In your `CMakeLists.txt`:

```cmake
include(FetchContent)
FetchContent_Declare(
    argmin
    GIT_REPOSITORY https://github.com/skrede/argmin.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(argmin)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE argmin::argmin)
```

`FetchContent_MakeAvailable(argmin)` pulls argmin and its Eigen dependency
automatically. Linking `argmin::argmin` puts the headers and Eigen on your
include path.

## Your first solve

Minimize the 2-D Rosenbrock function with L-BFGS-B (`main.cpp`):

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
    opts.max_iterations = 200;

    step_budget_solver solver{lbfgsb_policy<>{}, problem, x0, opts};
    auto result = solver.solve(opts);

    std::cout << "x:    " << result.x.transpose()   << '\n'    // -> 1 1
              << "f(x): " << result.objective_value << '\n';   // -> ~0
}
```

Build and run:

```bash
cmake -B build -DARGMIN_CMAKE_FETCH_DEPS=ON
cmake --build build
./build/my_app
```

Every solver follows this shape: pick a **policy** (`lbfgsb_policy`,
`kraft_slsqp_policy`, `lm_policy`, ...), wrap it in a `step_budget_solver`
together with the problem, a starting point, and `solver_options`, then call
`solve()` and read the `result`.

## Bring your own problem

A problem is any type that satisfies the concept the chosen policy requires --
there is no base class to inherit. Every problem declares its dimension as a
`static constexpr int problem_dimension` (use `argmin::dynamic_dimension` for a
runtime size) and exposes the members its problem class needs:

| Concept | Required members (beyond `value(x)` / `dimension()`) |
|---|---|
| `objective` | `value(x)`, `dimension()` |
| `differentiable` | `gradient(x, g)` |
| `bound_constrained` | `lower_bounds()`, `upper_bounds()` |
| `constrained` | `constraints(x, c)`, `constraint_jacobian(x, J)`, `num_equality()`, `num_inequality()` |
| `least_squares` | `residuals(x, r)`, `jacobian(x, J)`, `num_residuals()` |

Constraints are written with the equality rows first, inequality rows second,
inequalities in `g(x) >= 0` form. A small differentiable problem:

```cpp
struct paraboloid
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }
    double value(const Eigen::Vector<double, 2>& x) const { return x.squaredNorm(); }
    void gradient(const Eigen::Vector<double, 2>& x, Eigen::Vector<double, 2>& g) const { g = 2.0 * x; }
};
```

The full concept definitions are in
`lib/argmin/include/argmin/formulation/concepts.h`.

## Reading the result

`solve()` returns a `solve_result` with:

- `result.status` -- a `solver_status` enum. A successful stop is reported as
  `converged`, `xtol_reached`, or `ftol_reached` depending on which tolerance
  tripped first; `max_iterations`, `stalled`, and `invalid_problem` are the
  common non-success codes.
- `result.x` -- the solution vector.
- `result.objective_value` -- the objective at `result.x`.

For a constrained solve, the driver also exposes
`solver.constraint_violation()` (the infinity-norm of the constraint
residual).

## Choosing an execution mode

The same policy runs in three modes on `step_budget_solver`:

```cpp
solver.solve(opts);        // run to convergence / max_iterations
solver.step_n(budget);     // run at most `budget` iterations
solver.step();             // exactly one iteration
```

For a control loop or IK tick that cannot afford a blocking `solve()`, the
`stepper` primitive has no internal loop at all -- the caller's scheduler owns
the budget and calls `step()` once per tick. See
[examples/step_level_control.cpp](../examples/step_level_control.cpp).

## Next steps

- [Choosing a Solver](choosing-a-solver.md) -- match a problem class to a policy.
- [examples/](../examples/) -- runnable programs for each problem class.
- [Real-Time Safety Matrix](rt-safety-matrix.md) -- allocation-freedom and determinism per solver.
