# Python bindings

An importable extension module exposing a deliberately narrow slice of argmin to
Python. The surface is small on purpose: a binding is a distinct artifact with
distinct failure modes, and a green C++ suite says nothing about it, so
everything bound here carries its own tests at the interpreter boundary.

## Scope

The bindings are built behind `ARGMIN_BUILD_PYTHON`, an option that defaults to
`OFF`. No other build configuration, preset, or continuous-integration job turns
it on, so the extension is invisible to the rest of the project. One
continuous-integration job builds it and runs its suite; that job is the only
thing that will notice a library change breaking the bindings.

Everything is bound at dynamic dimension and `double` precision. That is not a
narrowing choice — every policy in the library hardcodes `double`, and every one
of them defaults to a dynamic dimension.

## What is bound

**Quadratic programs.** Three solvers, taking data in and returning a result,
with no callback anywhere in the loop:

| name | form |
| --- | --- |
| `DenseAdmmQpSolver(n, m)` | dense operator splitting, constructed for a decision length and a maximum row count |
| `DenseActiveSetQpSolver(n, m)` | dense active set; requires a feasible start, seeded through `warm_start()` |
| `SparseAdmmQpSolver()` | sparse operator splitting; matrices in compressed sparse column form |

Each exposes `solve`, `resolve`, `warm_start` and `reset`. A constraint count of
zero is the unconstrained program and is accepted; a decision length of zero is
not.

**Nonlinear methods.** Five policies through the step-budget driver:

| name | method |
| --- | --- |
| `LbfgsbSolver` | bound-constrained limited-memory quasi-Newton |
| `BobyqaSolver` | bound-constrained derivative-free trust-region interpolation |
| `CobylaSolver` | derivative-free constrained optimization by linear approximations |
| `SlsqpSolver` | sequential least-squares quadratic programming |
| `CmaesSolver` | covariance-matrix-adaptation evolution strategy |

Each exposes `solve`, `step`, `step_n`, `reset`, `reset_clear`, `abort` and
`status`, plus two read-only options snapshots: `options()` for the shared driver
configuration and `policy_options()` for the method's own. A gradient is optional
on the quasi-Newton method and falls back to the library's own finite
differences.

**Options.** `DenseQpOptions` and `SparseQpOptions` are passed as arguments;
everything else is set through keyword arguments on a solver's constructor and
read back through the snapshots. `SolverOptions`, `LineSearchOptions`,
`LbfgsbOptions`, `BobyqaOptions`, `CobylaOptions`, `SlsqpOptions`,
`QpSubproblemOptions`, `TrustRegionOptions`, `CmaesOptions` and
`CmaesDetectionOptions` are all bound over the library's own aggregates, so an
exposed default *is* the library's brace initializer rather than a second copy of
it that can drift. A test sweep parses the committed headers and asserts every
one of them, on a default-constructed instance and on the snapshot a solver
reports.

**Results and statuses.** `QpResult` and `SolveResult` carry the iterates,
residuals, counts and objective values. `QpStatus` and `SolverStatus` name how a
solve ended. `ArgminError` and `ErrorKind` are the error channel.

**Compiled test problems.** `Rosenbrock` and `ConstrainedTestProblem` are the
same problems compiled into the extension, each with a `solve_native()` entry
point that runs a solve with no interpreter callback in the loop. They exist so
an objective defined in Python can be compared against the identical objective
compiled in — which tests the boundary rather than the arithmetic.

## How to build and run

```sh
python3 scripts/bootstrap_python_env.py
cmake --preset python-bindings
cmake --build build/python -j2
ctest --test-dir build/python --output-on-failure
```

The bootstrap script creates a repository-local environment at `.venv` and
installs the exactly pinned, binary-only dependency set from
`python/requirements.txt` into it and nowhere else; it never writes to a system
interpreter, on any path including its failure paths. The build configures
against exactly that interpreter and fails loudly rather than falling back to one
on `PATH`.

The suite is registered as the single CTest test `python_bindings_pytest`, so it
runs with the rest of the project's tests in that tree. To run it directly:

```sh
PYTHONPATH=python PYTHONNOUSERSITE=1 .venv/bin/python -m pytest -q python/tests
```

The extension is written into `python/argmin/`, so `import argmin` works with the
repository root of the binding tree on the module search path.

## The error model

Two channels, mirroring the library's own design.

**An argument or precondition violation raises.** Every entry point validates
shapes, lengths, bound ordering and finiteness before any solver sees the data,
with checks that survive a release build. A violation raises `ArgminError`, which
carries a `.kind` attribute naming the category — `dimension_mismatch`,
`invalid_bounds`, `non_finite_input`, `capacity_exceeded`, `infeasible_start`,
`invalid_problem`, `invalid_array`, `invalid_callback`, `invalid_state`. Match on
`.kind`, not on the message.

**A solve outcome is not an exception.** Reaching an iteration cap, stalling, or
certifying a problem infeasible is an answer, and it travels as an enumeration
attribute on the returned result — `result.status` — not as a raise. A caller
that wraps a solve in `try` and forgets to inspect `status` has not handled the
common case.

An exception raised inside a callback is never allowed to unwind through the
solve loop. It is caught, stored, and routed through the solver's own abort flag;
the loop terminates through its ordinary status path and the original exception,
with its type and message intact, is re-raised after the call returns. Once
latched, no further callback re-enters the interpreter. A callback that fails on
its very first call fails during *construction*, because the policy evaluates the
objective at the starting point while initializing and there is no earlier call
for the failure to occur at.

## The lifetime contract

The Python object owns both the adapter holding your callables and the solver
itself, and the adapter is declared first so it is destroyed last. The library's
policies store a raw pointer to the problem and require it to outlive the solver;
that ownership edge is exactly what a garbage collector would otherwise violate.
Dropping every interpreter-side reference to your objective and forcing a
collection is safe: the solver holds strong references of its own.

Arrays are copies in both directions. A decision vector handed to a callback is a
fresh array; a returned array is copied into the solver. An array read off a
result owns its data, is unaffected by a later solve on the same solver, and
outlives the solver that produced it.

Solvers implement the garbage collector's traverse and clear hooks, so a solver
stored on the same object whose bound method it optimizes is collectable rather
than leaking for the interpreter's lifetime.

A re-entrant call is rejected: a second `solve`, `step` or `reset` while one is
already running raises with `invalid_state`. `abort()` deliberately does not take
that guard, so it remains callable from another thread while a solve runs.

## Performance, stated honestly

The quadratic-program path and the compiled-objective path are where competitive
numbers are honestly available. The interpreter lock is released for the whole
quadratic-program solve, and four workers running independent solves measure a
wall-time ratio of 0.26–0.28 against single-threaded on a 100-variable,
200-row problem.

An objective *defined in Python* is a different story, and it is worth stating
plainly rather than burying. Such an objective costs at least two interpreter
round trips per iteration — one to enter Python, one to convert and return —
against native iterations measured in fractions of a microsecond. The resulting
slowdown is **structural**: it is what putting an interpreter inside a loop
costs, not a defect in this binding and not something that can be tuned away at
this layer.

It is measured and printed, not asserted. `python/tests/test_native_parity.py`
solves the same two-variable Rosenbrock both ways, averaged over twenty repeats
after a warm-up, and prints the ratio on every run. A representative run on the
authoring host (Apple Clang 16, macOS arm64, CPython 3.13):

| method | compiled | defined in Python | cost factor |
| --- | --- | --- | --- |
| quasi-Newton | 12.5 µs per solve | 134.8 µs per solve | **10.8x** |
| interpolation | 141.7 µs per solve | 245.2 µs per solve | **1.7x** |

The two figures are the same phenomenon at two ratios of compiled work to
boundary crossings: the interpolation method does far more compiled work per
iteration, so the same absolute boundary cost is a smaller fraction of it. No
test anywhere in this suite asserts a wall-time bound on that path, and no
tolerance or iteration budget was chosen to make it finish sooner.

Correctness across the boundary is asserted, and exactly: an interpreter callable
that forwards straight back into the compiled evaluator produces the same status,
the same iteration count, the same objective value and the same iterate, bit for
bit, on every method and problem pair tested.

## What is not provided

Each of these is a deliberate exclusion, with the reason.

- **Wheels and packaging metadata.** Out of scope for this milestone. A sibling
  project shipped three separate packaging bugs of this class, which is enough
  evidence that packaging is its own body of work rather than a finishing touch.
  The tree is laid out so a wheel build needs no restructuring.
- **Zero-copy array views.** Copying at the boundary eliminates the retained-view
  hazard outright: a view onto a stack-lived vector, retained by Python after the
  callback returns, is a crash waiting to happen. Whether the copies cost enough
  to matter is a measurement question, and it may only be revisited behind a
  view-retention test that does not exist yet.
- **The real-time curated surface.** Meaningless with an interpreter in the loop.
  Nothing here makes, or should be read as making, any allocation-free or
  real-time claim: see the [RT safety matrix](rt-safety-matrix.md) for what the
  library itself guarantees and at what scope.
- **Fixed-dimension instantiations.** The bound surface has no dimension axis;
  adding one multiplies the instantiation cost for a capability nobody has asked
  for from Python.
- **Solver groups and schedules.** Composition machinery whose value is in a C++
  program that owns the loop.
- **Wrapper policies taking an inner policy.** Instantiation cost is the product
  of the wrapper list and the policy list, and the adapter family is deliberately
  closed at five.
- **Time-budget drivers.** The step-budget driver is structurally clock-free; a
  wall-clock budget is a different contract and would be the only clock in the
  bound surface.
- **The least-squares family.** Not free to add — it needs a residual-and-Jacobian
  adapter tier the family does not currently carry.
