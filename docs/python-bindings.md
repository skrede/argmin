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

Everything is bound at dynamic dimension and `double` precision. The library's
policies are scalar-generic — each takes `float` or `double` as its first
template argument — but the bindings instantiate each bound policy once, at
`double` and a dynamic dimension, because every further instantiation is paid
for in build time and binary size (see the measured cost below).

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
not. The dense solvers' `warm_start` takes a primal seed of the decision length
and an optional dual seed of at most the constraint capacity, used only when its
length equals the posed row count; the sparse solver has no capacity, so its
`warm_start` requires a preceding solve and takes both seeds at exactly the posed
lengths, an empty dual included.

**Nonlinear methods.** Four policies through the step-budget driver, importable
from `argmin`:

| name | method | Bounds |
| --- | --- | --- |
| `LbfgsbSolver` | bound-constrained limited-memory quasi-Newton | applied |
| `KraftSlsqpSolver` | sequential least-squares quadratic programming (Kraft 1988) | applied |
| `FilterSlsqpSolver` | sequential least-squares quadratic programming with filter acceptance (Fletcher and Leyffer 2002) | applied |
| `TrSqpSolver` | Byrd-Omojokun composite-step trust-region SQP, accurate mode; stops on the bound-aware KKT residual | applied |

Each exposes `solve`, `step`, `step_n`, `reset`, `reset_clear`, `abort` and
`status`, plus two read-only options snapshots: `options()` for the shared driver
configuration and `policy_options()` for the method's own. A gradient is optional
on all four and falls back to the library's own central differences. The three
SQP solvers take `constraints` and `constraint_jacobian` callables with
`num_equality` and `num_inequality` counts; inequalities are `g(x) >= 0`.

**Bounds.** All four nonlinear solvers apply `lower_bounds` and `upper_bounds`,
given as arrays or as zero-argument callables returning arrays; an unset side is
unbounded. The constrained solvers report a bound-active optimum differently,
because their stopping tests differ: on a bowl whose minimum lies beyond a
corner of the box, all three stop at that corner, but `KraftSlsqpSolver` and
`FilterSlsqpSolver` report `stalled`, since the bound-blind KKT residual they
stop on is not small there, while `TrSqpSolver` reports `converged`.

**Why the two newest solvers were added.** Both were chosen on the publication
benchmark cohort, `benchmarks/baselines/publication-regression.csv`, not on a
smoke test; the figures below tally its per-cell dispositions, and the test
suite recomputes them from it. `FilterSlsqpSolver` holds
`filter_slsqp_accurate` 26 pass / 1 expected_fail, the same as its
already-bound peer at `kraft_slsqp_accurate` 26 pass / 1 expected_fail, and it
brings no new stopping test: it shares the bound-blind KKT residual the Kraft
solver already ships with. `TrSqpSolver` holds
`tr_sqp_accurate` 22 pass / 5 expected_fail (its fast mode, which is not
bound, holds `tr_sqp_fast` 21 pass / 6 expected_fail), and it is one of only
two policies in the library that stop on the bound-aware KKT residual, which
projects the Lagrangian gradient onto the box and so can certify a solution
with an active bound; the other is `filter_trsqp`, which is not bound.

**Options.** `DenseQpOptions`, `SparseQpOptions` and `DenseActiveSetQpOptions`
are passed as arguments, the last because the active-set solver takes its own
three-field options rather than the splitting solvers'; both splitting options
types carry `validate_convexity`, which defaults on and refuses an asymmetric
objective matrix. Everything else is set through keyword arguments on a
solver's constructor and read back through the snapshots. `SolverOptions`,
`LineSearchOptions`, `LbfgsbOptions` with its `LbfgsbLineSearch` choice,
`KraftSlsqpOptions`, `FilterSlsqpOptions` with its `RestorationStrategy`,
`QpOptions` (the subproblem options both SLSQP solvers embed) and
`TrSqpOptions` are all bound over the library's own aggregates, so an exposed
default *is* the library's brace initializer rather than a second copy of it
that can drift. A test sweep parses the committed headers and asserts every one
of them, on a default-constructed instance and on the snapshot a solver reports.
`LineSearchOptions.relative_value_noise` is the half-width of an approximate
Wolfe band (Hager and Zhang 2005); it defaults to zero, which disables the band.

**Results and statuses.** `QpResult` and `SolveResult` carry the iterates,
residuals, counts and objective values. `QpResult.warm_started` is true when
either operator-splitting solver consumed a primal seed (the `warm_start()`
vector or the iterate retained from its previous solve); a dual seed whose
length does not match the posed problem is replaced by zeros without clearing
the flag, and the active-set solver always reports false. `QpSolveStatus` and
`SolverStatus` name how a solve ended. `ArgminError` and `ErrorKind` are the
error channel.

`SolverStatus.line_search_failed` is how `LbfgsbSolver` reports a step it could
not take: a line search that ends without sufficient decrease, on a non-finite
trial or on a non-positive step is never committed. The solver retries once from
a reset curvature memory, and if that search fails too (or the memory was
already empty) it holds the last accepted iterate and reports
`line_search_failed`. It is not a success status, but the held iterate is not
discarded either: near a solution a solve can end this way at a good point, so
inspect the result rather than treating it as a crash.

**Compiled test problems.** `Rosenbrock` and `ConstrainedTestProblem` are the
same problems compiled into the extension, each with a `solve_native()` entry
point that runs a solve with no interpreter callback in the loop. They exist so
an objective defined in Python can be compared against the identical objective
compiled in — which tests the boundary rather than the arithmetic. At the top
level, `Rosenbrock.solve_native` accepts the method `lbfgsb` and
`ConstrainedTestProblem.solve_native` accepts `kraft_slsqp`, `filter_slsqp` and
`tr_sqp`; the gated methods are solved through `argmin.experimental.solve_native`.

## Gated: argmin.experimental

`CobylaSolver`, `BobyqaSolver` and `CmaesSolver` each carry a confirmed defect
that can return a wrong answer without saying so. Derivative-free and stochastic
methods are a main reason to reach for this library from Python, so all three
stay reachable, but reaching them is a deliberate act: they live in a submodule
that `import argmin` does not load.

```python
from argmin.experimental import CobylaSolver
```

| name | method | known defect | publication cohort |
| --- | --- | --- | --- |
| `CobylaSolver` | derivative-free constrained optimization by linear approximations | COBYLA's bound repair can evaluate the objective outside the box under the default initial trust radius: a coordinate starting at 0.5 in [0, 0.6], beside a coordinate with a much wider range, has its first simplex vertex evaluated at 0.75. | `cobyla` 23 pass / 4 expected_fail |
| `BobyqaSolver` | bound-constrained derivative-free trust-region interpolation | BOBYQA reports a solve as converged when it ends by exhausting the 200000-pass safety cap on the internal state machine of one iteration, so a stuck run reads as a success. | `bobyqa` 4 pass / 7 expected_fail |
| `CmaesSolver` | covariance-matrix-adaptation evolution strategy | CMA-ES has no passing cell in the library's publication benchmark cohort (0 pass, 4 expected failures, 2 excluded), and it converges to local minima on the Ackley and Schwefel functions. | `cmaes` 0 pass / 4 expected_fail / 2 excluded |

Each gated solver class carries two attributes: `maturity`, which reads
`"experimental"`, and `known_defect`, the sentence in the table above, which the
class docstring also contains. Importing the submodule or constructing a gated
solver emits no warning; the import is the opt-in. The submodule also holds the
gated methods' own option and enum types — `CobylaOptions`, `BobyqaOptions`,
`TrustRegionOptions`, `CmaesOptions`, `CmaesDetectionOptions` and
`CmaesRestartStrategy` — and `argmin.experimental.solve_native`, the
compiled-problem solve for the three methods (`bobyqa` and `cmaes` on a
`Rosenbrock`, `cobyla` on a `ConstrainedTestProblem`). The top-level
`solve_native` refuses those method names with `invalid_problem` and names the
submodule entry point instead.

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
shapes, lengths and finiteness before any solver sees the data, with checks that
survive a release build. A violation raises `ArgminError`, which carries a
`.kind` attribute naming the category — `dimension_mismatch`, `invalid_bounds`,
`non_finite_input`, `capacity_exceeded`, `infeasible_start`, `invalid_problem`,
`invalid_options`, `factorization_failed`, `invalid_array`, `invalid_callback`,
`invalid_state`. Match on `.kind`, not on the message. `invalid_options` is an
option value outside its domain, such as a non-positive penalty or tolerance;
`factorization_failed` is a quadratic-program solve whose linear system could not
be factorized. A warm start the solver refuses raises like any other violation;
it is never discarded.

Bound ordering is checked by the quadratic-program solvers and by `Rosenbrock`,
which raise `invalid_bounds` for a lower bound above its upper bound. The
nonlinear solvers check each bound's length and refuse an entry that is not a
number, but they do not check ordering: an inverted box is not refused at
construction and reaches the method as given, so order the box before passing
it. A count option — `max_iterations`, `stall_window`, `policy_stall_window`,
`CmaesSolver`'s `population_size`, the `step_n` budget — must fit the unsigned
field it is stored in; zero, a negative value or a value past the field's
maximum raises `dimension_mismatch` rather than wrapping, and a value beyond a
C++ `int` is refused with `TypeError` when the arguments are converted.

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
that guard, so it remains callable from another thread while a solve runs. The
quadratic-program solvers carry the same guard on every method, `warm_start` and
`reset` included: a call entering an instance that another thread is solving on
raises `invalid_state` instead of racing it.

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

The interpolation row is `BobyqaSolver`, which is now reached through
`argmin.experimental`. The two figures are the same phenomenon at two ratios of
compiled work to boundary crossings: the interpolation method does far more
compiled work per iteration, so the same absolute boundary cost is a smaller
fraction of it. No test anywhere in this suite asserts a wall-time bound on that
path, and no tolerance or iteration budget was chosen to make it finish sooner.

Correctness across the boundary is asserted, and exactly: an interpreter callable
that forwards straight back into the compiled evaluator produces the same status,
the same iteration count, the same objective value and the same iterate, bit for
bit, on every method and problem pair tested.

The binding units are compiled `-Os`: in a release configuration nanobind's
module defaults optimize for size, and the last optimization flag wins over the
build type's `-O3`. Nanobind also links the module with `-Wl,-s`, so the
extension as built is already stripped.

### Instantiation cost, measured

Each bound policy is a template instantiation compiled into the extension, so
widening the surface costs build time and binary size. The cost of the two
newest solvers was measured as a before-and-after bracket over exactly their two
commits, with nothing else in between:

| quantity | before | after | after - before | per policy |
| --- | --- | --- | --- | --- |
| serial build (`-j1`), median (range) | 79.92 s (0.14 s) | 104.79 s (0.21 s) | +24.87 s (+31.1%) | +12.44 s |
| parallel build (`-j6`), median (range) | 18.48 s (0.05 s) | 27.94 s (0.04 s) | +9.46 s (+51.2%) | +4.73 s |
| stripped extension | 943920 B | 1181616 B | +237696 B (+25.2%) | +118848 B (116.06 KiB) |
| extension objects, summed serial compile | 70641 ms | 95729 ms | +25088 ms | +12544 ms |

The per-policy column is (after - before) / 2. Of the 25088 ms the summed object
compile time rose, the filter SLSQP binding unit accounts for 11204 ms, the
trust-region SQP binding unit for 9360 ms, and the compiled constrained test
problem, which gained two native instantiations, for 4347 ms (9859 ms to
14206 ms); the other 177 ms is spread over the unchanged units.

After the two additions the extension measured 1181616 bytes (1153.92 KiB,
1181.6 kB), with seven nonlinear policies (the four above and the three now
gated) and three quadratic-program solvers compiled in. As built equals
stripped at every measured point, because the module is linked with `-Wl,-s`.
Before the binding sources were split into one translation unit per concern,
the extension was 919344 bytes and built in 66.76 s serially (range 0.04 s) and
20.26 s at `-j6` (range 0.07 s); the split itself added 13.16 s of serial build
time and 24576 bytes, and saved 1.78 s at `-j6`.

Every build time is the median of three clean rebuilds of the extension target
from a snapshot of the measured commit, configured with the release
`python-bindings` preset, with no other compile running on the host. Compiler:
GCC 16.2.1; host: AMD Ryzen 7 5800X3D, Linux 6.18 x86_64, CPU governor
`performance`. Wall time was taken with the shell's `time` keyword; sizes are
`strip --strip-all` copies of the built module.

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
- **`byrd_lbfgsb`.** A candidate, not yet bound. It was held back while both
  L-BFGS-B policies could commit a failed line-search step; both now reject such
  a step, so binding it is a decision still to be taken
  (`byrd_lbfgsb` 19 pass / 5 expected_fail).
- **`filter_trsqp`.** Its feasibility-restoration step minimizes the squared
  norm of the full constraint vector, inequalities included, so it drives a
  satisfied inequality toward zero as if it were an equality.
- **`nw_sqp` and `filter_nw_sqp`.** Reference implementations for comparison
  work, not for production solves: on HS071-class problems both can accept a
  first step that satisfies the linearized inequality but violates the
  nonlinear one and then stop at that infeasible point (see the
  [solver-selection guide](choosing-a-solver.md)).
- **`isres`.** Its steps report no constraint violation, so the driver's
  stopping tests that refuse to stop on an infeasible step read every step as
  feasible; and its population options have no lower bound, so an integer from
  Python could size a population of zero.
- **The GCMMA variants.** Held back on their publication figures:
  `gcmma` 9 pass / 0 expected_fail / 4 excluded,
  `gcmma_raa_augmented` 10 pass / 0 expected_fail / 3 excluded and
  `gcmma_move_limit_shrink` 5 pass / 6 expected_fail / 2 excluded. Every
  excluded cell is one whose returned point passes the oracle under one
  compiler and fails it under another, built from the same source.
- **`tr_sqp`'s fast mode.** Only the accurate mode is instantiated; the fast
  mode would be a second instantiation of the same policy.
