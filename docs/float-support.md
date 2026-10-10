# Float support: what runs in single precision, and how it is checked

Every policy in argmin takes its scalar type as its first template argument, and
`float` is a supported scalar alongside `double`. "Supported" here is a gated
claim, not a compile check: for each algorithm family a test solves the family's
reference problems in both scalars and goes red when the `float` answer stops
tracking the `double` one. This page lists what each family's gate covers, how
the `float` defaults were chosen, and what is not claimed.

## The scalar parameter

Every policy is spelled `<Scalar, N>` (or `<Scalar>` for the dimension-agnostic
ones), where `Scalar` is `float` or `double` and defaults to `double`. The
option structs, the convergence criteria and `default_convergence` follow the
same pattern, so a `float` solve carries no `double` arithmetic from the library
side. User callbacks must use the policy's scalar exactly: an objective for a
`float` policy takes `Eigen::Vector<float, N>` and returns `float`, enforced by
the formulation concepts, with no implicit conversion at the boundary.

```cpp
argmin::rosenbrock<float> problem;
argmin::solver_options<argmin::default_convergence<float>> opts;
opts.max_iterations = 200;

argmin::step_budget_solver solver{argmin::lbfgsb_policy<float>{}, problem, problem.initial_point(), opts};
const auto result = solver.solve();
```

Accepted scalars are exactly `float` and `double`; `long double`, automatic
differentiation types and interval types are rejected by a concept.

## How the float defaults are chosen

Every precision-dependent default lives in a per-scalar table, one header per
family under `argmin/options/` (and a `detail::` table for a shared helper's
internal constants). The `double` row holds the values the library has always
used. A `float` value is derived from its `double` counterpart by preserving its
fraction of the available precision:

    t_float = eps_float ^ (ln t_double / ln eps_double)

rounded to one significant figure (`ln eps_float / ln eps_double = 0.4422`). For
example 1e-6 maps to 2e-3, 1e-8 to 3e-4 and 1e-12 to 5e-6. Dimensionless
parameters, magnitude caps that fit in `float` range, and integer budgets copy the
`double` value. Each `float` entry carries a short tag naming which of these
applies.

A `float` value departs from the rule only when the family's gate fails at the
rule value, and then it moves to the nearest value that passes. Three did:

| default | rule value | shipped | measured reason |
|---|---|---|---|
| `convergence_defaults<float>::feasibility_tolerance` | 2e-3 | 1e-3 | at 2e-3 the best-seen selection returned a `filter_nw_sqp` hs043 iterate with violation 1.53e-3 and an objective gap of 6.55e-3, against a gate of 3.45e-3 |
| `line_search_sqp_defaults<float>::feasibility_tolerance` | 2e-3 | 1e-3 | aligned with the driver value above so the line-search policies advertise the tolerance they are gated at |
| `isres_defaults<float>::feasibility_gate` | 2e-2 | 1e-3 | the faithful ISRES variant returns the lowest objective inside the gate, so at 2e-2 its hs023 result sat 1.2e-1 from the optimum in objective; 1e-3 passes on every measured seed |

The trust-region SQP feasibility tolerances in both modes also start at 1e-3,
the driver's value, rather than at their rule values (2e-2 fast, 2e-3 accurate):
the fast-mode rule value fails under the policies' own tolerances.

## Status per family

A family is listed as gated when its `float` solve reaches an accepted status and
lands within `k * sqrt(eps_float)` (relative) of the `double` solve of the same
problem, with the `double` run as the oracle. The constant `k` is stated per
family; it is wider where the method's own stopping rule is coarser (moving
asymptotes and derivative-free methods stop on model-level tolerances) or where
the two scalars cannot share a trajectory (a stochastic search rounds its samples
differently from the first generation on). Every gate runs in the Linux, macOS and
Windows build jobs, which select the `float` test label explicitly.

<!-- BEGIN GENERATED: float-status -->
| claim | guaranteed? | evidence |
|---|---|---|
| QP solvers (dense_active_set_qp, dense_admm_qp, sparse_admm_qp) in float track double | **yes** *(gated)* (on the seeded convex family, k = 1) | dense active-set QP in float tracks the double solution of the seeded convex family at its own defaults; dense splitting QP in float tracks the double solution of the seeded convex family at its own defaults; sparse splitting QP in float tracks the double solution of the seeded convex family at its own defaults; gcc-14; clang-18; build (Xcode 15.4); build (Xcode 16.2); build (Windows) |
| L-BFGS-B (lbfgsb, byrd_lbfgsb) in float tracks double | **yes** *(gated)* (unconstrained and active-bound Rosenbrock, k = 10) | L-BFGS-B in float tracks double on an unconstrained and an active-bound Rosenbrock at its own defaults; gcc-14; clang-18; build (Xcode 15.4); build (Xcode 16.2); build (Windows) |
| line-search SQP (kraft_slsqp, nw_sqp, filter_slsqp, filter_nw_sqp) in float tracks double | **yes** *(gated)* (constrained Hock-Schittkowski fixtures, k = 10) | kraft_slsqp in float tracks double on constrained Hock-Schittkowski fixtures at its own defaults; nw_sqp in float tracks double on constrained Hock-Schittkowski fixtures at its own defaults; filter SQP policies in float track double on constrained Hock-Schittkowski fixtures at their own defaults; gcc-14; clang-18; build (Xcode 15.4); build (Xcode 16.2); build (Windows) |
| trust-region SQP (tr_sqp, filter_trsqp, both modes) in float tracks double | **yes** *(gated)* (on the ten gated mode-and-fixture cells, k = 10; six further cells are experimental, see below) | trust-region SQP policies in float track double on constrained Hock-Schittkowski fixtures at their own defaults; gcc-14; clang-18; build (Xcode 15.4); build (Xcode 16.2); build (Windows) |
| least squares (lm, projected_gn, projected_gradient_gn) in float tracks double | **yes** *(gated)* (zero-residual and active-bound fixtures, k = 10; `projected_gradient_gn` on the active-bound fixture is experimental, see below) | least-squares policies in float track double on zero-residual and active-bound fixtures at their own defaults; gcc-14; clang-18; build (Xcode 15.4); build (Xcode 16.2); build (Windows) |
| moving asymptotes (mma, ccsa_quadratic, the three GCMMA variants) in float track double | **yes** *(gated)* (inequality-constrained fixtures, k = 30; GCMMA move-limit shrinking on hs076 is experimental, see below) | moving-asymptote policies in float track double on inequality-constrained fixtures at their own defaults; gcc-14; clang-18; build (Xcode 15.4); build (Xcode 16.2); build (Windows) |
| derivative-free (bobyqa, cobyla) in float tracks double | **yes** *(gated)* (bounded and constrained fixtures, k = 30) | derivative-free policies in float track double on bounded and constrained fixtures at their own defaults; BOBYQA in float tracks double on bounded fixtures at its own defaults; gcc-14; clang-18; build (Xcode 15.4); build (Xcode 16.2); build (Windows) |
| CMA-ES (all three variants) in float tracks double | **yes** *(gated)* (unconstrained and active-bound Rosenbrock, k = 100) | CMA-ES in float tracks double on Rosenbrock at its own defaults; gcc-14; clang-18; build (Xcode 15.4); build (Xcode 16.2); build (Windows) |
| ISRES (all three variants) in float tracks double | **yes** *(gated)* (on hs023, and the original variant on hs024, k = 100; two variants on hs024 are experimental, see below) | ISRES in float tracks double on an inequality-constrained fixture at its own defaults; gcc-14; clang-18; build (Xcode 15.4); build (Xcode 16.2); build (Windows) |
| wrappers (augmented_lagrangian, multistart, restarting) over a float inner policy track double | **yes** *(gated)* (k = 10) | wrappers over a float inner policy track double at their own defaults; gcc-14; clang-18; build (Xcode 15.4); build (Xcode 16.2); build (Windows) |
<!-- END GENERATED: float-status -->

This table is generated from `scripts/rt_matrix_float.json`: edit the data file,
not the table. Its tests and jobs are resolved against live test registration and
the workflows on the same footing as the [RT-safety matrix](rt-safety-matrix.md).

### Experimental cells

Some fixture and policy combinations inside a gated family are not gated, because
the `double` run itself does not end by a convergence criterion there, so it is no
oracle for the `float` run. Each compiles, runs in a separate experimental test
that records its result and keeps the checks that hold on every build, and is
tracked as an open issue against the `double` behavior:

- **Trust-region SQP, six cells** (`tr_sqp` accurate hs071; `tr_sqp` fast hs071,
  hs035 and hs028; `filter_trsqp` accurate and fast hs043): a `double` or `float`
  run ends in a trust-radius collapse or an exhausted budget on some build.
- **`projected_gradient_gn` on the active-bound least-squares fixture**: the
  `double` run also exhausts its 200-iteration budget, making sublinear progress
  along the active bound.
- **GCMMA move-limit shrinking on hs076**: the `double` run never stops by a
  criterion (f = -3.06, -3.35 and -4.58 after 200, 2,000 and 20,000 iterations,
  against an optimum of -4.6818).
- **ISRES, the faithful and Runarsson-Yao variants on hs024**: hs024's infinite
  upper bound makes the bound-relative collapse threshold infinite, so both
  variants stop after one generation in `double` as well.

## Determinism

`float` trajectories are not bit-pinned. The guarantees on the
[determinism page](determinism.md) are established for `double`; a `float` gate
is a tolerance against `double`, never an exact comparison. The stochastic
policies draw their `float` samples from portable raw-bit generators rather than
from standard-library distributions, whose output is implementation-defined, so a
seeded `float` CMA-ES or ISRES stream does not depend on the standard library
that built it.

## Embedded targets

The board-free arm-none-eabi link gate instantiates the four real-time policies
in both scalars on every commit and fails if a `float` instantiation is missing
from the image. The ESP32 probe builds in both scalars in an ESP-IDF continuous
integration job. On-device allocation counts per step remain an operator capture
on real hardware.

On a NUCLEO-H753ZI (Cortex-M7, arm-none-eabi-gcc 16.2), the probe measured
0.00 allocations per step for `kraft_slsqp`, `nw_sqp`, `filter_nw_sqp` and `lm`
in `double` and in `float`, with the blindness canary and the over-limit
guardrail probe passing. The
[capture record](../mcu/nucleo_h753zi/onchip-result.md#float-capture-2026-10-10)
also holds an earlier failing capture. In it, `float` `nw_sqp` and
`filter_nw_sqp` allocated 0.20 per step because their null step at the solution
copied the QP multipliers to the heap. The same path allocated in `double`; its
windows ended before convergence, so they never reached it. The host
allocation gate now arms an at-solution window after a converged solve for the
three SQP policies in both scalars. The record also lists cycles per step, but
they are not compared across scalars and carry no speed claim.

## What is not provided

- `float` from Python: the bindings stay `double`-only.
- Any statement about `float` speed relative to `double`. Cycle counts captured
  on hardware are recorded as measurements, not compared or gated.
