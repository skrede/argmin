# Changelog

All notable changes to argmin are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it reaches a stable `v1.0.0`.

argmin is in **public preview** (the `v0.3.x` line): the API surface and
policy lineup are still stabilizing, so minor versions may carry breaking
changes. Structured, per-release notes begin below; for fine-grained
history before the first tagged release, see the git log.

## [Unreleased]

### Solvers

Policies available at HEAD, grouped by problem class:

- **Unconstrained / box:** `lbfgsb`, `byrd_lbfgsb`
- **Equality + inequality constrained:** `kraft_slsqp`, `filter_slsqp`,
  `tr_sqp`, `filter_trsqp`, `nw_sqp`, `filter_nw_sqp`
- **Separable convex approximation:** `mma`, `gcmma`, `ccsa_quadratic`
- **Least-squares:** `lm`, `projected_gn`, `projected_gradient_gn`
- **Derivative-free:** `bobyqa`, `cobyla` (and `multistart` wrapping them)
- **Global / stochastic:** `cmaes`, restarting CMA-ES, `isres`
- **Meta:** `augmented_lagrangian` (wraps any unconstrained inner policy)

### Added

- Python: `FilterSlsqpSolver` (with `FilterSlsqpOptions` and
  `RestorationStrategy`) and `TrSqpSolver` (with `TrSqpOptions`, accurate mode),
  also available as the `filter_slsqp` and `tr_sqp` methods of
  `ConstrainedTestProblem.solve_native`. See
  [docs/python-bindings.md](docs/python-bindings.md) for the benchmark evidence
  behind both.
- `solver_status::line_search_failed` (Python `SolverStatus.line_search_failed`),
  reported by the L-BFGS-B policies when a line-search step and its retry from a
  reset curvature memory are both rejected.
- `line_search_options::relative_value_noise`, an opt-in approximate Wolfe band
  (Hager and Zhang 2005): a trial whose value is at most
  `phi(0) + relative_value_noise * |phi(0)|` is accepted on the strong-Wolfe
  curvature condition alone. It defaults to 0, which disables the band;
  Python's `LineSearchOptions` reports it.
- COBYLA's policy state carries `clamped_evaluations`, the number of evaluation
  points its safety clamp moved into the box; Python reports it as
  `CobylaSolver.clamped_evaluations`.

### Changed

- Python names follow the C++ types they bind, with no aliases for the old
  names: `SlsqpSolver` is `KraftSlsqpSolver`, `SlsqpOptions` is
  `KraftSlsqpOptions`, `QpSubproblemOptions` is `QpOptions`, `QpStatus` is
  `QpSolveStatus`, `CmaesRestart` is `CmaesRestartStrategy`, and the compiled
  Kraft method is selected as `kraft_slsqp` instead of `slsqp`.
- Python: BOBYQA and CMA-ES, with their own option and enum types, moved behind
  `import argmin.experimental`. Each gated solver carries `maturity` and
  `known_defect` attributes; their compiled-problem solves moved to
  `argmin.experimental.solve_native`, and the top-level `solve_native` refuses
  them with `invalid_problem`. `CobylaSolver` and `CobylaOptions` stay in the
  top-level `argmin` namespace, and `ConstrainedTestProblem.solve_native`
  accepts `cobyla`.
- COBYLA's automatic initial trust radius is capped at half the smallest finite
  bound range (Powell's rule, as BOBYQA already applies); a radius the caller
  supplies is not capped.
- The CMA-ES detection options are `cmaes_detection_options`, in
  `argmin/options/cmaes_detection_options.h` (formerly `cmaes_options`).
- Some L-BFGS-B solves that ended `roundoff_limited` near a solution now end
  `line_search_failed`.
- Every policy, the QP and line-search options, the convergence criteria and
  `default_convergence` take the scalar type as their first template argument
  (`float` or `double`, defaulting to `double`), for example
  `nw_sqp_policy<double, NV>` and `gradient_tolerance_criterion<>`.
- The QP options' `max_iterations` is `std::uint32_t`.
- `float` is supported per algorithm family, each gated against `double`; see
  [docs/float-support.md](docs/float-support.md).

### Fixed

- COBYLA never evaluates the objective or the constraints outside the box. Two
  sign errors in its bound repair, in the initial simplex step and in the
  clamp of a trial step that undershoots a lower bound, are corrected, and a
  counted clamp moves any remaining evaluation point into the box.
- A COBYLA box with a fixed or inverted coordinate (a lower bound not below its
  upper bound) ends with `invalid_problem` instead of a degenerate solve; from
  Python, `CobylaSolver` raises `ArgminError` with `invalid_bounds` at
  construction.
- Both L-BFGS-B policies no longer commit a failed line-search step: a
  non-finite trial, a non-positive step or a search without sufficient decrease
  leaves the iterate in place, the step is retried once from a reset curvature
  memory, and a second rejection ends the solve with `line_search_failed`.
- The constrained SQP solvers apply their bounds. `KraftSlsqpSolver` accepted
  `lower_bounds` and `upper_bounds` from Python but never passed them to the
  policy, so a solve could report `converged` outside the box.
- The MMA, CCSA quadratic and GCMMA dual solves accept roundoff-level
  line-search steps through the approximate Wolfe band, where the remaining
  decrease near the dual optimum falls below the roundoff of the dual value.
- GCMMA's `rho_wval` variant commits a trial that sits at the current iterate
  within roundoff, when it does not raise the constraint violation past its
  slack, instead of null-stepping until the iteration cap.
- Python count options (`max_iterations`, `stall_window`, `policy_stall_window`,
  CMA-ES `population_size` and the `step_n` budget) too large for their
  unsigned C++ field raise `ArgminError` with `dimension_mismatch` instead of
  wrapping; a `stall_window` of 65545 used to become 9.
- ISRES no longer reports `ftol_reached` after its first generation when a
  variable has an infinite bound: the sigma-collapse stop measures an unbounded
  dimension against the same reference range its initial step size uses.
- `ccsa_quadratic` in `float` keeps its move box resolvable: the minimum move
  limit is 3e-4 of the variable range instead of 1e-8, which is below `float`
  precision at unit scale.

### Notes

- Every policy plugs into `step_budget_solver<Policy>` and exposes a
  uniform `step()` / `step_n(budget)` / `solve()` interface.
- Several policies are certified allocation-free at a fixed compile-time
  dimension under the allocation-counting CI gate; see
  [docs/rt-safety-matrix.md](docs/rt-safety-matrix.md) for the
  per-solver breakdown.
- Known limitations at HEAD are tracked in the README's Status section
  and the solver-selection guide.
