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
- `bound_violation()` on `step_budget_solver`, `time_budget_solver`,
  `step_and_time_budget_solver` and `stepper`, and
  `step_result::bound_violation`: the largest distance of the iterate outside
  the box. The accessor is NaN for a rejected problem.

### Changed

- `reset(x0)` and `reset_clear(x0)` return `solver_status` instead of `void`,
  on every policy, solver driver and `basic_solver_group`. A start of the wrong
  length, at construction or on reset, ends with `invalid_problem` instead of
  being read out of bounds. So do more constraint rows than a declared
  `constraint_count`, a row or residual count changed between resets, a row
  shape the family cannot store (for example more equality rows than
  variables in the line-search SQP policies), and an ISRES or CMA-ES
  population option the policy cannot store. The status is latched: `step()`
  and `solve()` return it without evaluating the problem until a valid reset.
- A status that describes the returned point (`converged`, `ftol_reached`,
  `xtol_reached`, `stalled`) is reported only when that point is feasible
  within the feasibility tolerance in force (`constraint_tolerance` when set,
  else `feasibility_tolerance`). Feasibility covers both the largest
  constraint-row violation and the distance outside the box. A convergence
  criterion that fires at an infeasible point is not reported, and the solve
  continues. A policy that stops on its own test at an infeasible point ends
  the solve with `objective_stalled`. This holds for `solve()`, `step()`, the
  stepper and the solver group.
- A solve whose status certifies its final iterate returns that iterate.
  Before, an earlier iterate with a lower objective inside the feasibility
  tolerance could be returned under the final iterate's status. Exits that make
  no claim still return the best point seen, ranked feasible first on the
  larger of the row and box violation.
- The KKT residual's complementarity term is the product `max |mu_i c_i|`
  (Nocedal and Wright, Definition 12.1) instead of `max min(|mu_i|, |c_i|)`. It
  applies to the general constraint rows and to the box multipliers of the
  L-BFGS-B, projected Gauss-Newton and SQP solvers. A point with a large
  multiplier and a small nonzero constraint value is no longer certified
  `converged`.
- `nw_sqp`, `filter_nw_sqp`, `filter_slsqp` and `kraft_slsqp` measure
  stationarity, including the reported `gradient_norm`, on the box-projected
  Lagrangian gradient. A minimizer on a bound now ends `converged` instead of
  `stalled` or `max_iterations`.
- The multiplier estimate of `nw_sqp`, `filter_nw_sqp`, `filter_slsqp`,
  `tr_sqp` and `filter_trsqp` treats a coordinate on its bound as a working-set
  row. A KKT point where a bound and general constraint rows are both active
  (HS071) is certified `converged`.
- `tr_sqp` and `filter_trsqp` project a start outside the box onto it before
  evaluating it, at construction and on reset, as MMA and the GCMMA variants
  do.
- `tr_sqp` ends `roundoff_limited` when a rejected trial's actual and predicted
  merit reductions both lie within the merit's rounding error, instead of
  shrinking the radius until `trust_region_step_rejected`. `tr_sqp` and
  `filter_trsqp` treat a trust radius below the relative machine precision of
  the iterate as collapsed, even when `min_trust_radius` is smaller.
- `constraint_violation()` on the solver drivers is the largest row violation
  (L-infinity), the measure `solve_result::constraint_violation` and the
  feasibility tolerance use, instead of the sum (L1).
- `basic_solver_group` reports the status of the member whose point it
  returns, admitted on that member's rows and box, instead of its own verdict
  for whichever member ranked best. A member still running when the group stops
  reports `max_iterations` or `budget_exhausted`, or `objective_stalled` when
  the group's criteria stopped it. `results()` reports members the same way.
- `lbfgsb`, `byrd_lbfgsb`, `bobyqa`, `lm`, `projected_gn`,
  `projected_gradient_gn` and the three CMA-ES variants end with
  `invalid_problem` for a problem that declares equality or inequality rows,
  at construction or on reset and without evaluating it, instead of ignoring
  the rows. `restarting_policy` and `multistart_policy` report the rejection
  of the policy they wrap. A problem with the row interface and zero rows runs
  as before; constrained problems go through `augmented_lagrangian`.
- A solve rejected with `invalid_problem` counts no iteration and no
  evaluation, and reports NaN for `objective_value`, `gradient_norm` and
  `constraint_violation`, as the drivers' accessors do. Before, it could count
  one iteration and report a violation of 0.
- ISRES (all three variants) returns the best point it evaluated within the
  solve's feasibility tolerance, ranked by the solve's own best-iterate rule. A
  collapsed population reports `xtol_reached` instead of `converged` or
  `ftol_reached`, or `objective_stalled` when no evaluated point is within the
  tolerance. ISRES reports a NaN gradient norm and never claims optimality.

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

### Removed

- ISRES's `feasibility_gate` option. ISRES used it as a fixed gate on the
  summed squared constraint violation; its record is now ranked at the solve's
  feasibility tolerance.

### Fixed

- MMA and the GCMMA variants no longer treat an infeasible start as feasible
  when choosing the point to return. A solve that ended without a claim could
  return its start and report a constraint violation of 0.

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
