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
- `solver_status::non_finite_evaluation` and `solver_status::inner_solve_failed`
  (Python `SolverStatus.non_finite_evaluation` and
  `SolverStatus.inner_solve_failed`). The first reports an objective,
  constraint, gradient or Jacobian that evaluated non-finite at the start point,
  or later at a point the method cannot recover from; the second reports an
  inner solve (main QP, dual subproblem, trust-region linear program,
  restoration phase or augmented Lagrangian inner solve) that failed with no
  recovery left. Neither is a claim about the final iterate. Both are appended
  to the enum, so existing values keep their numbering, but every exhaustive
  `switch` over `solver_status` must handle them.
- `line_search_options::relative_value_noise`, an opt-in approximate Wolfe band
  (Hager and Zhang 2005): a trial whose value is at most
  `phi(0) + relative_value_noise * |phi(0)|` is accepted on the strong-Wolfe
  curvature condition alone. It defaults to 0, which disables the band;
  Python's `LineSearchOptions` reports it.
- `restarting_policy::options_type::max_restarts`: when set, the solve ends
  with the best run after that many restarts; unset, restarts continue until
  the iteration budget ends the solve.
- COBYLA's policy state carries `clamped_evaluations`, the number of evaluation
  points its safety clamp moved into the box; Python reports it as
  `CobylaSolver.clamped_evaluations`.
- `bound_violation()` on `step_budget_solver`, `time_budget_solver`,
  `step_and_time_budget_solver` and `stepper`, and
  `step_result::bound_violation`: the largest distance of the iterate outside
  the box. The accessor is NaN for a rejected problem.

### Changed

- Breaking: a terminal status stays until `reset(x0)` or `reset_clear(x0)`.
  `step()` checks `abort()` on entry and ends `aborted` without running the
  policy, and from any terminal status it returns that status with an
  unevaluated step result (NaN values, a null step, no evaluations).
  `step_n()` and `solve()` continue only from `budget_exhausted` and
  `time_limit_reached`, the statuses meaning the previous call's own iteration
  or time budget ran out; from any other terminal status, `max_iterations`
  included, they take no step and report that status. A second `solve()` after
  a claim used to step on from the claimed point.
- Breaking: `stepper::converged()` is true only for a status that claims the
  final iterate (`converged`, `ftol_reached`, `xtol_reached`, `stalled`), so it
  is false after `diverged`, `invalid_problem` and every other failure. The new
  `stepper::finished()` is true for any status other than `running`, the old
  meaning of `converged()`: a `while(!st.converged()) st.step();` loop becomes
  `while(!st.finished()) st.step();`.
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
- `basic_solver_group` retires a member once that member's own status, judged
  by its own convergence criteria, is not `running`, and never steps it again.
  The group ends when every member has retired or its budget is spent;
  `step_n()` no longer judges steps from different members against one shared
  convergence object, so its `opts` argument supplies only `max_iterations`.
  The group reports the status of the member whose point it returns, admitted
  on that member's rows and box, instead of its own verdict for whichever
  member ranked best; a member still running when the budget ends reports
  `max_iterations` or `budget_exhausted`. A `step()` after every member retired
  returns that status with an unevaluated step result instead of
  `budget_exhausted`. `results()` reports members the same way.
- `solve_result::gradient_norm` is the norm reported by the step that produced
  the returned point, as `x`, `objective_value` and `constraint_violation` are,
  instead of the last step's. It is NaN when the returned point is the start
  point (a solve that aborts, has a zero budget or is called on a terminal status
  before its first step, or whose steps never rank above the start), and no
  gradient is evaluated to fill it. Before, a solve that ended before its first
  step reported 0. A solver group reports the gradient norm of the member whose
  point it returns.
- `solve_result::function_evaluations` and `step_result::evaluations` of
  `lbfgsb`, `byrd_lbfgsb`, `lm`, `projected_gn`, `projected_gradient_gn` and
  the augmented Lagrangian count every objective call the solver made since
  construction or the last reset: the start-point evaluations of construction
  and reset, the one each `solve()` or `step_n()` makes at its start point, and,
  for the augmented Lagrangian, every call of its inner solver. A least-squares
  solver counts each residual evaluation, the ones of a finite-difference
  Jacobian included. Before, `lbfgsb` also counted each strong-Wolfe slope
  evaluation, the other five reported one evaluation per step, and none counted
  the start-point calls.
- The same counts hold for `kraft_slsqp`, `nw_sqp`, `filter_slsqp`,
  `filter_nw_sqp`, `tr_sqp`, `filter_trsqp`, `mma`, `ccsa_quadratic` and the
  three GCMMA alternates: every line-search trial, second-order correction,
  restoration re-evaluation and conservativity re-evaluation is counted.
  Before, they reported one evaluation per step and did not count the
  start-point calls.
- The same counts hold for `bobyqa`, `cobyla`, the three CMA-ES and three ISRES
  variants, `multistart` and the restarting decorator: every interpolation-set,
  rescue and generation evaluation is counted, repair and penalty re-evaluations
  included, and a decorator counts every evaluation of every start and restart
  of the policy it wraps. Before, the population methods reported one
  evaluation per generation, `cobyla` missed its initial simplex, both
  derivative-free solvers missed the start-point call of each solve, and
  `multistart` summed its inner solver's per-step counts.
- `solve()` and `step_n()` on a status they cannot continue from evaluate
  nothing and return the result of the call that ended the solve: the same `x`,
  objective, constraint violation and gradient norm. Before, they evaluated the
  objective at the current iterate and reported it with a NaN gradient norm.
  Without such a result, as when `step()` reached the status, they report the
  current iterate as before.
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

- The SQP solvers (`kraft_slsqp`, `filter_slsqp`, `nw_sqp`, `filter_nw_sqp`,
  `tr_sqp`, `filter_trsqp`) report `non_finite_evaluation` for an objective,
  constraint, gradient or Jacobian that is non-finite at the start point, on
  construction and on `reset` / `reset_clear`, also when wrapped in
  `multistart_policy`. A step that leaves them at an iterate with a non-finite
  value ends the solve with `non_finite_evaluation`, and that iterate is never
  returned as the best point. Inside `multistart_policy` such a step ends only
  that member, a later member with a non-finite start ends without a step,
  neither is returned over a finite member, and the next member starts. If
  every member ends non-finite, the solve ends with `non_finite_evaluation` at
  its best finite evaluated point.
- The moving-asymptote solvers (`mma`, `gcmma`, `ccsa_quadratic` and the GCMMA
  alternates) and the least-squares solvers (`lm`, `projected_gn`,
  `projected_gradient_gn`) report `non_finite_evaluation` for an objective,
  constraint, residual, gradient or Jacobian that is non-finite at the start
  point, on construction and on `reset` / `reset_clear`.
- These solvers treat a non-finite trial as a failed trial and never commit it:
  `mma` takes a null step and contracts its asymptotes; `gcmma`,
  `ccsa_quadratic` and the alternates treat it as a non-conservative trial, also
  when only the gradient or constraint Jacobian at a conservative trial is
  non-finite; `lm`, `projected_gn` and `projected_gradient_gn` reject it, also
  for a non-finite Jacobian at an accepted trial, and raise their damping or
  shrink their radius. An outer step whose conservativity loop ends on a
  non-finite trial is a null step that keeps the grown penalty or the shrunk
  move-limit window for the next step. A solve ends with
  `non_finite_evaluation` only at the solver's limit: for `mma`, a failed trial
  made with the asymptotes at their minimum distance; for `gcmma`,
  `ccsa_quadratic` and the alternates, a trial within the iterate's resolution
  that is non-finite or follows a non-finite rejection, so such a trial ends the
  solve even when it is finite; for `lm`, `projected_gn` and
  `projected_gradient_gn`, a failed trial at the damping ceiling or radius
  floor, or a damped step that no longer moves the iterate. `lm` no longer ends
  `diverged` on its first non-finite trial.
- `lbfgsb`, `byrd_lbfgsb`, the augmented Lagrangian, `bobyqa`, `cobyla` and the
  CMA-ES variants report `non_finite_evaluation` for an objective, gradient or
  constraint that is non-finite at the start point, on construction and on
  `reset` / `reset_clear`; the augmented Lagrangian checks its objective and
  rows there and reports a non-finite constraint Jacobian at the start on its
  first step, when its inner solve meets it. ISRES, which never evaluates the
  start, does so when no individual of its first population evaluates finite.
- These solvers treat a non-finite trial as a failed trial and never commit or
  return it, except in the `bobyqa` case below. `lbfgsb` and `byrd_lbfgsb`
  evaluate the gradient at an accepted trial before the commit and reject the
  trial when it is non-finite; a line search that fails in both memories after
  meeting a non-finite value ends `non_finite_evaluation` instead of
  `line_search_failed`. `bobyqa` and `cobyla` store a non-finite evaluation as a
  large finite barrier value (PRIMA's moderated extreme barrier, 2^100 or about
  1.3e30 in double precision and 2^64 in single), so the trial ranks worst and
  the trust region shrinks. `cobyla` never lets such a point become the
  incumbent. `bobyqa` can rank one first only on an objective whose finite
  values exceed the barrier; it may then return that point with the barrier as
  its objective value, and the run makes no claim about it. Neither claims a
  solution while such a value can still shape its model: while either holds a
  non-finite flag, a status that would certify the iterate (`converged`,
  `ftol_reached`, `xtol_reached` or `stalled`) is reported as
  `non_finite_evaluation`, whichever stopping test accepted the iterate, the
  solver's own or a convergence criterion such as a caller-set tolerance; a
  budget limit such as `max_iterations` is reported as is. `bobyqa` sets its
  flag on any non-finite evaluation, its initial interpolation points' included,
  and holds it until it rebuilds the interpolation model from finite values (a
  rescue or a restart), so a run that recovers from a single non-finite value
  without such a rebuild ends `non_finite_evaluation` even at a good point.
  `cobyla` flags each simplex vertex that evaluated non-finite and holds a flag
  while a trial at its current radius was non-finite or its linear models are
  built on a flagged vertex. Both track this with flags rather than reading it
  from the stored value, so a finite value above the barrier is never taken for
  a non-finite one. The CMA-ES variants and ISRES rank a non-finite individual
  after every finite one and never hold it as the best point, and a generation
  with no finite individual ends `non_finite_evaluation` (CMA-ES before its
  distribution moves; a search distribution that is itself non-finite still ends
  `diverged`, before it is sampled). The augmented Lagrangian never commits an
  outer iterate whose objective or rows are non-finite: it tightens the penalty
  as after an infeasible outer iterate, and ends `non_finite_evaluation` once
  the penalty is at its floor. It reports an inner solve that ended at a
  non-finite region without moving as a null step instead of `stalled`, and ends
  `non_finite_evaluation` when that happens with the penalty at its floor.
- ISRES no longer reads a NaN constraint row as satisfied: its violation is NaN,
  so the individual ranks worst.
- `stall_tolerance_criterion` counts only steps that move the iterate: a null
  step neither fills its window nor restarts it, so a window of K spans K moves.
  A run of null steps, such as a conservative moving-asymptote solver's against
  a non-finite region, no longer ends a solve `stalled` at a point that is not
  stationary. A run of only null steps, such as `nw_sqp`'s on a zero QP
  direction, `filter_nw_sqp`'s at its reset cap or the augmented Lagrangian's
  while its inner solve runs in chunks, no longer ends a solve `stalled`; the
  solve goes on to the solver's own limit or to `max_iterations`.
- `tr_sqp` rejects a trial whose gain ratio is not finite and shrinks its trust
  radius, instead of accepting a step into a NaN objective. The filter of
  `filter_trsqp`, `filter_slsqp` and `filter_nw_sqp` never accepts a non-finite
  objective or violation, and `filter_slsqp` fires no second-order correction
  after a non-finite full step.
- The point a solve returns ranks a non-finite objective or violation after
  every finite one, so a NaN at the solve's own evaluation of the start no
  longer pins the returned point there, and a trial evaluating to -inf never
  becomes it. The same order ranks ISRES's best evaluated point, the augmented
  Lagrangian's best inner iterate, the best member of `basic_solver_group`, and
  the member `multistart_policy` and `restarting_policy` return.
- `restarting_policy` `reset` and `reset_clear` start a solve as construction
  does: the caller's inner options (a restart's escalated population and
  per-restart seed are dropped), the base seed, no restarts, and the new start
  as the restart mean of an unbounded problem. A warm `reset` no longer keeps
  the escalated population.
- A `multistart_policy` `reset` starts its restart count over, so the solve
  after it runs every start, as one after `reset_clear` or construction does.
- `kraft_slsqp_policy` and `filter_slsqp_policy` restart their
  `line_search_calls` counter on `reset`, as its documentation states; a warm
  reset previously kept counting from the run before.
- `multistart_policy` and `restarting_policy` decide when they are done and
  report what their runs did. Each run is judged by the inner policy's own
  status, else by the decorator's copy of the solve's convergence settings, so
  the solve's own criteria no longer end a multistart or restarting solve on
  the first run's verdict. `multistart_policy` starts every run it promises
  (1 + `max_restarts`); `restarting_policy` ends on a `converged`,
  `ftol_reached` or `xtol_reached` claim or on `invalid_problem` and restarts
  on every other end of a run, including its own stagnation count. At the last
  run the solve ends with the best run, ranked feasibility first, and reports
  that run's point, objective, constraint violation, gradient norm and status.
  A step that starts a new run no longer reports a placeholder gradient norm
  of 0 or 1, and restart exhaustion no longer reports `budget_exhausted`. A
  multistart over a fixed-dimension problem without a box restarts from its
  best run's point instead of a point drawn from unset bounds.

- A reset CMA-ES or ISRES solver behaves like a new one. `reset` and `reset_clear`
  reseed from `options.seed` (or from `std::random_device` when it is unset),
  rebuild the initial covariance scaling and the population, including the
  bounded-problem population floor, and ISRES evaluates the population it
  redraws. A fixed seed now reproduces a fresh run after a reset instead of
  continuing the stream. The CMA-ES `reset_clear` keeps the start it is handed
  as the mean; the restart decorator draws each restart's mean in the box and
  derives a per-restart seed from the base seed, so a seeded restarting solve
  stays reproducible. An unseeded `restarting_policy` draws its base seed from
  `std::random_device` on init and every reset; it previously pinned an unset
  inner seed to 1 and replayed the same run.

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
- Convergence criteria no longer carry history from one solve into the next.
  `stall_tolerance_criterion` and `convergence_policy` gain a `reset()` that
  clears the stall window and the best-seen feasible objective. An explicit
  `solve(opts)` or `step_n(budget, opts)` resets `opts.convergence` unless the
  call resumes after `budget_exhausted` or `time_limit_reached`, and `reset`
  and `reset_clear` reset the solver's stored criteria. Before, a second solve
  on the same options object could stall at once on the first solve's best
  objective, and a solver reset kept the stall history.
- The defaulted `solve()` and `step_n(budget)` of the three budget drivers no
  longer copy the convergence object on every call, and the MMA, CCSA
  quadratic and GCMMA dual solves no longer build a convergence object with a
  512-entry stall buffer on every outer step. The defaulted paths run on the
  solver's stored criteria, so their stall history continues across
  `step_n` chunks as it does across `step()` calls.
- A stall window above the criterion's 512-entry buffer
  (`stall_tolerance_criterion::capacity`) reports `invalid_problem` instead of
  being shortened to 512. A solver whose stored window is too large reports it
  from construction, `reset` and `reset_clear`, and an explicit `solve(opts)`
  or `step_n(budget, opts)` with such options returns `invalid_problem`
  without stepping.
- A policy hint no longer overrides a caller's convergence setting on any solve
  path. Breaking: `stall_tolerance_criterion::window` and
  `track_best_seen_feasible` are `std::optional` (empty when the caller did not
  set them), and `window_in_force()`, `threshold_in_force()` and
  `best_seen_feasible_in_force()` report the value used: the caller's, then the
  current policy's hint, then the library default
  (`convergence_defaults::stall_window` is 50). The hints apply at construction
  and at every explicit `solve(opts)` and `step_n(budget, opts)` without being
  written into the caller's fields, so a reused options object carries no hint
  from another policy. Python: `SolverOptions.stall_window` is `None` when unset
  and `SolverOptions.stall_window_in_force` reports the window used.
- Breaking: `augmented_lagrangian_policy`'s `constraint_tolerance` option is
  `std::optional`. The outer feasibility test uses the solver's
  `constraint_tolerance` when set, then this option, then the default (1e-6).
- The strong Wolfe line search counts every slope evaluation against its
  `max_iterations` budget, in the bracket, the zoom and the approximate Wolfe
  test. It used to check the budget only before an objective evaluation, so a
  search could make one evaluation past its budget.
- The ISRES policies hint a stall window of 500 generations instead of 200,
  which stopped runs on plateaus of up to 232 generations that later improved.
  The augmented Lagrangian, BOBYQA and CMA-ES policies hint the library default
  of 50 instead of 100 and 200. A hinted window acts only when the caller sets a
  stall threshold.

### Notes

- Every policy plugs into `step_budget_solver<Policy>` and exposes a
  uniform `step()` / `step_n(budget)` / `solve()` interface.
- Several policies are certified allocation-free at a fixed compile-time
  dimension under the allocation-counting CI gate; see
  [docs/rt-safety-matrix.md](docs/rt-safety-matrix.md) for the
  per-solver breakdown.
- Known limitations at HEAD are tracked in the README's Status section
  and the solver-selection guide.
