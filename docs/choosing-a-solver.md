# Choosing a solver

nablapp ships 18 policies. The right one depends on three things about your problem: whether gradients are available, what kind of constraints there are, and how large `n` is. This page maps problem class to policy with one-line rationale and known caveats.

If you only remember one rule: **`lbfgsb` for unconstrained / box-constrained, `kraft_slsqp` for general constrained, `cmaes` for global / non-smooth.** Everything else is for cases where those don't fit.

## Decision table

| Your problem | Recommended policy | Notes |
|---|---|---|
| Smooth, unconstrained, gradients available | `lbfgsb` | Workhorse. Quasi-Newton with limited memory; scales to large `n`. |
| Smooth, unconstrained, very small `n` (≤ 6) | `lbfgsb` or `byrd_lbfgsb` | Either works; `byrd_lbfgsb` adds Byrd 1995 free-variable shortcut. |
| Smooth, **box-constrained**, gradients available | `lbfgsb` | Native box bounds via Cauchy point + subspace minimisation. |
| Least-squares (sum of squares of residuals), unconstrained | `lm` | Levenberg-Marquardt; exploits the Gauss-Newton structure, much faster than treating it as generic unconstrained. |
| Least-squares, box-constrained | `projected_gn` (J explicitly) or `projected_gradient_gn` (J via gradient) | Active-set Gauss-Newton with bound projection; the standard choice for inverse kinematics. |
| Derivative-free, smooth, small `n` (≤ 20) | `bobyqa` | Powell's quadratic-model trust region. Local solver — finds the nearest minimum. |
| Derivative-free, multimodal, small `n` | `multistart_bobyqa` | Wraps `bobyqa` with random restarts. Modest improvement, not a global solver. |
| Equality + inequality constrained, gradients available | `kraft_slsqp` | Faithful Kraft 1988 SLSQP with native box bounds. The default constrained solver. |
| Same as above, **filter line-search** | `filter_slsqp` | Fletcher-Leyffer 2002 filter acceptance instead of L1 merit. Sometimes converges where `kraft_slsqp` stalls; usually a few more iterations. |
| Constrained problem with separable structure (topology / structural opt) | `mma`, `gcmma`, or `ccsa_quadratic` | Method of Moving Asymptotes (Svanberg 1987 / 2002). Best when the objective and constraints are well-approximated by separable convex functions in `1/(U-x)` and `1/(x-L)`. |
| Constrained problem, derivative-free | `cobyla` (with caveats) or `augmented_lagrangian` wrapping `bobyqa` | See "Known limitations" below before reaching for COBYLA. |
| General constrained, want to use any unconstrained solver as the inner | `augmented_lagrangian` | Meta-policy. Wraps any unconstrained or box-constrained policy and handles equality + inequality via penalty + multiplier updates. |
| Bound-constrained, multimodal, no gradients | `cmaes` or `restarting_cmaes` | Covariance Matrix Adaptation Evolution Strategy. `restarting_cmaes` adds Hansen-Auger IPOP restarts for harder landscapes. |
| Bound-constrained, multimodal, **with constraints** | `isres` | Runarsson-Yao 2005 stochastic ranking ES. Handles inequality constraints via stochastic ranking. |
| Local solver but you want random restarts | `multistart_policy<InnerPolicy>` | Wraps any local policy. |
| CMA-ES with restart strategy | `restarting_policy<cmaes_policy<>>` | The IPOP-CMA-ES restart wrapper. |

## Known limitations at the current HEAD

Be aware of these before adopting a solver into a downstream pipeline:

### `nw_sqp` and `filter_nw_sqp` on HS071-class problems

Both N&W-flavour SQP variants can accept an iter-0 step that satisfies the *linearised* inequality but nonlinearly violates it by a large margin, then park at that infeasible point with `status = max_iterations` and a final objective lower than the feasible optimum. Reproduces on HS071 (`f = 13.45` returned vs `f* = 17.014`).

**Workaround:** use `kraft_slsqp` or `filter_slsqp` for the same problem class. Both reach `f* ± 1e-8`. The N&W variants exist primarily as references for ongoing comparison work and should not be used for production solves at this time. Tracked as `SEED-015`.

### COBYLA on linear-equality reformulations of unconstrained QPs

Phase 33 hotfixed COBYLA's adaptive `parmu` parameter, closing the silent-wrong-optimum bug on HS024. However, several other Powell-faithfulness gaps remain (rho contraction, simplex denormalisation, axis-aligned geometry steps, lax conditioning threshold). On HS048 / HS050 / HS051 — unconstrained quadratics posed with linear equality constraints — COBYLA returns `f` orders of magnitude away from the true optimum (`f = 84 / 7516 / 7.7` against `f* = 0`).

**Workaround:** for problems with explicit equality constraints, prefer `kraft_slsqp`, `filter_slsqp`, or `augmented_lagrangian` wrapping `bobyqa`. A Powell-faithfulness rewrite of COBYLA is scheduled for v0.3.x.

### CMA-ES on Ackley / Schwefel

`cmaes` and `restarting_cmaes` converge to local minima rather than the global optimum on `ackley_2`, `ackley_10`, and `schwefel_2`. The local-minimum escape is weaker than the Hansen-Auger active-CMA reference. On Rastrigin (2-D and 10-D) and Griewank (2-D) the median seed reaches machine zero; the gap is concentrated on landscapes with very flat basins.

**Workaround:** for those specific landscapes, run with multiple seeds and take the best, or use `restarting_cmaes` which improves the success rate on Rastrigin / Griewank but does not close the Ackley / Schwefel gap. An active-CMA variant is in the v0.3.x scope.

### MMA / GCMMA conjoined-gate FAILs

MMA and GCMMA reach the correct objective on the cells they were designed for (HS076 within 1e-3 of the published reference value), but fail the conjoined "objective AND iteration-count" gate on HS043 and a handful of related cells when the iteration budget is tightened against the published reference budget. This is a convergence-speed issue, not a wrong-answer issue — the solvers reach the right objective; they just take more iterations than the reference. Ongoing, tracked under `SEED-009` (cv-aware descent rejection).

## How to choose between near-equivalent options

**`lbfgsb` vs `byrd_lbfgsb`.** Both implement L-BFGS-B. Default to `lbfgsb`; switch to `byrd_lbfgsb` only if profiling shows the GCP + subspace path dominates and most of your iterations have all-free or all-active variables (Byrd 1995's fast path optimises that case).

**`kraft_slsqp` vs `filter_slsqp`.** Default to `kraft_slsqp`. Switch to `filter_slsqp` if you observe stalls at infeasible points or oscillation under the L1 merit function — Fletcher-Leyffer's filter accepts steps that the L1 merit would reject when one of objective or constraint violation improves.

**`mma` vs `gcmma` vs `ccsa_quadratic`.** Use `mma` for Svanberg 1987 baseline. Use `gcmma` (globally convergent MMA, Svanberg 2002) when the inner subproblem must be conservative — typically when the objective has very different curvature than the asymptote approximation predicts. `ccsa_quadratic` swaps the rational MMA approximation for a quadratic; rarely the right call but kept for parity with reference implementations.

**`cmaes` vs `restarting_cmaes`.** Use `restarting_cmaes` when you don't know whether the global optimum is in the initial basin. The IPOP restart doubles the population on each restart up to a cap, dramatically improving the chance of escaping local minima, at the cost of a 2-10x increase in function evaluations.

**`augmented_lagrangian` vs `kraft_slsqp` for general constraints.** `kraft_slsqp` is faster when gradients are available and accurate. `augmented_lagrangian` is what you want when (a) you want to use a derivative-free inner like `bobyqa`, (b) gradients are noisy, or (c) you want to plug in an inner you've already characterised (the meta-policy is parameterised on the inner type).

## Compile-time vs dynamic dimensions

Every policy template takes an `N` parameter that propagates to inner Eigen types. Pass a concrete `int` for fixed dimensions (`lbfgsb_policy<2>` for a 2-D problem) and `nablapp::dynamic_dimension` (the default) for runtime-sized problems. Fixed dimensions enable stack allocation, more aggressive inlining, and eliminate heap traffic in the hot path — the difference is meaningful for small `n` in tight loops (IK, MPC).

```cpp
basic_solver<lbfgsb_policy<2>, 2, rosenbrock<double, 2>> fast{problem, x0, opts};
basic_solver<lbfgsb_policy<>>                            generic{problem, x0, opts};
```
