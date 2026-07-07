# `solver/alternative/cmaes/` — research variant policies

Three different boundary-handling mechanisms for CMA-ES on bounded
problems. None of these are part of the production API; production
CMA-ES lives in `solver/cmaes_policy.h` and is a thin alias to the
empirical winner among the variants here.

Each variant satisfies the same `argmin::nlp_solver` concept as the
production policies and can be plugged into `step_budget_solver`
interchangeably for benchmarking.

| Variant | Mechanism | Reference | Empirical role |
|---|---|---|---|
| `repair_l2_penalty_policy.h` | Clip x_i to [lower, upper]; add Σd² penalty to objective; CMA-ES sees repaired x. | Hansen (2009) §2 boundary handling tutorial. | Loses on the boundary-active cells (schwefel_2, griewank_2): the L2 penalty introduces non-physical gradient signal at the box boundary, biasing the search away from boundary-active optima. Preserved as the prior production behavior for empirical reproducibility. |
| `pwq_reparameterization_policy.h` | Piecewise-quadratic invertible geno→pheno transform; CMA-ES sees unbounded geno coords; objective evaluated at pheno. No penalty term. | libcmaes `pwq_bound_strategy.cc:35-125` (Steven G. Johnson adaptation of MATLAB CMA-ES `boundary_transformation.c`). | **Production winner.** The only variant that ever reaches the `libcmaes_ipop` optimum on `schwefel_2` (2.5e-05) at fixed seed; the lowest 5-seed median on `griewank_2`; tied on `rastrigin_*` and `ackley_*`. By sum of log10 median objective across the 6 bounded global cells: pwq −14.80, no_repair −14.58, l2 −14.34. |
| `no_repair_adaptive_penalty_policy.h` | Evaluate at unrepaired x; penalty Σ w_i (x_i − clip_i)² with adaptive w_i (EMA of rank-mu diagonal, half-life c_c). | Hansen (2009) §3 adaptive boundary penalty. | Smallest objective on `ackley_*` (where the optimum is at the origin and boundary handling is not load-bearing); does not reach the libcmaes_ipop optimum on `schwefel_2`. The two-evals-per-offspring cost shows up on the wall column without a corresponding objective win on the boundary-active cells. |

See the A/B verdict at `.planning/phases/34.2-cmaes-convergence-quality-vs-libcmaes/34.2-03-AB-RESULT.md` and the persistent comparison harness at `benchmarks/micro_cmaes.cpp`.

## Lifecycle

These variants are compiled, tested, and benchmarked alongside the
production code. They are not subject to the same API-stability
promises as `solver/` types but are kept buildable so the comparison
can be re-run in any future commit. If a variant is dropped, the
deletion should be a single atomic commit referencing the empirical
decision.
