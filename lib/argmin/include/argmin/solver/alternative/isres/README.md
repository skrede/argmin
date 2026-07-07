# `solver/alternative/isres/` — research variant policies

Three different reproductions of the ISRES (Improved Stochastic Ranking
Evolution Strategy) per-generation step under research-grade citation
discipline. None of these are part of the production API; production
ISRES lives in `solver/isres_policy.h` and is a thin alias to the
empirical winner among the variants here.

Each variant satisfies the `argmin::nlp_solver` concept and can be
plugged into `step_budget_solver` interchangeably for benchmarking.

| Variant | Mechanism | Reference | Empirical role |
|---|---|---|---|
| `nlopt_faithful_policy.h` | NLopt 2.10.0 `isres.c` reproduction. Top-mu DE-style differential variation `x[rk] += gamma * (x0[0] - x0[k+1])` with physical-slot-0 anchored snapshot; per-mutation sigma upper clamp `(ub-lb)/sqrt(n)`; alpha-smoothing `sigma_out = sigma_parent + alpha * (sigma_new - sigma_parent)`; resample-on-bound with bounded retry; L2-squared violations. | Runarsson & Yao (2005), IEEE Trans. SMC-C 35(2):233-243; NLopt 2.10.0 `isres.c` (Steven G. Johnson 2009). | Current production alias. Stable cross-implementation reference; matches NLopt operator semantics line-for-line so behaviour is reproducible against the upstream control row. Loses to `runarsson_yao_paper` on solution quality across the multimodal Rastrigin / Schwefel cells and the constrained HS subset at the standard 100-trial seed protocol. |
| `original_argmin_policy.h` | Frozen baseline. Pre-rewrite ISRES preserved verbatim: alpha-pull-to-best operator (NOT differential variation), `j % mu` cyclic parent indexing, hard-clip bounds via `cwiseMax(lower).cwiseMin(upper)`, L1 violation aggregation, no sigma upper clamp, no sigma-smoothing remap, dead `[mu, lambda)` cyclic fill retained. | Pre-rewrite argmin implementation; preserved as research artifact for empirical comparison. | Frozen research artifact. Faster wall-time than the corrected variants on every cell because the pull-to-best operator collapses the population aggressively, but produces incorrect optima on Schwefel (terminates at the local saddle ~118) and is dominated by `runarsson_yao_paper` on solution quality across the constrained HS subset. Retained verbatim so the empirical paper can credibly demonstrate the algorithmic mismapping the rewrite addresses. |
| `runarsson_yao_paper_policy.h` | Paper-faithful. BEST-anchored DE recombination `x_new = x_1 + gamma * (x_i - x_1)` with rank-0-by-fitness snapshot; sigma upper clamp + alpha-smoothing inherited from NLopt with explicit citation gap on numeric defaults; L2-squared violations; resample-on-bound. | Runarsson & Yao (2005), IEEE Trans. SMC-C 35(2):233-243; ISRES+ Liu et al. 2023 Bioinformatics 39(7):btad403 Table 2 (numeric default attribution gap). | Empirically the best-quality variant on the comparison test set: reaches the cited optimum (or a comparable-quality iterate) on HS024, HS035, HS076, Rastrigin 2D, Rastrigin 5D, Schwefel 2D, and the bounds-degenerate cell at the cost of 2-4x wall-time vs `nlopt_faithful`. Production-alias swap candidate; not locked in by default to keep the cross-implementation NLopt reference operator semantics in the production path. |

See `benchmarks/micro_isres.cpp` for the comparison harness; the
"Empirical role" column above summarises the post-bench observations
and the production-alias selection.

## Lifecycle

These variants are compiled, tested, and benchmarked alongside the
production code. They are not subject to the same API-stability
promises as `solver/` types but are kept buildable so the comparison
can be re-run in any future commit. If a variant is dropped, the
deletion should be a single atomic commit referencing the empirical
decision.
