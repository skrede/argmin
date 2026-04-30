# `solver/alternative/isres/` — research variant policies

Three different reproductions of the ISRES (Improved Stochastic Ranking
Evolution Strategy) per-generation step under research-grade citation
discipline. None of these are part of the production API; production
ISRES lives in `solver/isres_policy.h` and is a thin alias to the
empirical winner among the variants here.

Each variant satisfies the `nablapp::nlp_solver` concept and can be
plugged into `basic_solver` interchangeably for benchmarking.

| Variant | Mechanism | Reference | Empirical role |
|---|---|---|---|
| `nlopt_faithful_policy.h` | NLopt 2.10.0 `isres.c` reproduction. Top-mu DE-style differential variation `x[rk] += gamma * (x0[0] - x0[k+1])` with physical-slot-0 anchored snapshot; per-mutation sigma upper clamp `(ub-lb)/sqrt(n)`; alpha-smoothing `sigma_out = sigma_parent + alpha * (sigma_new - sigma_parent)`; resample-on-bound with bounded retry; L2-squared violations. | Runarsson & Yao (2005), IEEE Trans. SMC-C 35(2):233-243; NLopt 2.10.0 `isres.c` (Steven G. Johnson 2009). | Pending empirical results. |
| `original_nablapp_policy.h` | Frozen baseline. Pre-rewrite ISRES preserved verbatim: alpha-pull-to-best operator (NOT differential variation), `j % mu` cyclic parent indexing, hard-clip bounds via `cwiseMax(lower).cwiseMin(upper)`, L1 violation aggregation, no sigma upper clamp, no sigma-smoothing remap, dead `[mu, lambda)` cyclic fill retained. | Pre-rewrite nablapp implementation; preserved as research artifact for empirical comparison. | Pending empirical results. |
| `runarsson_yao_paper_policy.h` | Paper-faithful. BEST-anchored DE recombination `x_new = x_1 + gamma * (x_i - x_1)` with rank-0-by-fitness snapshot; sigma upper clamp + alpha-smoothing inherited from NLopt with explicit citation gap on numeric defaults; L2-squared violations; resample-on-bound. | Runarsson & Yao (2005), IEEE Trans. SMC-C 35(2):233-243; ISRES+ Liu et al. 2023 Bioinformatics 39(7):btad403 Table 2 (numeric default attribution gap). | Pending empirical results. |

See `benchmarks/micro_isres.cpp` for the comparison harness; the
empirical decision trail and the production-alias selection commit
populate the "Empirical role" column at phase close.

## Lifecycle

These variants are compiled, tested, and benchmarked alongside the
production code. They are not subject to the same API-stability
promises as `solver/` types but are kept buildable so the comparison
can be re-run in any future commit. If a variant is dropped, the
deletion should be a single atomic commit referencing the empirical
decision.
