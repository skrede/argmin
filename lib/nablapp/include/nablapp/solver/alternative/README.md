# `solver/alternative/` — research variant policies

This directory holds alternative algorithm implementations preserved as
artifacts of the empirical comparison reported in the published paper.
None of these are part of the production API; production policies live
directly in `solver/`.

Each variant satisfies the same `nablapp::nlp_solver` concept as the
production policies and can be plugged into `basic_solver` interchangeably
for benchmarking.

## Subdirectories

### `gcmma/` — Globally Convergent MMA conservativity-globalization variants

Three different mechanisms for enforcing the conservativity condition
`g_tilde_i(x_trial) >= g_i(x_trial)` (Svanberg 2002 §4.2) on the MMA
reciprocal approximation. The production `solver/gcmma_policy.h` aliases
the empirical winner (`rho_wval_policy`).

| Variant | Mechanism | Reference | Empirical role |
|---|---|---|---|
| `move_limit_shrink_policy.h` | Contract `[alpha, beta]` toward `x_k` on non-conservative trial; preserves analytic primal `x_j(y)`. | Original to nablapp. | Non-viable: stalls short of `f*` on HS076 and burns wall time on contracted-trust-region dual solves. |
| `rho_wval_policy.h` | Augment approximation with separable quadratic penalty `rho * 0.5 * w_j * (x - x_k)^2`; grow `rho` per NLopt mma.c lines 388-391; numerical Newton primal. | Svanberg 2002 §4.2 + NLopt mma.c (Steven G. Johnson 2008-2012). | **Production winner**: fastest GCMMA variant on HS024/HS043/HS076 by both wall time and outer iter count. |
| `raa_augmented_policy.h` | Augment approximation with asymptote-divergent penalty `raa * (U - L) * (x - x_k)^2 / ((U - x)(x - L))`; canonical Svanberg 2002 globalization with global-convergence proof; numerical Newton primal. | Svanberg 2002 SIAM J. Optim. 12(2):555-573, eq. 3.4-3.8. | Converges correctly but ~2x slower than `rho_wval`; the asymptote-divergent `d_j` penalty does not pay off on this benchmark set. |

See `benchmarks/micro_mma.cpp` for the comparison harness and the commit
history (commits `f37d28e`, `d845463`, `4a591f7`, `e75ca6a`, `635bd2e`)
for the empirical decision trail.

## Lifecycle

These variants are compiled, tested, and benchmarked alongside the
production code. They are not subject to the same API-stability promises
as `solver/` types but are kept buildable so the comparison can be
re-run in any future commit. If a variant is dropped, the deletion
should be a single atomic commit referencing the empirical decision.
