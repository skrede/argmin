# `detail/alternative/` -- research variant detail helpers

Sampling-level variants that swap in for the production
`detail/cmaes_sampling.h::sample_offspring` body. None of these are
part of the production API; production sampling lives in
`detail/cmaes_sampling.h` and aliases to the empirical winner.

This subdirectory extends the `solver/alternative/` convention
(variants live alongside production, are compiled+tested+benchmarked,
are subject to the same lifecycle rules) to the `detail/` layer for
sampling-level helpers.

| Variant | Mechanism | Reference | Empirical role |
|---|---|---|---|
| `cmaes_sampling_marsaglia.h` | Marsaglia polar Gaussian on xoshiro256+; both halves of each polar pair are consumed locally within one call (no persistent cache), so the output is a pure function of the RNG stream. | Marsaglia & Bray (1964), SIAM Review 6(3). | **Production winner** (perf-record A/B on micro_cmaes): drops the production std::normal_distribution<xoshiro256+> slice from 10.55% to 8.39% self-time, lowest combined sample_offspring + Gaussian transform cost, at indistinguishable end-to-end wall. Wins on implementation simplicity (no compile-time table). |

A 256-region compile-time-table Ziggurat variant (Marsaglia & Tsang 2000;
Doornik 2005) was also evaluated but removed: its published-style
construction truncated the tail at the sentinel |z| ~ 3.911 (the strip-0
acceptance ratio was >> 1, so the tail fallback was never reached), a
correctness defect whose marginal speed edge did not justify the
maintenance risk for a non-production variant.

See `benchmarks/micro_cmaes.cpp` for the perf-record A/B and the
empirical decision.

## Lifecycle

These variants are compiled, tested, and benchmarked alongside the
production code. They are not subject to the same API-stability
promises as `detail/` types but are kept buildable so the comparison
can be re-run in any future commit. If a variant is dropped, the
deletion should be a single atomic commit referencing the empirical
decision.
