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
| `cmaes_sampling_marsaglia.h` | Marsaglia polar Gaussian on xoshiro256+; pair-cached via thread_local. | Marsaglia & Bray (1964), SIAM Review 6(3). | **Production winner** (Plan 05 perf-record A/B on micro_cmaes): drops the production std::normal_distribution<xoshiro256+> slice from 10.55% to 8.39% self-time, lowest combined sample_offspring + Gaussian transform cost (11.86%), tied with Ziggurat at end-to-end wall (495 ms vs 496 ms 5-rep median). Wins on tiebreaker by implementation simplicity (30 LOC body, no compile-time table). |
| `cmaes_sampling_ziggurat.h` | 256-region compile-time-table Ziggurat with Marsaglia-Tsang tail-handling. | Marsaglia & Tsang (2000), J. Statistical Software 5(8); Doornik (2005). | Loses by 0.16 pp on combined sample_offspring + Gaussian transform self-time (12.02% vs 11.86% Marsaglia); tied at end-to-end wall. The 256-region table-lookup pressure offsets the lower per-draw cost. Stays buildable for future re-comparison via temporary include swap in `detail/cmaes_sampling.h`. |

See `benchmarks/micro_cmaes.cpp` and the plan A/B verdict doc for the
perf-record A/B and the empirical decision.

## Lifecycle

These variants are compiled, tested, and benchmarked alongside the
production code. They are not subject to the same API-stability
promises as `detail/` types but are kept buildable so the comparison
can be re-run in any future commit. If a variant is dropped, the
deletion should be a single atomic commit referencing the empirical
decision.
