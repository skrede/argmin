# argmin Documentation

## Getting Started

- [Getting Started](getting-started.md) -- zero to compiling in a few minutes: install, a first solve, and bringing your own problem.

## Guides & References

- [Choosing a Solver](choosing-a-solver.md) -- a problem-class-to-policy map: which policy fits an unconstrained, constrained, least-squares, derivative-free, or global problem.
- [Real-Time Safety Matrix](rt-safety-matrix.md) -- the per-solver breakdown of allocation-freedom, bounded iterations, wall-clock-freedom, exceptions-off cleanliness, and seeded determinism, each cell citing its proving artifact.
- [Publication Benchmarking](benchmarking/publish-bench.md) -- methodology, provenance requirements, and baseline eligibility rules for the publication-grade benchmark harness.

## Examples

- [examples/](../examples/) -- runnable programs: an unconstrained solve, a constrained (SLSQP) solve, least-squares, derivative-free, and step-level / budgeted execution for control loops.

## Contributing

- [CONTRIBUTING.md](../CONTRIBUTING.md) -- contribution workflow, coding conventions, commit-message format, and the branching model.
