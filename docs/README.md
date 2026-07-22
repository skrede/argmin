# argmin Documentation

## Getting Started

- [Getting Started](getting-started.md) -- zero to compiling in a few minutes: install, a first solve, and bringing your own problem.

## Guides & References

- [Choosing a Solver](choosing-a-solver.md) -- a problem-class-to-policy map: which policy fits an unconstrained, constrained, least-squares, derivative-free, or global problem.
- [Real-Time Safety Matrix](rt-safety-matrix.md) -- the per-solver breakdown of allocation-freedom, bounded iterations, wall-clock-freedom, exceptions-off cleanliness, and seeded determinism, each cell either citing its proving artifact or marked as argued from reasoning alone.
- [Determinism](determinism.md) -- what "deterministic" is guaranteed to mean and at which scope: run-to-run and post-`reset()` bit-exactness, cross-instantiation decision identity, bounded cross-instantiation numeric agreement, and the deliberate anti-feature of cross-architecture bit-identity.
- [Embedded Cross-Compiling](embedded.md) -- carrying the zero-allocation contract onto bare-metal targets: the Eigen stack-temporary flags (`EIGEN_ALLOCA`, `EIGEN_STACK_ALLOCATION_LIMIT`), the newlib strict-dialect trap, and the stack-budgeting rule.
- [Python Bindings](python-bindings.md) -- the bound surface, the error and lifetime contracts across the interpreter boundary, the measured cost of an objective defined in Python, and the list of things deliberately not provided.
- [Publication Benchmarking](benchmarking/publish-bench.md) -- methodology, provenance requirements, and baseline eligibility rules for the publication-grade benchmark harness.

## Examples

- [examples/](../examples/) -- runnable programs: an unconstrained solve, a constrained (SLSQP) solve, least-squares, derivative-free, and step-level / budgeted execution for control loops.

## Contributing

- [CONTRIBUTING.md](../CONTRIBUTING.md) -- contribution workflow, coding conventions, commit-message format, and the branching model.
