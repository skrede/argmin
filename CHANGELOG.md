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

### Changed

- Every policy, the QP and line-search options, the convergence criteria and
  `default_convergence` take the scalar type as their first template argument
  (`float` or `double`, defaulting to `double`), for example
  `nw_sqp_policy<double, NV>` and `gradient_tolerance_criterion<>`.
- The QP options' `max_iterations` is `std::uint32_t`.
- `float` is supported per algorithm family, each gated against `double`; see
  [docs/float-support.md](docs/float-support.md).

### Fixed

- `ccsa_quadratic` in `float` keeps its move box resolvable: the minimum move
  limit is 3e-4 of the variable range instead of 1e-8, which is below `float`
  precision at unit scale.

### Notes

- Every policy plugs into `step_budget_solver<Policy>` and exposes a
  uniform `step()` / `step_n(budget)` / `solve()` interface.
- Several policies are certified allocation-free at a fixed compile-time
  dimension under the allocation-counting CI gate; see
  [docs/rt-safety-matrix.md](docs/rt-safety-matrix.md) for the
  per-solver breakdown.
- Known limitations at HEAD are tracked in the README's Status section
  and the solver-selection guide.
