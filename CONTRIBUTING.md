# Contributing to argmin

Thanks for your interest in contributing. argmin is a nonlinear
optimization library aimed at embedded, real-time, and control
applications; contributions of all sizes -- documentation fixes, bug
reports, code improvements, new solver policies -- are welcome.

## Quick Links

- [Filing an Issue](#filing-an-issue)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Building and Testing](#building-and-testing)
- [Coding Conventions](#coding-conventions)
- [Commit Message Format](#commit-message-format)
- [Branching Model](#branching-model)
- [License](#license)

## Filing an Issue

- **Bug reports.** Include a minimal reproducer (a short program that
  triggers the issue), the compiler + version (`g++ --version` or
  equivalent), the CMake version, and the platform. Numerical-issue
  reports should include the problem (objective, constraints, bounds)
  and the starting point that triggers the misbehavior, plus the
  observed and expected result (status, `x`, objective value).
- **Feature requests.** Describe the use case first, the proposed API
  shape second. A concrete problem class makes it easier to evaluate the
  proposal against the library's scope.
- **Questions.** Open an issue tagged "question" -- there is no chat
  channel to fragment discussion.

If you're not sure whether something is a bug or expected numerical
behavior, an issue is the right place to ask.

## Submitting a Pull Request

1. **Fork** the repository and create a feature branch from `develop`
   (or from the active `milestone/<version>` branch if you're
   contributing to in-progress milestone work -- ask first if unsure).
2. **Build and test** locally. The full ctest suite should pass
   (`ctest --output-on-failure` in your build directory) before you
   submit. New features need new tests; new tests need to pass.
3. **Match the coding conventions.** See [Coding Conventions](#coding-conventions)
   below. PRs that diverge stylistically will be asked to converge
   before review.
4. **Write a clear PR description.** What problem does this solve? What
   approach did you take? What alternatives did you consider? Link
   relevant issues.
5. **One logical change per PR.** Don't bundle unrelated cleanups with
   feature work; file separate issues for things you spot along the way.

## Building and Testing

See [docs/getting-started.md](docs/getting-started.md) for end-user
install and build instructions. For contributors:

```bash
# Configure with tests enabled and dependencies auto-fetched.
cmake -B build -DARGMIN_BUILD_TESTS=ON -DARGMIN_CMAKE_FETCH_DEPS=ON

# Build and run the suite.
cmake --build build
ctest --test-dir build --output-on-failure
```

Additional opt-in targets: `ARGMIN_BUILD_EXAMPLES`, `ARGMIN_BUILD_BENCHMARKS`,
`ARGMIN_BUILD_PROPERTY_TESTS` (RapidCheck), and `ARGMIN_BUILD_FUZZ_TESTS`.

The library must stay clean under the sanitizer and cross-standard CI
jobs (Address/UB/Memory sanitizers, the C++20 floor, and an
`-fno-exceptions -fno-rtti` instantiation probe). Changes on a
real-time-claimed policy path must not introduce heap allocation on the
warm-started steady-state step -- the allocation-counting gate fails the
build if they do.

## Coding Conventions

argmin targets idiomatic C++20 with first-class embedded targets. The
conventions below apply to all contributions.

### Language and style

- **C++20 is the floor** (required for wide MCU toolchain support). Use
  modern features where they improve clarity -- `concepts`, `if
  constexpr`, designated initializers, `std::ranges`. C++23-only
  features must be feature-test-macro guarded with a C++20 fallback, or
  dropped. `std::expected` is C++23, so the library ships an in-library
  `argmin::expected` backport via `<argmin/expected.h>` that aliases the
  standard type when the toolchain provides it.
- **Exception-free and RTTI-free.** The library builds
  `-fno-exceptions -fno-rtti`. Return `expected`-based errors rather than
  throwing; do not use `dynamic_cast` / `typeid`. New code must not add
  `throw` sites.
- **Eigen is the only required dependency.** argmin reinvents no matrix
  math.
- **Naming:** lowercase types and functions (`lbfgsb_policy`,
  `step_budget_solver`); private member variables take a trailing
  underscore (`n_`, `updates_since_reset_`); template parameters in
  `CamelCase` (`Scalar`, `Policy`, `Problem`, `N`).
- **Braces:** Allman style (opening brace on its own line for functions,
  classes, and control flow). 4-space indentation, no tabs.

### Headers

- **Header-only.** Public headers live under
  `lib/argmin/include/argmin/`; there is no compiled library artifact.
- **Header guards** in the form `HPP_GUARD_ARGMIN_<FOLDER>_<FILE>_H`. No
  `#pragma once`.
- **No matching closing-namespace comment** (`// namespace argmin` is
  noise after the closing brace; just close the brace).
- **No matching `#endif` comment** for the include guard.

### Include ordering

Includes are grouped into three sections separated by a single blank
line:

1. Internal project includes (`#include "..."`).
2. Third-party library includes (`#include <Eigen/...>`,
   `#include <catch2/...>`).
3. Standard library includes (`#include <vector>`, `#include <expected>`).

Within each section, group by folder location with a blank line between
folder groups. Within each folder group, sort by line length first, then
alphabetically. Order matters only where a header genuinely must precede
another.

### Documentation comments

Every public class, function, template, and concept carries a doc-comment
naming its purpose, parameters, return value, and edge cases worth
flagging. Keep doc-comments at the *why* level for non-obvious behavior;
don't restate what well-named identifiers already communicate. Algorithm
headers cite their source (textbook + section, or the original paper).

### Testing

- **Unit tests** via Catch2: cover every public API path.
- **Property-based tests** via RapidCheck (optional target): for
  numerical kernels, property tests catch what handpicked cases miss.
- **Fixed-seed determinism.** Tests must not depend on an
  implementation-defined RNG stream: `std::normal_distribution` /
  `std::uniform_real_distribution` produce different sequences across
  standard libraries. Derive random inputs from the raw engine output,
  and prefer well-conditioned inputs over assertions that ride the
  floating-point rounding boundary.

## Commit Message Format

```
{Prefix}: {summary sentence}.

- {what was done; one bullet per logical item}
- {another item if applicable}
```

Allowed prefixes:

| Prefix          | When to use |
|-----------------|-------------|
| `Feature:`      | New user-visible capability or API surface. |
| `Fix:`          | Bug fix or correctness-affecting change. |
| `Refactor:`     | Internal reorganization with no user-visible behavior change. |
| `Docs:`         | Documentation-only changes (README, docs/, doc-comments). |
| `Examples:`     | Changes to `examples/` only. |
| `Optimization:` | Performance change with no API or correctness impact. |
| `WIP:`          | Work-in-progress commits whose code does not yet compile. Use sparingly; squash before merging. |

The summary line is brief and descriptive (under 72 characters where
possible). The bullet list expands on the what. Single-item commits may
omit the bullet list. Keep one logical change per commit.

## Branching Model

```
master   <- develop   <- milestone/<version>
(releases)  (integration)  (active work)
```

- **master** holds releases.
- **develop** is the integration branch; milestone branches merge into
  develop when each milestone ships.
- **milestone/v0.X.Y** branches host active work on a specific milestone.
  External contributors should normally branch from `develop`; if you're
  contributing to in-progress milestone work, ask first via an issue or
  PR comment.
- Merge direction is always milestone -> develop -> master.

## License

By contributing, you agree that your contributions will be licensed under
the [Apache License 2.0](LICENSE), the same license that covers the
project.
