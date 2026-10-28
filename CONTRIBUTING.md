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

The code style -- language floor, size limits, comments, includes, naming,
construction, ownership, formatting and testing -- is specified in
[CONVENTIONS.md](CONVENTIONS.md). Units over the size ceilings are
registered in [EXCEPTIONS.md](EXCEPTIONS.md) or recorded in the
shrink-only debt baseline it describes, and the size gate fails on
anything else.

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
