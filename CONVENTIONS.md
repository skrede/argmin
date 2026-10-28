# Coding conventions

The goal is a library that is lean -- small, readable, fast -- with no loss of capability or speed.
These conventions are written for people, and they apply to everyone, including the agents that write
much of the code. This file is the authoritative statement of argmin's code style; `.clang-format`
records the token-level shape, and [EXCEPTIONS.md](EXCEPTIONS.md) is the register of sanctioned
over-limit units.

The rules serve readability. Where one of them makes a particular spot harder to read, readability
wins and that spot is a fine exception -- except for the size ceilings, which a machine checks and
which yield only through the registers described below.

## No planning artifacts

Never reference an ID, key or label produced by a planning or issue-tracking tool -- phase numbers,
milestone names, plan or task numbers, requirement IDs, invariant labels -- anywhere in the codebase:
identifiers, comments, documentation, examples and commit messages. State the thing itself in plain
words. If a reader would have to open a planning document to decode a comment, the comment is wrong.
The same goes for narrative from how the code was built ("the slice", "the gate", "this milestone"),
which means nothing to a reader of the finished code.

## Language and portability

- **C++20 is the floor**, which keeps the library usable on the wide range of MCU toolchains. Use
  modern features where they make the code clearer -- concepts, `if constexpr`, designated
  initializers, ranges. A C++23-only feature is guarded by its feature-test macro with a C++20
  fallback, or not used. `std::expected` is C++23, so `<argmin/expected.h>` provides
  `argmin::expected`, which aliases the standard type when the toolchain has it.
- The code compiles and runs on macOS, Linux and Windows, and on the embedded targets. Keep
  platform-specific code isolated, never in the solver core.
- **Header-only.** Public headers live under `lib/argmin/include/argmin/`; there is no compiled
  library artifact.
- **Exception-free and RTTI-free.** The library builds with `-fno-exceptions -fno-rtti`. Report
  errors through `expected` and status values rather than throwing; do not use `dynamic_cast` or
  `typeid`; new code adds no `throw`.
- **Eigen is the only required dependency.** argmin reinvents no matrix arithmetic.
- Follow the conventions of modern C++ libraries -- the standard library, asio, boost. Existing
  project conventions take precedence. When unsure, or when a popular idiom contradicts the project,
  ask rather than guess.

## Size

Lines include comments and blank lines; the budget is the unit as it sits on screen.

| Unit     | Target     | Hard ceiling |
| -------- | ---------- | ------------ |
| Function | 5-15 lines | 25 lines     |
| File     | ~100 lines | 200 lines    |

Each scope does one thing -- a library, a namespace, a class, a function -- and the smaller that one
thing is, the cleaner the code reads. A unit over its ceiling is usually its one thing drawn too broad:
decompose it, do not widen the budget. Group code that is read together and split where a
responsibility genuinely separates, never to chase a number.

Avoid:

- **Salami-slicing** -- carving a cohesive unit into fragments to fit a limit.
- **One-function files** -- a file that exists only to hold a single function.
- **Artificial purity** -- a wrapper or layer that adds indirection without separating a real
  responsibility.

Internal helpers go in a `detail/` directory and the `detail::` namespace, which keeps the public
headers lean. A `detail/` file is still a real file and obeys the same limits.

### How the ceilings are measured

A file's length is every line in it. A function's length is the count clang-tidy's
`readability-function-size` check reports: the body, from its opening brace to its closing brace. The
signature, the template header and the member initializer list fall outside the count; blank lines
and comments inside the body fall within it. Template patterns are measured whether or not anything
instantiates them, and code in an inactive `#if` branch is not measured.

The checkers measure the files under `lib/` that are tracked in version control:

- `scripts/check_file_size.py` measures file lengths. It needs no build and no configure.
- `scripts/check_function_size.py` runs clang-tidy 18 over a generated translation unit that includes
  every tracked header under `lib/argmin/include/`, so no header escapes because nothing includes it.
  It needs clang-tidy 18 (`--clang-tidy`, or the `CLANG_TIDY` environment variable) and the Eigen
  headers (`--eigen-include`). Where clang 18 cannot parse the default standard library, pass the
  compiler a fitting one, for example `--extra-arg=--gcc-install-dir=/usr/lib/gcc/<triple>/13.4.1`.

Run both from the repository root. Each failure has its own exit code, listed by `--help`.

### The two registers

A unit over its ceiling is in exactly one of two registers, or the gate fails.

- **[EXCEPTIONS.md](EXCEPTIONS.md)** holds the permanent exceptions: units that are a single cohesive
  whole that splitting would harm, each with a reason naming that harm. A registered unit may grow
  to its recorded count plus ten percent before the gate fails. There is no in-code marker.
- **`scripts/size_baseline.tsv`** is the debt baseline: the units that predate the ceilings and have
  not yet been decomposed, each with its exact line count. It only shrinks. Growth past a recorded
  count fails; so does a row that records more lines than its unit has, and a row whose unit is back
  under the ceiling or gone. When a change shrinks or decomposes a unit, lower or delete its row in
  the same change. The gate compares the baseline against the pull request's base, or the previous
  push, and fails if a row was added or a count raised. A unit that grows past its ceiling is
  decomposed or registered as a permanent exception, never absorbed into the baseline.

A change is not done until every unit it touches is within budget or its overage is registered.

## Comments

The codebase is written for people to read, and a comment must say something the code cannot. **The
default is no comment**: clear names and signatures carry the meaning. Write a comment only when it

- **(a)** cites a source -- a paper, RFC, specification, URL, bug report or standards clause (an
  algorithm header cites the textbook section or paper it implements);
- **(b)** names a non-obvious algorithm or technique so a reader can look it up; or
- **(c)** explains a non-obvious design choice in one or two sentences -- a bug workaround, a
  deliberate tradeoff, or a contract, ordering, side effect or invariant that is real but not visible
  from the code.

Never write comments that restate a name, signature, type or obvious control flow; tables that
transcribe the code beneath them; sign-posting or narration ("this section does X", "first we ...
then we ..."); bookkeeping or process markers such as size-gate justifications, ledgers or "see the
doc" pointers; or design history ("relocated from ...", "X was the example").

A comment longer than a short tag goes on its own line above the code, never trailing; a brief
trailing tag on a data line, such as a unit or an enumerator's value, is fine. Keep a multi-line
comment tight, with no blank `//` lines padding it out. Comments count toward the size budget.

## Includes

Project includes (`#include "..."`) come first, third-party libraries second, the standard library
third. The three sections are separated by one blank line.

- Within a section, includes are grouped by folder, one blank line between groups, and never more
  than one blank line between any two groups.
- In the project section the file's own subsystem (its folder) comes first, then the remaining
  project folders.
- Within a folder group, includes sort by file-name length, shortest first; names of equal length
  sort alphabetically.

Order differs from this only where one header genuinely must precede another. `.clang-format` does
not sort includes, so the order is kept by hand.

## Header guards and namespaces

- Header guards, not `#pragma once`, in the form `HPP_GUARD_<NAMESPACE>_<FOLDER>_<FILE>_H`, for
  example `HPP_GUARD_ARGMIN_DETAIL_CAUCHY_POINT_H`.
- No `// namespace argmin` comment after a closing namespace brace, and no comment after the guard's
  `#endif`.

## Naming

- Types, functions and variables are lowercase with underscores (`lbfgsb_policy`,
  `step_budget_solver`). Template parameters are `CamelCase` (`Scalar`, `Policy`, `Problem`, `N`).
- Private data members take a trailing underscore (`n_`, `updates_since_reset_`).
- A member holding a callback or hook (a `move_only_function`) takes a `_cb` suffix and groups with
  the other callbacks; the suffix marks it as injected behavior, not state.

## Types and attributes

- **Spell out return types.** `auto` as a function's return type hides what a reader looks for
  first; the only exception is a genuinely unspellable type, such as a closure. `auto` for a local is
  fine where the initializer makes the type obvious.
- **No `[[nodiscard]]`**, and no other attribute the compiler does not require.
- **Fixed-width integers** (`std::int32_t`, `std::uint16_t`, ...) rather than `int`, `long` and
  friends, and never a platform-specific alias where a fixed-width type serves. A dimension or index
  handed to or taken from Eigen uses `Eigen::Index`, the type Eigen itself uses.

## API design

- No raw pointers in public APIs. Pass by `const&`, by value, by `std::span`, or by smart pointer
  where that is the right ownership story. On a zero-allocation path, take a read-only Eigen input as
  `const Eigen::MatrixBase<Derived>&`: `Eigen::Ref<const T>` silently copies into a temporary when the
  argument does not match its layout.
- Distinguish **required** (no default), **required with a default** (the override is optional) and
  **`std::optional`** (absence is itself meaningful). An options field with a literature default is a
  plain value carrying that default, never an `optional` standing in for one.
- The library is pre-1.0: no `[[deprecated]]` and no compatibility shims. A superseded type or
  behavior is removed outright.

## Construction

- A class with a constructor initializes its members in the **constructor's init list, never
  in-declaration**, so there is one source of truth and two defaults can never drift apart.
- A **pure aggregate** with no user-declared constructor, typically an options struct, carries its
  defaults in-declaration. It stays an aggregate, so a caller designated-initializes only the fields
  it overrides and every other field keeps its default.
- The init list is written in member-declaration order, since members initialize in declaration
  order regardless (no `-Wreorder`). A member consumed by another member's constructor is declared
  before its consumer, and that dependency overrides the member triangle below.

## Class layout

Data comes before the operations on it:

```
class T
{
    // leading private nested helper types, if any

public:
    constructors
    public functions

private:
    data members
    private functions
};
```

The data members are ordered by the member triangle, subject to construction order.

## Lifetimes and ownership

- Lifetimes are structural and single-owner; the owner sequences teardown.
- **Raw pointers are an anti-pattern.** `unique_ptr` for sole ownership, `shared_ptr` for genuinely
  shared ownership, and references for everything else: an owner hands its long-lived utilities out
  by reference, through a context object the parts reach. A raw pointer survives only where absence
  is a real, modeled state, such as a nullable lookup.
- A non-owning relationship that must outlive its holder is a reference, stored as
  `std::reference_wrapper<T>` when it lives in a container that needs assignability.
- No `shared_from_this` or `enable_shared_from_this`, and no liveness flags guarding a posted task
  against an expired target; fix the ownership and teardown order instead.

## Formatting

`.clang-format` records the token-level shape, and no CI job runs it; use it as a starting point and
wrap by hand where the shapes below read better. The values that matter:

- Allman braces, four-space indentation, no tabs, a 220-column limit.
- No space before a parenthesis: `if(`, `for(`, `while(`, `switch(`.
- `case` labels sit at the indentation of their `switch`, and a label and its body are on separate
  lines.
- `const` goes on the left, and pointers and references bind to the type (`const Scalar& x`).
- Constructor initializers break before the comma.

### Shape

These are judgment calls for the eye, not mechanical rules.

- **Downward waterfall.** When a signature or declaration wraps, each continuation line is at least
  as long as the one above it, so the wrapped arguments slope down and the eye falls to the next one.
- **Member triangle.** A class's data members are ordered by declaration-line length, shortest first,
  so they form a downward slope. A single space precedes each name; names are not aligned into a
  column. Semantic grouping wins where it helps, such as keeping the callbacks together.
- **Logical blank lines.** A blank line inside a function separates its logical parts, the way a
  paragraph break does. Member functions are separated by one blank line; the data members are
  packed together.

## Testing

- **Unit tests** use Catch2 and cover every public API path.
- **Property-based tests** use RapidCheck (`ARGMIN_BUILD_PROPERTY_TESTS`); for numerical kernels they
  catch what handpicked cases miss.
- **Fixed-seed determinism.** A test must not depend on an implementation-defined random stream:
  `std::normal_distribution` and `std::uniform_real_distribution` produce different sequences across
  standard libraries. Derive random inputs from the raw engine output, and prefer well-conditioned
  inputs over assertions that ride a floating-point rounding boundary.

## English

American spelling: "stabilizing", not "stabilising"; "color", not "colour".

## Commit messages

```
{Prefix}: {summary sentence}.

- {what was done, one line per item}
- {another item if applicable}
```

The summary line is brief and descriptive, and the bullets expand on it; a single-item commit may
omit them. The prefixes:

| Prefix          | When to use |
|-----------------|-------------|
| `Feature:`      | New user-visible capability or API surface. |
| `Fix:`          | Bug fix or correctness-affecting change. |
| `Refactor:`     | Internal reorganization with no user-visible behavior change. |
| `Build:`        | Build system, CI and tooling. |
| `Docs:`         | Documentation only. |
| `Examples:`     | Changes to `examples/` only. |
| `Optimization:` | Performance change with no API or correctness impact. |
| `Test:`         | Tests only. |
| `WIP:`          | A commit whose code does not compile. |

The rule against planning artifacts applies to commit messages too.
