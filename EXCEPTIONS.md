# Size exceptions

Files are held to about one hundred lines and two hundred at most, counting every line. Functions are
held to five to fifteen lines and twenty-five at most. Two registers record the units over those
ceilings, and two checkers enforce them.

This file is the register of **permanent exceptions**: units that are over a ceiling and are
sanctioned to stay that way, each with its reason. A row belongs here only when the unit is a single
cohesive whole that splitting would *harm*. The first answer to an over-limit unit is to decompose it.

`scripts/size_baseline.tsv` is the **debt baseline**: the units that predate the ceilings and have
not yet been decomposed, each with its exact line count. It only shrinks. A row's count may go down
and a row may be deleted; a row may not be added and a count may not rise. The gate compares the
baseline against the pull request's base, or the push's previous commit, and fails if it grew. A
unit that grows past its ceiling is decomposed or registered here, never absorbed into the baseline.

## What the checker enforces

`scripts/check_file_size.py` measures the file lengths. It fails on:

- an over-limit file that is in neither register;
- a stale row, whose file is at or under the ceiling or no longer exists;
- growth: a registered file past its recorded count plus ten percent, or a baseline file past its
  exact recorded count;
- an inflated row, which records more lines than its file has and must be re-recorded down;
- a malformed register or baseline, or a file listed in both;
- a baseline that gained a row or a count since the comparison commit.

Each failure has its own exit code and message, because each has a different fix. A malformed row is
a failure, not a skip. There is no in-code marker and there are no silent exceptions.

A function's length is the line count clang-tidy's `readability-function-size` check reports: the
body from its opening brace to its closing brace, so the signature and the member initializer list
are not counted.

## Scope

The checkers measure every file under `lib/` that is tracked in version control and has a C++
extension. Untracked files are invisible to them, since the file list comes from `git ls-files`.
`tests/` and `benchmarks/` are not measured.

## Reasons

A reason must name what splitting the unit would harm. One that only says the unit is large, or
cohesive, is rejected. A reason that restates the line count is rejected, because the count has its
own column. A reason that promises a future split is rejected too: this file registers decisions
already taken, not work outstanding.

Each row records the unit's length when it was registered. A registered unit may grow to that count
plus ten percent, rounded up, before the checker fails. When a change exceeds that, re-record the
count deliberately or decompose the unit.

## Files

| Path | Lines at registration | Why |
| --- | --- | --- |
| lib/argmin/include/argmin/test_functions/hock_schittkowski.h | 2139 | a data table transcribing the Hock-Schittkowski catalogue, one problem definition after another with its published starting point and optimum; splitting it would scatter one reference table across files that are only ever read and checked against the source together |

## Functions

| Path | Function | Lines at registration | Why |
| --- | --- | --- | --- |
