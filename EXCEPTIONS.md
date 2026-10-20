# Size exceptions

Functions are held to 5-15 lines (25 at most) and files to about 100 lines (200 at most). The units
below are the only ones allowed past those ceilings. Each one belongs to a named class. A unit joins a
class only when it matches the class's description, so the class is the justification, not the unit.

The library does not meet these limits yet. As of 2026-10-20, 71 of the 167 headers in `lib/` exceed 200
lines, and 26 functions exceed 200 lines, most of them a solver's `step()`. They predate the limits, and
a decomposition is planned. They are not exceptions: this file lists only units sanctioned to stay over
the limits afterwards.

## Init-list-dominated constructors

A constructor whose body is empty or trivial and whose length comes from a member initializer list
that names every member in declaration order. Splitting it would move the initializers away from the
one place the project requires them to be, so the list's length follows the member count, not the
constructor's logic.

| Unit | Location | Lines |
|------|----------|-------|
| `cobyla_engine<Scalar>::cobyla_engine()` | `lib/argmin/include/argmin/detail/cobyla_engine_cycle.h` | 32 |
