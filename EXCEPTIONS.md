# Size exceptions

Functions are held to 5-15 lines (25 at most) and files to about 100 lines (200 at most). The units
below are the only ones allowed past those ceilings. Each one belongs to a named class. A unit joins a
class only when it matches the class's description, so the class is the justification, not the unit.

## Init-list-dominated constructors

A constructor whose body is empty or trivial and whose length comes from a member initializer list
that names every member in declaration order. Splitting it would move the initializers away from the
one place the project requires them to be, so the list's length follows the member count, not the
constructor's logic.

| Unit | Location | Lines |
|------|----------|-------|
| `cobyla_engine<Scalar>::cobyla_engine()` | `lib/argmin/include/argmin/detail/cobyla_engine_cycle.h` | 32 |
