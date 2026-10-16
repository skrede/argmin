"""Every bound options value type, checked against the header that defines it.

This generalizes the two-header quadratic-program parity check into a
table-driven sweep over the whole surface: the two quadratic-program option
types, the shared driver options, and every policy option struct reachable
through a solver's options snapshot. Each expected value is recovered by
parsing the committed library header independently, so a default that drifts
away from the struct it mirrors fails the suite rather than agreeing with
itself.

The policy structs are checked twice: once on a default-constructed bound
instance, and once on the snapshot a solver constructed with no option keyword
reports. Only the second one can catch a keyword path that substitutes a
default of its own on the way in.

What the completeness assertion covers, and what it cannot see
--------------------------------------------------------------
`test_the_table_covers_every_bound_options_value_type` introspects the module
for bound value types -- a type defined in the extension, not an enumeration,
constructible with no arguments, and carrying no public callable -- and fails
on any that has no table entry. That is mechanical for anything exposed as a
bound value type, which is currently every options set on this surface,
because the nonlinear methods carry a readback snapshot for exactly this
purpose.

It is structurally blind to one shape: an options set that is keyword-only,
with no bound value type and no readback accessor. Such a set has no object to
introspect and no attribute to read, so nothing here can see it and nothing
here will notice its defaults drifting. The rule that follows is therefore a
rule about the surface, not about this module: an options set added to this
binding must be exposed as a bound value type reachable through a snapshot, or
it is unsweepable.
"""

import pytest

import argmin

from defaults_table import NOT_AN_OPTIONS_SET, TABLE
from defaults_solvers import POLICY_OPTION_TYPE, UNCONFIGURED, bound_value_types
from defaults_parser import (
    Designated,
    Empty,
    EnumMember,
    Unresolved,
    check_entry,
    declared_category,
    recover,
    resolve,
)


@pytest.mark.parametrize("binding_name", sorted(TABLE))
def test_the_parser_recovers_the_roster(repository_root, binding_name):
    for source in TABLE[binding_name].sources:
        members, _ = recover(repository_root, source)
        assert len(members) >= source.minimum_fields, (binding_name, source.struct, members)


@pytest.mark.parametrize("binding_name", sorted(TABLE))
def test_every_committed_default_is_the_exposed_default(repository_root, binding_name):
    check_entry(repository_root, getattr(argmin, binding_name)(), binding_name)


@pytest.mark.parametrize("binding_name", sorted(TABLE))
def test_a_misspelled_field_is_not_found(repository_root, binding_name):
    bound = getattr(argmin, binding_name)()
    for source in TABLE[binding_name].sources:
        members, _ = recover(repository_root, source)
        for field, _, _ in members:
            assert not hasattr(bound, source.renamed(field) + "_"), (
                f"{binding_name} answers to {field}_, so the attribute lookup this "
                "module relies on cannot discriminate"
            )



@pytest.mark.parametrize("solver_name", sorted(UNCONFIGURED))
def test_the_policy_snapshot_reports_the_committed_defaults(repository_root, solver_name):
    snapshot = UNCONFIGURED[solver_name]().policy_options()
    check_entry(repository_root, snapshot, POLICY_OPTION_TYPE[solver_name])


# The driver's stall window is the one field a policy overwrites: the driver
# copies the policy's window over it unconditionally, where the neighboring
# hints take the larger of the two. That is library behavior, so the field is
# excluded from the committed-default comparison and its actual value is pinned
# against the policy instead.
@pytest.mark.parametrize("solver_name", sorted(UNCONFIGURED))
def test_the_driver_snapshot_reports_the_committed_defaults(repository_root, solver_name):
    solver = UNCONFIGURED[solver_name]()
    snapshot = solver.options()
    committed = argmin.SolverOptions()

    for source in TABLE["SolverOptions"].sources:
        members, constants = recover(repository_root, source)
        for field, initializer, head in members:
            if source.reason(field):
                continue
            attribute = source.renamed(field)
            if attribute == "stall_window":
                continue
            expected = resolve(initializer, constants)
            if isinstance(expected, (Empty, Designated, Unresolved, EnumMember)):
                continue
            if declared_category(head) is float:
                expected = float(expected)
            assert getattr(snapshot, attribute) == expected, (solver_name, attribute)

    assert snapshot.stall_window == solver.policy_options().stall_window
    if snapshot.stall_window != committed.stall_window:
        assert solver.policy_options().stall_window != committed.stall_window



def test_the_table_covers_every_bound_options_value_type():
    unlisted = sorted(set(bound_value_types()) - set(TABLE) - set(NOT_AN_OPTIONS_SET))
    assert not unlisted, (
        f"{unlisted} are bound options value types with no entry in this sweep. Add "
        "the header and struct that define them, or list them as not a "
        "configuration set with a reason."
    )


def test_the_sweep_sees_every_type_it_claims_to():
    discovered = bound_value_types()
    for name in TABLE:
        assert name in discovered, f"{name} is in the table but is not a bound value type"
    for name, reason in NOT_AN_OPTIONS_SET.items():
        assert name in discovered and reason.strip(), name


# An assertion over a set difference that has only ever been evaluated on a
# passing input has not been shown to discriminate. This runs the same set
# arithmetic against a discovered type that is in neither list.
def test_the_completeness_assertion_is_discriminating():
    discovered = set(bound_value_types()) | {"FutureOptions"}
    unlisted = sorted(discovered - set(TABLE) - set(NOT_AN_OPTIONS_SET))
    assert unlisted == ["FutureOptions"]
