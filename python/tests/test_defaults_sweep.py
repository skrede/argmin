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

import re
import math
import pathlib
import inspect
import dataclasses

import numpy as np
import pytest

import argmin

INCLUDE_ROOT = pathlib.Path("lib") / "argmin" / "include" / "argmin"

INFINITY = "std::numeric_limits<double>::infinity()"


class Unresolved:
    def __init__(self, text):
        self.text = text


class Empty:
    pass


@dataclasses.dataclass(frozen=True)
class EnumMember:
    name: str


@dataclasses.dataclass(frozen=True)
class Designated:
    fields: tuple


@dataclasses.dataclass(frozen=True)
class Source:
    header: str
    struct: str
    minimum_fields: int
    rename: tuple = ()
    unchecked: tuple = ()

    def renamed(self, field):
        return dict(self.rename).get(field, field)

    def reason(self, field):
        return dict(self.unchecked).get(field)


@dataclasses.dataclass(frozen=True)
class Entry:
    sources: tuple


def entry(*sources):
    return Entry(sources=tuple(sources))


# Every bound options value type, mapped to the committed header and struct
# that defines it. A field listed as unchecked carries the reason it cannot be
# compared here; a field with no entry and no reason fails.
TABLE = {
    "DenseQpOptions": entry(Source("options/dense_qp_options.h", "dense_qp_options", 17)),
    "SparseQpOptions": entry(Source("options/sparse_qp_options.h", "sparse_qp_options", 17)),
    "LineSearchOptions": entry(Source("line_search/options.h", "line_search_options", 5)),
    "QpSubproblemOptions": entry(Source("options/qp_options.h", "qp_options", 2)),
    "TrustRegionOptions": entry(Source("options/trust_region_options.h", "trust_region_options", 6)),
    "CmaesDetectionOptions": entry(Source("options/cmaes_options.h", "cmaes_options", 4)),
    "LbfgsbOptions": entry(Source("solver/lbfgsb_policy.h", "options_type", 3)),
    "BobyqaOptions": entry(Source("solver/bobyqa_policy.h", "options_type", 5)),
    "CobylaOptions": entry(Source("solver/cobyla_policy.h", "options_type", 4)),
    "SlsqpOptions": entry(Source("solver/kraft_slsqp_policy.h", "options_type", 6)),
    "CmaesOptions": entry(
        Source(
            "solver/alternative/cmaes/pwq_reparameterization_policy.h",
            "options_type",
            9,
            rename=(("lambda", "population_size"), ("cmaes", "detection")),
            unchecked=(
                (
                    "eigendecomposition_skip_generations",
                    "not on the bound surface: an override for a formula the policy "
                    "derives, deferred with the rest of the tuning knobs",
                ),
                (
                    "stagnation_window",
                    "not on the bound surface, for the same reason as the "
                    "eigendecomposition override above",
                ),
            ),
        )
    ),
    # The driver's convergence member is a template parameter, so the header
    # that declares it carries no literal to compare. The thresholds the
    # binding exposes as flat properties live in the criterion structs, which
    # are swept here as their own sources.
    "SolverOptions": entry(
        Source(
            "solver/options.h",
            "solver_options",
            4,
            unchecked=(
                (
                    "convergence",
                    "a template parameter with no literal of its own; its criteria "
                    "are swept below as separate sources",
                ),
            ),
        ),
        Source(
            "solver/convergence.h",
            "gradient_tolerance_criterion",
            1,
            rename=(("threshold", "gradient_threshold"),),
        ),
        Source(
            "solver/convergence.h",
            "objective_tolerance_criterion",
            2,
            rename=(("threshold", "objective_threshold"),),
        ),
        Source(
            "solver/convergence.h",
            "step_tolerance_criterion",
            2,
            rename=(("threshold", "step_threshold"),),
            unchecked=(
                (
                    "feasibility_tolerance",
                    "the criterion's own copy; the driver-level tolerance of the "
                    "same name is the one the binding exposes",
                ),
            ),
        ),
        Source(
            "solver/convergence.h",
            "stall_tolerance_criterion",
            4,
            rename=(("threshold", "stall_threshold"), ("window", "stall_window")),
            unchecked=(
                (
                    "track_best_seen_feasible",
                    "not on the bound surface: a driver-internal selection switch, "
                    "not a tolerance",
                ),
                (
                    "feasibility_tolerance",
                    "the criterion's own copy, as in the step criterion above",
                ),
            ),
        ),
    ),
}

# A bound value type that is not a configuration set. Each needs a reason, so
# the escape hatch from the completeness assertion stays visible.
NOT_AN_OPTIONS_SET = {
    "QpResult": "an output value object; its fields are produced by a solve rather "
    "than configured by a caller",
    "SolveResult": "an output value object, as above",
}

STRUCT_TO_BINDING = {entry_.sources[0].struct: name for name, entry_ in TABLE.items()}
# The policy option structs are all spelled options_type, so the struct name
# alone cannot address them; nothing nests one inside another, and only the
# nested lookup uses this map.
STRUCT_TO_BINDING.pop("options_type", None)


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def matching_brace(text, opening):
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    raise AssertionError("unbalanced braces")


def struct_body(text, struct):
    match = re.search(r"\bstruct\s+" + re.escape(struct) + r"\b\s*\{", text)
    assert match, f"no struct {struct} in the header"
    opening = match.end() - 1
    return text[opening + 1 : matching_brace(text, opening)]


def declared_constants(text):
    pattern = r"static\s+constexpr\s+[\w:<>,\s]*?\b(\w+)\s*=\s*([^;]+);"
    return {name: value.strip() for name, value in re.findall(pattern, text)}


def brace_initialized_members(body):
    """Top-level `Type name{init};` declarations, in declaration order.

    A member function body and a requires-clause are both a brace run that is
    not followed by a semicolon, which is what distinguishes them from a member
    initializer here. Splitting on semicolons alone cannot do it, because a
    function definition carries none.
    """
    members = []
    index = 0
    head_start = 0
    while index < len(body):
        character = body[index]
        if character == "{":
            closing = matching_brace(body, index)
            after = closing + 1
            while after < len(body) and body[after].isspace():
                after += 1
            if after < len(body) and body[after] == ";":
                head = body[head_start:index].strip()
                name = re.search(r"(\w+)\s*$", head)
                if name and not name.group(1).endswith("_"):
                    members.append((name.group(1), body[index + 1 : closing].strip(), head))
                index = after + 1
            else:
                index = closing + 1
            head_start = index
            continue
        if character == ";":
            index += 1
            head_start = index
            continue
        index += 1
    return members


def split_top_level(text, separator):
    parts = []
    depth = 0
    current = ""
    for character in text:
        if character in "{<(":
            depth += 1
        elif character in "}>)":
            depth -= 1
        if character == separator and depth == 0:
            parts.append(current)
            current = ""
            continue
        current += character
    parts.append(current)
    return [part.strip() for part in parts if part.strip()]


def resolve(expression, constants, depth=0):
    text = " ".join(expression.split()).strip().rstrip(",").strip()
    if text == "":
        return Empty()
    if text in constants and depth < 4:
        return resolve(constants[text], constants, depth + 1)
    if text == "true":
        return True
    if text == "false":
        return False
    if text.replace(" ", "") == INFINITY.replace(" ", ""):
        return math.inf
    if text.startswith("."):
        fields = []
        for part in split_top_level(text, ","):
            name, _, value = part.partition("=")
            fields.append((name.strip().lstrip("."), resolve(value, constants, depth + 1)))
        return Designated(fields=tuple(fields))
    literal = re.fullmatch(r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)[fFuUlL]*", text)
    if literal:
        digits = literal.group(1)
        return float(digits) if re.search(r"[.eE]", digits) else int(digits)
    member = re.fullmatch(r"\w+::(\w+)", text)
    if member:
        return EnumMember(name=member.group(1))
    return Unresolved(text)


def declared_category(head):
    if re.search(r"\bbool\b", head):
        return bool
    if re.search(r"\b(double|float)\b", head):
        return float
    if re.search(r"\b(int\d*_t|u?int|std::size_t|size_t|short|long)\b", head):
        return int
    return None


def base_type(head):
    tokens = head.split()
    return tokens[-2].split("<")[0] if len(tokens) >= 2 else ""


def recover(root, source):
    path = root / INCLUDE_ROOT / source.header
    text = strip_comments(path.read_text())
    members = brace_initialized_members(struct_body(text, source.struct))
    return members, declared_constants(text)


def check_source(root, bound, source, overrides, trail):
    members, constants = recover(root, source)
    assert len(members) >= source.minimum_fields, (
        f"{trail}{source.struct}: recovered {len(members)} fields from "
        f"{source.header}, expected at least {source.minimum_fields}"
    )

    for field, initializer, head in members:
        reason = source.reason(field)
        if reason:
            assert reason.strip(), f"{trail}{field} is unchecked with no reason"
            continue

        attribute = source.renamed(field)
        assert hasattr(bound, attribute), f"{trail}{attribute} is not exposed"
        actual = getattr(bound, attribute)

        expected = (
            dict(overrides)[field]
            if overrides and field in dict(overrides)
            else resolve(initializer, constants)
        )
        where = f"{trail}{attribute}"

        if isinstance(expected, Unresolved):
            pytest.fail(f"{where}: cannot resolve the committed initializer {expected.text!r}")
        elif isinstance(expected, Empty):
            if "std::optional" in head:
                assert actual is None, f"{where} is {actual!r}, header says default-constructed"
            else:
                nested = STRUCT_TO_BINDING[base_type(head)]
                check_entry(root, actual, nested, (), where + ".")
        elif isinstance(expected, Designated):
            nested = STRUCT_TO_BINDING[base_type(head)]
            check_entry(root, actual, nested, expected.fields, where + ".")
        elif isinstance(expected, EnumMember):
            assert actual.name == expected.name, f"{where} is {actual.name}, header says {expected.name}"
        else:
            category = declared_category(head)
            if category is float:
                expected = float(expected)
            assert actual == expected, f"{where} is {actual!r}, header says {expected!r}"
            if category is not None:
                assert type(actual) is category, f"{where} is a {type(actual).__name__}"


def check_entry(root, bound, binding_name, overrides=(), trail=""):
    for source in TABLE[binding_name].sources:
        check_source(root, bound, source, overrides, trail or binding_name + ".")


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


def rosenbrock(x):
    return float((1.0 - x[0]) ** 2 + 5.0 * (x[1] - x[0] ** 2) ** 2)


def unit_disk(x):
    return np.array([1.0 - x[0] ** 2 - x[1] ** 2])


def unit_disk_jacobian(x):
    return np.array([[-2.0 * x[0], -2.0 * x[1]]])


def feasible_start():
    return np.array([0.3, 0.3])


# Constructed with the problem arguments and NOTHING else: any option keyword
# here would make the snapshot report what the test asked for rather than what
# the library defaults to.
UNCONFIGURED = {
    "LbfgsbSolver": lambda: argmin.LbfgsbSolver(objective=rosenbrock, x0=feasible_start()),
    "BobyqaSolver": lambda: argmin.BobyqaSolver(objective=rosenbrock, x0=feasible_start()),
    "CobylaSolver": lambda: argmin.CobylaSolver(
        objective=rosenbrock,
        x0=feasible_start(),
        constraints=unit_disk,
        num_equality=0,
        num_inequality=1,
    ),
    "SlsqpSolver": lambda: argmin.SlsqpSolver(
        objective=rosenbrock,
        x0=feasible_start(),
        constraints=unit_disk,
        constraint_jacobian=unit_disk_jacobian,
        num_equality=0,
        num_inequality=1,
    ),
    "CmaesSolver": lambda: argmin.CmaesSolver(objective=rosenbrock, x0=feasible_start()),
}

POLICY_OPTION_TYPE = {
    "LbfgsbSolver": "LbfgsbOptions",
    "BobyqaSolver": "BobyqaOptions",
    "CobylaSolver": "CobylaOptions",
    "SlsqpSolver": "SlsqpOptions",
    "CmaesSolver": "CmaesOptions",
}


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


def bound_value_types():
    found = {}
    for name in argmin.__all__:
        value = getattr(argmin, name, None)
        if not inspect.isclass(value) or value.__module__ != argmin._argmin.__name__:
            continue
        if issubclass(value, BaseException) or hasattr(value, "__members__"):
            continue
        if any(
            not attribute.startswith("_") and callable(getattr(value, attribute, None))
            for attribute in dir(value)
        ):
            continue
        try:
            value()
        except TypeError:
            continue
        found[name] = value
    return found


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
