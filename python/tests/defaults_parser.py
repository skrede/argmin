"""Value resolution and the per-source header check that test_defaults_sweep runs."""

import re
import math
import pathlib
import dataclasses

import pytest

from defaults_table import TABLE, STRUCT_TO_BINDING
from header_text import (
    brace_initialized_members,
    declared_constants,
    matching_brace,
    split_top_level,
    strip_comments,
    struct_body,
)

INCLUDE_ROOT = pathlib.Path("lib") / "argmin" / "include" / "argmin"

INFINITY = re.compile(r"std::numeric_limits<(?:double|Scalar|scalar_type)>::infinity\(\)")

# A per-scalar table reference and a scalar cast both resolve to the double row,
# which is what the double-only binding exposes; the float row is never read.
TABLE_REFERENCE = re.compile(r"(\w+)<(?:Scalar|scalar_type|S|double)>::(\w+)")
SCALAR_CAST = re.compile(r"(?:Scalar|scalar_type|S)\((.*)\)")
BRACE_CAST = re.compile(r"std::(?:size_t|u?int(?:8|16|32|64)_t)\{(.*)\}")
# The binding instantiates the trust-region policy in its accurate mode, so a
# default chosen on the mode resolves to the branch taken when it is not fast.
MODE_CONDITIONAL = re.compile(r"\(\s*Mode\s*==\s*sqp_mode::fast\s*\)\s*\?(.*)")


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



def double_row_constants(table):
    root = pathlib.Path(__file__).resolve().parents[2] / INCLUDE_ROOT
    specialization = re.compile(r"\bstruct\s+" + re.escape(table) + r"<double>\s*\{")
    for path in sorted(root.rglob("*.h")):
        text = strip_comments(path.read_text())
        match = specialization.search(text)
        if match:
            opening = match.end() - 1
            return declared_constants(text[opening + 1 : matching_brace(text, opening)])
    return None


def false_branch(rest):
    depth = 0
    for index, character in enumerate(rest):
        depth += (character in "({") - (character in ")}")
        neighbors = rest[index - 1 : index] + rest[index + 1 : index + 2]
        if character == ":" and depth == 0 and ":" not in neighbors:
            return rest[index + 1 :]
    return None


def unwrapped(text):
    for form in (SCALAR_CAST, BRACE_CAST):
        match = form.fullmatch(text)
        if match:
            return match.group(1)
    conditional = MODE_CONDITIONAL.fullmatch(text)
    return false_branch(conditional.group(1)) if conditional else None


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
    if INFINITY.fullmatch(text.replace(" ", "")):
        return math.inf
    inner = unwrapped(text)
    if inner is not None and depth < 4:
        return resolve(inner, constants, depth + 1)
    reference = TABLE_REFERENCE.fullmatch(text)
    if reference and depth < 4:
        table, member = reference.groups()
        row = double_row_constants(table)
        if row is not None and member in row:
            return resolve(row[member], row, depth + 1)
        return Unresolved(text)
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
    member = re.fullmatch(r"(?:\w+::)+(\w+)", text)
    if member:
        return EnumMember(name=member.group(1))
    return Unresolved(text)


def declared_category(head):
    if re.search(r"\bbool\b", head):
        return bool
    if re.search(r"\b(double|float|Scalar|scalar_type|S)\b", head):
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
