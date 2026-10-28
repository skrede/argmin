#!/usr/bin/env python3
"""The rules EXCEPTIONS.md encodes: which files it governs, and what makes a row valid."""

import re
from dataclasses import dataclass, field
from pathlib import Path

CEILING = 200
GROWTH_NUMERATOR = 11
GROWTH_DENOMINATOR = 10
REGISTRY_NAME = "EXCEPTIONS.md"
SCOPE_PREFIX = "lib/"
EXTENSIONS = frozenset({".h", ".hh", ".hpp", ".hxx", ".inl", ".ipp", ".c", ".cc", ".cpp", ".cxx"})
FILES_SECTION = "Files"
FUNCTIONS_SECTION = "Functions"

EXIT_UNLISTED = 1
EXIT_STALE = 3
EXIT_GROWTH = 4
EXIT_REGISTRY = 5
EXIT_NO_INPUT = 6
EXIT_INFLATED = 7
EXIT_BASELINE_GREW = 8

HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*$")
SEPARATOR = re.compile(r"^\|(?:\s*:?-+:?\s*\|)+$")
COUNT_IN_REASON = re.compile(r"\b\d+\s*(?:lines?|-line)\b")
SPLIT_HARM = re.compile(r"split|decompos|separat|scatter", re.IGNORECASE)
FUTURE_WORK = re.compile(r"\b(?:planned|will be|to be split|todo|for now|eventually|later)\b",
                         re.IGNORECASE)


@dataclass
class Register:
    files: dict[str, int] = field(default_factory=dict)
    functions: dict[tuple[str, str], list[int]] = field(default_factory=dict)
    faults: list[str] = field(default_factory=list)


def in_scope(path: str) -> bool:
    return path.startswith(SCOPE_PREFIX) and Path(path).suffix in EXTENSIONS


def allowance(recorded: int) -> int:
    return (recorded * GROWTH_NUMERATOR + GROWTH_DENOMINATOR - 1) // GROWTH_DENOMINATOR


def registry_rows(lines: list[str]):
    """Every table line that is neither a separator nor a header is a row and must parse."""
    section = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        heading = HEADING.match(stripped)
        if heading:
            section = heading.group(1)
            continue
        if not stripped.startswith("|") or SEPARATOR.match(stripped):
            continue
        following = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if SEPARATOR.match(following):
            continue
        cells = [cell.strip().strip("`").strip() for cell in stripped.strip("|").split("|")]
        yield index + 1, section, cells


def reason_fault(reason: str, counts: set[str], where: str) -> str | None:
    if not reason:
        return f"{where}: the reason is empty, which is a silent exception with extra steps"
    if COUNT_IN_REASON.search(reason) or any(re.search(rf"\b{n}\b", reason) for n in counts):
        return f"{where}: the reason restates the unit's line count instead of justifying it"
    if not SPLIT_HARM.search(reason):
        return f"{where}: the reason must name what splitting this unit would harm"
    if FUTURE_WORK.search(reason):
        return f"{where}: the reason promises future work; this registers decisions already taken"
    return None


def file_row_fault(cells: list[str], where: str, counts: dict[str, int]) -> str | None:
    if len(cells) < 3 or not cells[0]:
        return f"{where}: a file row needs a path, a recorded line count and a reason"
    if not in_scope(cells[0]):
        return f"{where}: the path is outside the checked scope"
    if not cells[1].isdigit():
        return f"{where}: the recorded count column is not a number"
    measured = counts.get(cells[0])
    known = {cells[1]} | ({str(measured)} if measured is not None else set())
    return reason_fault(cells[2], known, where)


def function_row_fault(cells: list[str], where: str) -> str | None:
    if len(cells) < 4 or not cells[0] or not cells[1]:
        return f"{where}: a function row needs a path, a function, a recorded line count and a reason"
    if not in_scope(cells[0]):
        return f"{where}: the path is outside the checked scope"
    if not cells[2].isdigit():
        return f"{where}: the recorded count column is not a number"
    return reason_fault(cells[3], {cells[2]}, where)


def add_file_row(register: Register, cells: list[str], where: str, counts: dict[str, int]) -> None:
    fault = file_row_fault(cells, where, counts)
    if fault is not None:
        register.faults.append(fault)
    elif cells[0] in register.files:
        register.faults.append(f"{where}: the path is registered twice")
    else:
        register.files[cells[0]] = int(cells[1])


def add_function_row(register: Register, cells: list[str], where: str) -> None:
    fault = function_row_fault(cells, where)
    if fault is not None:
        register.faults.append(fault)
    else:
        register.functions.setdefault((cells[0], cells[1]), []).append(int(cells[2]))


def parse_registry(text: str, counts: dict[str, int]) -> Register:
    register = Register()
    for number, section, cells in registry_rows(text.splitlines()):
        where = f"{REGISTRY_NAME} line {number}, {cells[0] or '<no path>'}"
        if section == FILES_SECTION:
            add_file_row(register, cells, where, counts)
        elif section == FUNCTIONS_SECTION:
            add_function_row(register, cells, where)
        else:
            register.faults.append(f"{where}: a table row under the heading {section!r}, which is "
                                   f"neither {FILES_SECTION!r} nor {FUNCTIONS_SECTION!r}")
    return register
