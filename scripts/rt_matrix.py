#!/usr/bin/env python3
"""Render the real-time safety matrix tables from a checked-in cell data file.

Reads a JSON file declaring, for every driver and policy, what each real-time
column claims and what proves it, and rewrites the marker-delimited table
regions of the matrix document from it. Everything outside the markers -- the
column legend, the scope argument, the footnote bodies -- is hand-written prose
and is never touched: that analysis is what makes the document trustworthy and
no script can derive it.

Why a data file plus a renderer rather than hand-maintained prose:
a "yes" in this table is a real-time guarantee that a downstream worst-case
execution-time budget is entitled to rely on, so a "yes" whose proving artifact
does not exist exports a false guarantee. Hand-maintained cells drift exactly
that way -- they came to cite a continuous-integration job name that resolves to
nothing, and gates that were registered but never selected by any job. Prose
cannot be checked; data can. The schema below makes the dangerous state
unrepresentable rather than merely discouraged:

    - a cell claiming "yes" must declare an evidence class; there is no way to
      spell an unqualified yes,
    - a "gated" cell must name at least one artifact,
    - an "argued" cell must carry a written rationale, which is published in the
      evidence column next to the claim rather than hidden in a comment.

So a claim can be weakened, but it cannot be made silently. The renderer refuses
to emit a document from a data file that tries.

Usage:
    rt_matrix.py render <cells.json> <doc.md>   # rewrite the marked regions in place
    rt_matrix.py check  <cells.json> <doc.md>   # render to memory, diff against the doc

Exit codes:
    0   the document matches the render / the render succeeded
    1   drift: a marked region disagrees with what the data file renders
    2   usage error, malformed JSON, or a schema violation

Exit 1 and exit 2 are deliberately distinct: "the document has drifted" and "the
data file is unrenderable" are different failures with different fixes, and a
caller that conflates them cannot tell a stale document from a broken one.

Data file format:
    {
      "tables": [
        {
          "id": "policies",                     # must match a marker pair in the doc
          "columns": [ ...the ratified claim columns, in order... ],
          "rows": [
            {
              "module": "kraft_slsqp",
              "cells": {
                "allocation-free?": {
                  "verdict": "yes",             # yes | no | per-policy | not-claimed
                  "class": "gated",             # required iff verdict is yes; gated | argued
                  "evidence": [                 # required iff class is gated, non-empty
                    {"kind": "ctest_test",      # ctest_test | ctest_label | ci_job
                     "name": "sqp_alloc_gate_kraft",
                     "note": "zero mode, 0.00/step"}   # optional
                  ],
                  "rationale": "...",           # required iff class is argued, non-empty
                  "qualifier": "...",           # optional, renders inside the cell
                  "footnotes": ["restore"]      # optional, renders as [^restore]
                }
              }
            }
          ]
        }
      ]
    }

Row order is the data file's order and is never sorted, so the same data file
always renders byte-identically.
"""

import re
import sys
import json
import difflib

VERDICTS = ("yes", "no", "per-policy", "not-claimed")
CLASSES = ("gated", "argued")
EVIDENCE_KINDS = ("ctest_test", "ctest_label", "ci_job")

# The claim columns ratified with the downstream stack, in order. Rows publish
# against this set, so a table that adds or drops one stops being drop-in and is
# rejected rather than rendered.
COLUMNS = [
    "allocation-free?",
    "bounded-iterations?",
    "wall-clock-free?",
    "exceptions-off-clean?",
    "deterministic(seeded)?",
]

VERDICT_TEXT = {
    "no": "no",
    "per-policy": "per policy",
    "not-claimed": "not RT-claimed",
}

CLASS_TEXT = {
    "gated": "**yes** *(gated)*",
    "argued": "yes *(argued)*",
}

_BEGIN = "<!-- BEGIN GENERATED: {} -->"
_END = "<!-- END GENERATED: {} -->"
_MODULE_RE = re.compile(r"^\|\s*`([^`]+)`\s*\|")


class SchemaError(Exception):
    """The data file cannot be rendered as written."""

    def __init__(self, problems):
        super().__init__("; ".join(problems))
        self.problems = problems


def _load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except OSError as exc:
        raise SchemaError([f"cannot read {path}: {exc}"])
    except json.JSONDecodeError as exc:
        raise SchemaError([f"{path} is not valid JSON: {exc}"])


def _validate(data):
    problems = []

    def bad(msg):
        problems.append(msg)

    if not isinstance(data, dict) or not isinstance(data.get("tables"), list):
        raise SchemaError(["top level must be an object with a 'tables' list"])
    if not data["tables"]:
        raise SchemaError(["'tables' is empty; there is nothing to render"])

    seen_ids = []
    for t_index, table in enumerate(data["tables"]):
        where = f"table[{t_index}]"
        if not isinstance(table, dict):
            bad(f"{where}: must be an object")
            continue
        tid = table.get("id")
        if not isinstance(tid, str) or not tid:
            bad(f"{where}: missing a string 'id'")
            continue
        where = f"table '{tid}'"
        if tid in seen_ids:
            bad(f"{where}: duplicate table id")
        seen_ids.append(tid)

        if table.get("columns") != COLUMNS:
            bad(f"{where}: 'columns' must be exactly the ratified set, in order: {COLUMNS}")
        if not isinstance(table.get("rows"), list) or not table["rows"]:
            bad(f"{where}: 'rows' must be a non-empty list")
            continue

        for row in table["rows"]:
            if not isinstance(row, dict):
                bad(f"{where}: a row is not an object")
                continue
            module = row.get("module")
            if not isinstance(module, str) or not module:
                bad(f"{where}: a row is missing a string 'module'")
                continue
            cells = row.get("cells")
            if not isinstance(cells, dict):
                bad(f"{where}/{module}: missing a 'cells' object")
                continue
            missing = [c for c in COLUMNS if c not in cells]
            extra = [c for c in cells if c not in COLUMNS]
            if missing:
                bad(f"{where}/{module}: missing claim columns {missing}")
            if extra:
                bad(f"{where}/{module}: unknown claim columns {extra}")
            for col in COLUMNS:
                if col in cells:
                    _validate_cell(cells[col], f"{where}/{module}/{col}", bad)

    if problems:
        raise SchemaError(problems)


def _validate_cell(cell, where, bad):
    if not isinstance(cell, dict):
        bad(f"{where}: must be an object")
        return

    known = {"verdict", "class", "evidence", "rationale", "qualifier", "footnotes"}
    unknown = sorted(set(cell) - known)
    if unknown:
        bad(f"{where}: unknown fields {unknown}")

    verdict = cell.get("verdict")
    if verdict not in VERDICTS:
        bad(f"{where}: 'verdict' must be one of {list(VERDICTS)}, got {verdict!r}")
        return

    cls = cell.get("class")
    if verdict == "yes":
        # The load-bearing rule: an unqualified yes is unrenderable.
        if cls is None:
            bad(f"{where}: a 'yes' must declare an evidence class; there is no unqualified yes")
            return
        if cls not in CLASSES:
            bad(f"{where}: 'class' must be one of {list(CLASSES)}, got {cls!r}")
            return
    elif cls is not None:
        bad(f"{where}: only a 'yes' carries a class; '{verdict}' must not have one")
        return

    if cls == "gated":
        evidence = cell.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            bad(f"{where}: a gated claim must name at least one artifact")
        else:
            for item in evidence:
                if not isinstance(item, dict):
                    bad(f"{where}: an evidence entry is not an object")
                    continue
                e_unknown = sorted(set(item) - {"kind", "name", "note"})
                if e_unknown:
                    bad(f"{where}: unknown evidence fields {e_unknown}")
                if item.get("kind") not in EVIDENCE_KINDS:
                    bad(f"{where}: evidence 'kind' must be one of {list(EVIDENCE_KINDS)}, "
                        f"got {item.get('kind')!r}")
                if not isinstance(item.get("name"), str) or not item.get("name"):
                    bad(f"{where}: an evidence entry is missing a non-empty 'name'")
                if "note" in item and (not isinstance(item["note"], str) or not item["note"]):
                    bad(f"{where}: an evidence 'note' must be a non-empty string")
        if cell.get("rationale") is not None:
            bad(f"{where}: a gated claim cites artifacts, not a rationale")
    elif cls == "argued":
        rationale = cell.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            bad(f"{where}: an argued claim must carry a written rationale")
        if cell.get("evidence") is not None:
            bad(f"{where}: an argued claim names no artifact; drop 'evidence' or class it gated")
    else:
        if cell.get("evidence") is not None:
            bad(f"{where}: only a gated claim carries evidence")
        if cell.get("rationale") is not None:
            bad(f"{where}: only an argued claim carries a rationale")

    qualifier = cell.get("qualifier")
    if qualifier is not None and (not isinstance(qualifier, str) or not qualifier.strip()):
        bad(f"{where}: 'qualifier' must be a non-empty string")

    footnotes = cell.get("footnotes")
    if footnotes is not None:
        if not isinstance(footnotes, list) or not footnotes:
            bad(f"{where}: 'footnotes' must be a non-empty list")
        else:
            for ref in footnotes:
                if not isinstance(ref, str) or not ref:
                    bad(f"{where}: a footnote reference must be a non-empty string")


def _cell_text(cell):
    verdict = cell["verdict"]
    if verdict == "yes":
        text = CLASS_TEXT[cell["class"]]
    else:
        text = VERDICT_TEXT[verdict]
    qualifier = cell.get("qualifier")
    if qualifier:
        text += f" ({qualifier})"
    for ref in cell.get("footnotes", []):
        text += f" [^{ref}]"
    return text


def _evidence_text(row):
    parts = []
    for col in COLUMNS:
        cell = row["cells"][col]
        cls = cell.get("class")
        if cls == "gated":
            for item in cell["evidence"]:
                name = item["name"]
                if item.get("note"):
                    name += f" ({item['note']})"
                parts.append(name)
        elif cls == "argued":
            parts.append(cell["rationale"])
    unique = []
    for part in parts:
        if part not in unique:
            unique.append(part)
    return "; ".join(unique) if unique else "none named"


def _render_table(table):
    header = "| module | " + " | ".join(COLUMNS) + " | evidence |"
    rule = "|" + "---|" * (len(COLUMNS) + 2)
    lines = [header, rule]
    for row in table["rows"]:
        cells = [_cell_text(row["cells"][col]) for col in COLUMNS]
        lines.append("| `" + row["module"] + "` | " + " | ".join(cells)
                     + " | " + _evidence_text(row) + " |")
    return lines


def _region_bounds(doc_lines, region):
    begin, end = _BEGIN.format(region), _END.format(region)
    try:
        i = doc_lines.index(begin)
    except ValueError:
        raise SchemaError([f"the document has no '{begin}' marker; "
                           f"a table the data file declares would go unrendered"])
    try:
        j = doc_lines.index(end, i + 1)
    except ValueError:
        raise SchemaError([f"the document has no '{end}' marker after its opening marker"])
    return i, j


def _module_of(line):
    match = _MODULE_RE.match(line)
    return match.group(1) if match else None


def _differing_modules(actual, rendered):
    modules = []
    matcher = difflib.SequenceMatcher(None, actual, rendered)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        for line in list(actual[i1:i2]) + list(rendered[j1:j2]):
            module = _module_of(line)
            if module and module not in modules:
                modules.append(module)
    return modules


def _regions(data):
    return [(table["id"], _render_table(table)) for table in data["tables"]]


def render(data, doc_path):
    with open(doc_path) as fh:
        text = fh.read()
    lines = text.split("\n")
    for region, rendered in _regions(data):
        i, j = _region_bounds(lines, region)
        lines[i + 1:j] = rendered
    with open(doc_path, "w") as fh:
        fh.write("\n".join(lines))
    return 0


def check(data, doc_path):
    with open(doc_path) as fh:
        lines = fh.read().split("\n")
    drifted = []
    for region, rendered in _regions(data):
        i, j = _region_bounds(lines, region)
        actual = lines[i + 1:j]
        if actual != rendered:
            drifted.append((region, actual, rendered))

    if not drifted:
        print("rt matrix check passed")
        return 0

    for region, actual, rendered in drifted:
        modules = _differing_modules(actual, rendered)
        where = ", ".join(f"`{m}`" for m in modules) if modules else "(table structure)"
        print(f"RT MATRIX DRIFT: table '{region}' does not match the data file.",
              file=sys.stderr)
        print(f"RT MATRIX DRIFT: table '{region}': rows differing: {where}",
              file=sys.stderr)
        diff = difflib.unified_diff(actual, rendered,
                                    fromfile=f"{doc_path} ({region})",
                                    tofile=f"rendered ({region})",
                                    lineterm="")
        for line in diff:
            print(f"  {line}", file=sys.stderr)
    print("RT MATRIX DRIFT: re-run 'rt_matrix.py render' and commit the result, "
          "or correct the data file if the document is right.", file=sys.stderr)
    return 1


def main(argv):
    if len(argv) != 4 or argv[1] not in ("render", "check"):
        print("usage: rt_matrix.py render <cells.json> <doc.md>\n"
              "       rt_matrix.py check  <cells.json> <doc.md>", file=sys.stderr)
        return 2
    mode, cells_path, doc_path = argv[1], argv[2], argv[3]
    try:
        data = _load(cells_path)
        _validate(data)
        if mode == "render":
            return render(data, doc_path)
        return check(data, doc_path)
    except SchemaError as exc:
        print(f"RT MATRIX SCHEMA ERROR: {cells_path} cannot be rendered:", file=sys.stderr)
        for problem in exc.problems:
            print(f"  {problem}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"RT MATRIX ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
