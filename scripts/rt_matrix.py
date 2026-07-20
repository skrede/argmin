#!/usr/bin/env python3
"""Render claim tables from a checked-in data file, and resolve what they cite.

Reads a JSON file declaring, for every row, what each claim column asserts and
what proves it, and rewrites the marker-delimited table regions of a document
from it. Everything outside the markers -- the column legend, the scope
argument, the footnote bodies, the per-tier prose -- is hand-written and is
never touched: that analysis is what makes a document trustworthy and no script
can derive it.

Two documents publish through this renderer, under the two schemas below: the
real-time safety matrix, one row per module, and the tiered determinism claim,
one row per property. They share a schema validator and a resolver on purpose.
The determinism claim is the most citable guarantee the library makes, and a
claim published without traceable evidence is exactly the asserted prose the
matrix stopped being; there is no second, weaker path for it.

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

The schema makes a dishonest cell unrenderable, but it cannot tell whether the
artifact a cell names still exists -- a data file could cite a gate deleted last
quarter and render it as gated forever. That is what the resolve mode below is
for.

Usage:
    rt_matrix.py render  <data.json> <doc.md>   # rewrite the marked regions in place
    rt_matrix.py check   <data.json> <doc.md>   # render to memory, diff against the doc
    rt_matrix.py resolve <data.json> <build_dir> <workflows_dir>

Exit codes:
    0   the document matches the render / the render succeeded / every name resolves
    1   drift: a marked region disagrees with what the data file renders, or
        an evidence name does not resolve against live state
    2   usage error, malformed JSON, a schema violation, or the build tree
        cannot answer the question that was asked of it

Exit 1 and exit 2 are deliberately distinct: "the document has drifted" and "the
data file is unrenderable" are different failures with different fixes, and a
caller that conflates them cannot tell a stale document from a broken one. The
resolve mode leans on the same split even harder -- see below.

The resolve mode
----------------
Resolves every 'evidence' name in the data file against live state:

    ctest_test    the name appears in 'ctest -N' for <build_dir>
    ctest_label   the label appears in 'ctest --print-labels' for <build_dir>
    ci_job        the name matches a job 'name:' in a workflow under <workflows_dir>

and then cross-checks reachability: for every gated cell, at least one cited test
must be *selected by the ctest invocation of at least one cited job*. Existence is
not execution. A name resolving proves an artifact exists; it does not prove that
anything runs it. Twelve allocation gates existed and were cited while eleven of
them never registered in continuous integration -- resolution alone would have
called that green. So a gated cell whose named test no cited job selects fails.
The honest remedies are to fix the job or to render the cell as argued; rendering
it gated is not one of them.

Why the preconditions are asserted rather than assumed
------------------------------------------------------
The answer this mode gives is a property of *how the tree was configured*, not of
the source. Run against a tree configured without benchmarks, it would report the
allocation gates as dangling -- a false red. Run on a developer host with every
optional dependency present, it would report all of them fine while continuous
integration runs a fraction -- a false green. Same script, opposite verdicts, same
data file.

So the mode asserts what it needs and refuses to answer otherwise:

    - <build_dir> must be a configured tree (it must have a CTestTestfile.cmake),
    - the test binary must be *built*: test registration runs the binary at build
      time to enumerate its cases, so on a configured-not-built tree every case
      name is a '_NOT_BUILT-<hash>' placeholder, which would otherwise present as
      every case citation dangling,
    - a label a cited job selects by must be registered in the tree, when the tree
      can be seen to be configured in a way that would not register it.

All of these exit 2, never 1. The difference between "this name does not exist"
and "I cannot see this name from here" is the difference between a gate and a coin
flip: a false red trains readers to ignore the check, and a false green defeats its
purpose. Conflating them is how a validator becomes decorative.

The same discipline governs everything this mode does not understand. Job names
are templated, and only '${{ matrix.<key> }}' is expanded -- against the job's own
declared matrix list, so the substituted value is one the job can actually produce.
Any other expression in a job name, and any ctest option in a cited job whose
effect on test selection is not known here, exits 2 naming what was not understood.
There is deliberately no wildcard fallback: a pattern that turned the placeholder
into '.+' would happily resolve a job name that cannot exist, and a validator
trusted to be strict that silently is not is worse than none at all.

Known approximation, stated rather than hidden: a job's selection is evaluated
against <build_dir>, not against a tree configured the way that job configures it.
A job that runs a bare ctest therefore appears to select everything <build_dir>
registers. Cite a job whose configuration actually registers the test.

Data file format:
    {
      "schema": "rt-claims",                    # selects the row label and claim columns
      "tables": [
        {
          "id": "policies",                     # must match a marker pair in the doc
          "columns": [ ...the schema's claim columns, in order... ],
          "rows": [
            {
              "module": "kraft_slsqp",          # the row label, whatever the schema calls it
              "cells": {
                "allocation-free?": {
                  "verdict": "yes",             # yes | no | per-policy | not-claimed
                  "class": "gated",             # required iff verdict is yes; gated | argued
                  "evidence": [                 # required iff class is gated, non-empty
                    {"kind": "ctest_test",      # ctest_test | ctest_label | ci_job
                     "name": "sqp_alloc_gate_kraft",
                     "note": "zero mode, 0.00/step"}   # optional
                  ],
                  "rationale": "...",           # required iff class is argued; optional on a
                                                # non-claim; forbidden otherwise
                  "qualifier": "...",           # optional, renders inside the cell
                  "footnotes": ["restore"]      # optional, renders as [^restore]
                }
              }
            }
          ]
        }
      ]
    }

The schema is a closed set, declared here rather than in the data file, so a
document cannot quietly add, drop, or rename a claim column: the ratified matrix
column set stays ratified, and a tier table cannot borrow the matrix's schema to
render something the matrix does not publish.

Row order is the data file's order and is never sorted, so the same data file
always renders byte-identically.
"""

import os
import re
import sys
import json
import shlex
import shutil
import difflib
import subprocess

VERDICTS = ("yes", "no", "per-policy", "not-claimed", "excluded")
CLASSES = ("gated", "argued")
EVIDENCE_KINDS = ("ctest_test", "ctest_label", "ci_job")

# Test discovery runs several prefixed passes over the same executable, so a
# tagged case registers twice -- once bare, once under a prefix. Cited names are
# the bare case strings, so a prefixed registration is folded back onto its bare
# name rather than presented as a near-miss.
DISCOVERY_PREFIXES = ("oracle-pin: ", "robustness: ", "asan-move: ")

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

# The tiered determinism claim asks one question of each property, so it has one
# claim column. The property itself is the row label, and it is prose rather than
# an identifier, so it is not rendered as code.
TIER_COLUMNS = ["guaranteed?"]

CLASS_TEXT = {
    "gated": "**yes** *(gated)*",
    "argued": "yes *(argued)*",
}

SCHEMAS = {
    "rt-claims": {
        "label": "module",
        "label_code": True,
        "columns": COLUMNS,
        "verdict_text": {
            "no": "no",
            "per-policy": "per policy",
            "not-claimed": "not RT-claimed",
            # A non-yes verdict, deliberately not a class on a yes: a cell whose
            # iteration count is bounded only by a caller-set budget or single-step
            # structure has no intrinsic cap to gate, so rendering it as any yes
            # form would mint a prose-only yes the schema exists to forbid. Being
            # non-yes exempts it from the every-yes-needs-a-gate rule.
            "excluded": "not intrinsically bounded *(reasoned exclusion)*",
        },
    },
    "determinism-tiers": {
        "label": "claim",
        "label_code": False,
        "columns": TIER_COLUMNS,
        "verdict_text": {
            "no": "no",
            "per-policy": "per policy",
            # A tier that is not claimed is not a weak claim, and it must not
            # read as one next to three claims that are gated. It is bold for
            # the same reason a gated yes is: the reader's eye must land on the
            # distinction, not on the absence.
            "not-claimed": "**not claimed**",
        },
    },
}

_BEGIN = "<!-- BEGIN GENERATED: {} -->"
_END = "<!-- END GENERATED: {} -->"
_MODULE_RE = re.compile(r"^\|\s*`([^`]+)`\s*\|")
_LABEL_RE = re.compile(r"^\|\s*([^|]+?)\s*\|")


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


def _schema(data):
    return SCHEMAS[data["schema"]]


def _validate(data):
    problems = []

    def bad(msg):
        problems.append(msg)

    if not isinstance(data, dict) or not isinstance(data.get("tables"), list):
        raise SchemaError(["top level must be an object with a 'tables' list"])
    if not data["tables"]:
        raise SchemaError(["'tables' is empty; there is nothing to render"])
    if data.get("schema") not in SCHEMAS:
        raise SchemaError([f"top level must declare a 'schema' from {sorted(SCHEMAS)}; "
                           f"got {data.get('schema')!r}"])
    schema = _schema(data)
    columns = schema["columns"]

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

        if table.get("columns") != columns:
            bad(f"{where}: 'columns' must be exactly the '{data['schema']}' set, "
                f"in order: {columns}")
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
            missing = [c for c in columns if c not in cells]
            extra = [c for c in cells if c not in columns]
            if missing:
                bad(f"{where}/{module}: missing claim columns {missing}")
            if extra:
                bad(f"{where}/{module}: unknown claim columns {extra}")
            for col in columns:
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
        rationale = cell.get("rationale")
        if verdict == "excluded":
            # A reasoned exclusion is only honest if it names its reason: the cell
            # publishes "not intrinsically bounded" and the reader is owed why. The
            # reason renders in the evidence column, the same place a gated claim's
            # artifacts do, because that column answers why believe this cell.
            if not isinstance(rationale, str) or not rationale.strip():
                bad(f"{where}: a reasoned exclusion must carry a written rationale")
        elif rationale is not None:
            # A non-claim is the one remaining verdict a reader is entitled to a
            # reason for: "we do not guarantee this" invites "why not", and a claim
            # deliberately not made is indistinguishable from one nobody got
            # around to unless the file says which it is. A 'no' and a 'per-policy'
            # are measured verdicts and carry no reason here.
            if verdict != "not-claimed":
                bad(f"{where}: only an argued claim, a reasoned exclusion, or a "
                    f"non-claim carries a rationale")
            elif not isinstance(rationale, str) or not rationale.strip():
                bad(f"{where}: a non-claim's rationale must be a non-empty string")

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


def _cell_text(cell, schema):
    verdict = cell["verdict"]
    if verdict == "yes":
        text = CLASS_TEXT[cell["class"]]
    else:
        text = schema["verdict_text"][verdict]
    qualifier = cell.get("qualifier")
    if qualifier:
        text += f" ({qualifier})"
    for ref in cell.get("footnotes", []):
        text += f" [^{ref}]"
    return text


def _evidence_text(row, columns):
    parts = []
    for col in columns:
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
        elif cell["verdict"] in ("not-claimed", "excluded") and cell.get("rationale"):
            parts.append(cell["rationale"])
    unique = []
    for part in parts:
        if part not in unique:
            unique.append(part)
    return "; ".join(unique) if unique else "none named"


def _render_table(table, schema):
    columns = schema["columns"]
    header = "| " + schema["label"] + " | " + " | ".join(columns) + " | evidence |"
    rule = "|" + "---|" * (len(columns) + 2)
    lines = [header, rule]
    for row in table["rows"]:
        cells = [_cell_text(row["cells"][col], schema) for col in columns]
        label = f"`{row['module']}`" if schema["label_code"] else row["module"]
        lines.append("| " + label + " | " + " | ".join(cells)
                     + " | " + _evidence_text(row, columns) + " |")
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


def _module_of(line, schema):
    match = (_MODULE_RE if schema["label_code"] else _LABEL_RE).match(line)
    return match.group(1) if match else None


def _differing_modules(actual, rendered, schema):
    modules = []
    matcher = difflib.SequenceMatcher(None, actual, rendered)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        for line in list(actual[i1:i2]) + list(rendered[j1:j2]):
            module = _module_of(line, schema)
            if module and module not in modules:
                modules.append(module)
    return modules


def _regions(data):
    schema = _schema(data)
    return [(table["id"], _render_table(table, schema)) for table in data["tables"]]


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
        modules = _differing_modules(actual, rendered, _schema(data))
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


class CannotSee(Exception):
    """The tree cannot answer the question asked of it; guessing is refused."""

    def __init__(self, problems):
        super().__init__("; ".join(problems))
        self.problems = problems


_TEST_LINE_RE = re.compile(r"^\s*Test\s+#\d+:\s*(.+?)\s*$")


def _ctest(build_dir, args):
    """Run ctest with a list argv. Never shell=True: no data-file string is ever
    interpolated into a command line -- cited names are compared, not executed."""
    if shutil.which("ctest") is None:
        raise CannotSee(["ctest is not on PATH; cannot resolve names"])
    argv = ["ctest", "--test-dir", build_dir] + list(args)
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, shell=False)
    except OSError as exc:
        raise CannotSee([f"cannot run ctest in {build_dir}: {exc}"])
    return proc.stdout


def _ctest_names(build_dir, selectors=()):
    names = []
    for line in _ctest(build_dir, ["-N"] + list(selectors)).split("\n"):
        match = _TEST_LINE_RE.match(line)
        if match:
            names.append(match.group(1))
    return names


def _ctest_labels(build_dir):
    labels, started = [], False
    for line in _ctest(build_dir, ["--print-labels"]).split("\n"):
        if line.strip() == "All Labels:":
            started = True
            continue
        if started and line.strip():
            labels.append(line.strip())
    return labels


def _fold_prefixes(names):
    folded = set()
    for name in names:
        folded.add(name)
        for prefix in DISCOVERY_PREFIXES:
            if name.startswith(prefix):
                folded.add(name[len(prefix):])
    return folded


def _cache_value(build_dir, key):
    path = os.path.join(build_dir, "CMakeCache.txt")
    try:
        with open(path) as fh:
            for line in fh:
                if line.startswith(key + ":"):
                    return line.split("=", 1)[1].strip() if "=" in line else None
    except OSError:
        return None
    return None


def _assert_can_see(build_dir):
    """Assert the preconditions of an answer, or refuse to give one."""
    if not os.path.isdir(build_dir):
        raise CannotSee([f"no configured build tree at {build_dir}; cannot resolve names"])
    if not os.path.isfile(os.path.join(build_dir, "CTestTestfile.cmake")):
        raise CannotSee([f"{build_dir} is not a configured build tree "
                         f"(it has no CTestTestfile.cmake); cannot resolve names"])
    names = _ctest_names(build_dir)
    if not names:
        raise CannotSee([f"the configured tree at {build_dir} registers no tests at all; "
                         f"cannot resolve names"])
    unbuilt = sorted(n for n in names if "_NOT_BUILT-" in n)
    if unbuilt:
        raise CannotSee([f"the test binary is not built in {build_dir} (test registration "
                         f"reports the placeholder '{unbuilt[0]}'); case names cannot be "
                         f"enumerated, so no citation can be resolved from here"])
    return names


# ---------------------------------------------------------------------------
# Workflow inspection. No YAML parser is available -- the generator is
# stdlib-only by deliberate choice -- so this is textual, bounded, and loud
# about anything it does not understand.

_JOB_RE = re.compile(r"^  ([A-Za-z0-9_.-]+):\s*$")
_JOB_NAME_RE = re.compile(r"^    name:\s*(\S.*?)\s*$")
_EXPR_RE = re.compile(r"\$\{\{([^}]*)\}\}")
_MATRIX_KEY_RE = re.compile(r"^matrix\.([A-Za-z0-9_-]+)$")
_KEY_RE = re.compile(r"^-?\s*([A-Za-z0-9_-]+):\s*(.*)$")
_RUN_RE = re.compile(r"^(\s*)run:\s*(.*)$")


def _scalar(text):
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    return text


def _job_blocks(text):
    lines = text.split("\n")
    start = None
    for index, line in enumerate(lines):
        if line.rstrip() == "jobs:":
            start = index
            break
    if start is None:
        return []
    blocks, current = [], None
    for line in lines[start + 1:]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            break
        match = _JOB_RE.match(line)
        if indent == 2 and match:
            current = (match.group(1), [])
            blocks.append(current)
        elif current is not None:
            current[1].append(line)
    return blocks


def _matrix_values(block_lines):
    """strategy.matrix key -> declared values, from a plain list, a block list,
    or the keys of include entries. Values come from the declaration, so nothing
    outside it can ever be produced."""
    values, in_matrix, matrix_indent = {}, False, None
    pending_key, pending_indent = None, None

    def record(key, value):
        values.setdefault(key, [])
        if value not in values[key]:
            values[key].append(value)

    for line in block_lines:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if stripped == "matrix:":
            in_matrix, matrix_indent = True, indent
            continue
        if in_matrix and indent <= matrix_indent:
            in_matrix = False
        if not in_matrix:
            continue
        if pending_key is not None and indent > pending_indent and stripped.startswith("- ") \
                and ":" not in stripped:
            record(pending_key, _scalar(stripped[2:]))
            continue
        pending_key = None
        match = _KEY_RE.match(stripped)
        if not match:
            continue
        key, rest = match.group(1), match.group(2).strip()
        if key in ("include", "exclude"):
            continue
        if rest.startswith("[") and rest.endswith("]"):
            for item in rest[1:-1].split(","):
                if item.strip():
                    record(key, _scalar(item))
        elif rest:
            record(key, _scalar(rest))
        else:
            pending_key, pending_indent = key, indent
    return values


def _expand_job_name(name, matrix, where):
    exprs, seen = [], set()
    for raw in _EXPR_RE.findall(name):
        if raw not in seen:
            seen.add(raw)
            exprs.append(raw)
    results = [name]
    for raw in exprs:
        key_match = _MATRIX_KEY_RE.match(raw.strip())
        if not key_match:
            raise CannotSee([f"{where}: the job name uses the expression "
                             f"'${{{{{raw}}}}}', which this resolver does not expand; "
                             f"it will not guess a job name it cannot derive"])
        key = key_match.group(1)
        if key not in matrix:
            raise CannotSee([f"{where}: the job name uses '${{{{ matrix.{key} }}}}' but the "
                             f"job declares no matrix key '{key}'; cannot expand it"])
        token = "${{" + raw + "}}"
        results = [text.replace(token, value) for text in results for value in matrix[key]]
    return results


def _run_scripts(block_lines):
    scripts, index = [], 0
    while index < len(block_lines):
        match = _RUN_RE.match(block_lines[index])
        if not match:
            index += 1
            continue
        indent, rest = len(match.group(1)), match.group(2).strip()
        if rest in ("|", ">", "|-", ">-", "|+", ">+"):
            body, cursor = [], index + 1
            while cursor < len(block_lines):
                line = block_lines[cursor]
                if line.strip() and len(line) - len(line.lstrip()) <= indent:
                    break
                body.append(line.strip())
                cursor += 1
            scripts.append("\n".join(body))
            index = cursor
            continue
        if rest:
            scripts.append(rest)
        index += 1
    return scripts


def _collect_jobs(workflows_dir):
    """job name (expanded) -> list of run scripts."""
    if not os.path.isdir(workflows_dir):
        raise CannotSee([f"no workflow directory at {workflows_dir}; cannot resolve job names"])
    jobs = {}
    for entry in sorted(os.listdir(workflows_dir)):
        if not entry.endswith((".yml", ".yaml")):
            continue
        path = os.path.join(workflows_dir, entry)
        try:
            with open(path) as fh:
                text = fh.read()
        except OSError as exc:
            raise CannotSee([f"cannot read {path}: {exc}"])
        for job_id, block in _job_blocks(text):
            declared = None
            for line in block:
                match = _JOB_NAME_RE.match(line)
                if match:
                    declared = _scalar(match.group(1))
                    break
            # A job with no explicit name is displayed under its identifier.
            names = [job_id] if declared is None else _expand_job_name(
                declared, _matrix_values(block), f"{entry}: job '{job_id}'")
            for name in names:
                jobs.setdefault(name, []).extend(_run_scripts(block))
    return jobs


# ---------------------------------------------------------------------------
# ctest selection. Only the forms this repository actually uses are parsed;
# anything else is named and refused rather than assumed to be reachable.

_SELECTOR_FLAGS = {"-R": "regex", "--tests-regex": "regex",
                   "-L": "label", "--label-regex": "label"}
_LIST_ONLY_FLAGS = {"-N", "--show-only"}
_BENIGN_NOARG = {"--output-on-failure", "-V", "--verbose", "-VV", "--extra-verbose",
                 "--progress", "--stop-on-failure", "--force-new-ctest-process",
                 "--schedule-random", "--quiet", "-Q"}
_BENIGN_ONEARG = {"--test-dir", "-j", "--parallel", "-C", "--build-config", "--timeout",
                  "--repeat", "--output-junit", "--test-output-size-passed",
                  "--test-output-size-failed", "--output-log"}
_BENIGN_EQUALS = {"--no-tests", "--repeat", "--parallel", "--timeout", "--output-junit"}


def _ctest_selectors(script, where):
    """Selector argument lists for each ctest invocation in a shell script that
    actually runs tests. Listing invocations (-N) are skipped: they enumerate,
    they do not execute, and this check is about execution."""
    selections = []
    for line in script.split("\n"):
        if "ctest" not in line:
            continue
        try:
            tokens = shlex.split(line, comments=True)
        except ValueError as exc:
            raise CannotSee([f"{where}: cannot read the command line {line!r} ({exc}); "
                             f"cannot tell what it selects"])
        index = 0
        while index < len(tokens):
            token = tokens[index]
            index += 1
            if not (token == "ctest" or token.endswith("$(ctest") or token.endswith("`ctest")):
                continue
            args, listing_only = [], False
            while index < len(tokens):
                token = tokens[index]
                if token in ("&&", "||", ";", "|"):
                    break
                index += 1
                if token in _LIST_ONLY_FLAGS:
                    listing_only = True
                elif token in _SELECTOR_FLAGS:
                    if index >= len(tokens):
                        raise CannotSee([f"{where}: the ctest option {token} has no value"])
                    args.extend([token, tokens[index]])
                    index += 1
                elif token in _BENIGN_NOARG:
                    pass
                elif token in _BENIGN_ONEARG:
                    index += 1
                elif token.startswith("--") and "=" in token \
                        and token.split("=", 1)[0] in _BENIGN_EQUALS:
                    pass
                elif re.fullmatch(r"-j\d+", token):
                    pass
                elif token.startswith("-"):
                    raise CannotSee([f"{where}: the ctest option '{token}' is not one this "
                                     f"resolver knows the selection effect of; refusing to "
                                     f"assume it leaves the cited tests reachable"])
                else:
                    raise CannotSee([f"{where}: the ctest argument '{token}' is not one this "
                                     f"resolver understands; refusing to assume it leaves "
                                     f"the cited tests reachable"])
            if not listing_only:
                selections.append(args)
    return selections


def _selected(build_dir, args, cache):
    key = tuple(args)
    if key not in cache:
        cache[key] = _fold_prefixes(_ctest_names(build_dir, args))
    return cache[key]


def _assert_labels_visible(build_dir, args, labels, where):
    """A label a cited job selects by must be registered here. When the tree is
    visibly configured in a way that would not register it, that is a limit of
    this vantage point, not a missing name."""
    for index, token in enumerate(args):
        if _SELECTOR_FLAGS.get(token) != "label":
            continue
        parts = args[index + 1].split("|")
        if not all(re.fullmatch(r"[A-Za-z0-9_-]+", part) for part in parts):
            continue  # a real regex, not an alternation of literals; do not decompose it
        for part in parts:
            if part in labels:
                continue
            if _cache_value(build_dir, "ARGMIN_BUILD_BENCHMARKS") == "OFF":
                raise CannotSee([
                    f"{where}: selects by the ctest label '{part}', which is not registered "
                    f"in {build_dir} because that tree is configured without benchmarks, so "
                    f"the allocation gates do not register; cannot resolve their names from "
                    f"here"])


def _gated_cells(data):
    """Every gated cell, one at a time, identified by its own table, row, and
    column. Nothing is deduplicated across rows: two rows may legitimately cite
    the same test -- two determinism tiers do -- and each row's citation must be
    resolved on its own, or one row's evidence would silently stand in for
    another's and a rename would go red in only one of the two places it lies."""
    for table in data["tables"]:
        for row in table["rows"]:
            for column in _schema(data)["columns"]:
                cell = row["cells"][column]
                if cell.get("class") == "gated":
                    yield table["id"], row["module"], column, cell


def _job_selectors(data, jobs, build_dir, labels):
    """Parse every cited job's ctest invocations, and assert this tree can see
    what they select. This runs *before* any name is resolved, and that ordering
    is the whole point: a tree configured without the gates would otherwise report
    their names as dangling -- a false red indistinguishable from a real deletion.
    Anything not understood, or not visible from here, stops the run at exit 2."""
    selectors = {}
    for table_id, module, column, cell in _gated_cells(data):
        for item in cell["evidence"]:
            job = item["name"]
            if item["kind"] != "ci_job" or job in selectors or job not in jobs:
                continue
            where = f"job '{job}'"
            selectors[job] = []
            for script in jobs[job]:
                for args in _ctest_selectors(script, where):
                    _assert_labels_visible(build_dir, args, labels, where)
                    selectors[job].append(args)
    return selectors


def resolve(data, build_dir, workflows_dir):
    names = _assert_can_see(build_dir)
    tests = _fold_prefixes(names)
    labels = set(_ctest_labels(build_dir))
    jobs = _collect_jobs(workflows_dir)
    selectors = _job_selectors(data, jobs, build_dir, labels)

    failures = []
    counts = {"ctest_test": 0, "ctest_label": 0, "ci_job": 0}
    cache = {}

    for table_id, module, column, cell in _gated_cells(data):
        where = f"table '{table_id}', module `{module}`, column '{column}'"

        cited_tests, cited_jobs = [], []
        for item in cell["evidence"]:
            kind, name = item["kind"], item["name"]
            counts[kind] += 1
            if kind == "ctest_test":
                cited_tests.append(name)
                if name not in tests:
                    failures.append(f"{where}: cites the test '{name}', which no test "
                                    f"registered in {build_dir} is named")
            elif kind == "ctest_label":
                if name not in labels:
                    failures.append(f"{where}: cites the ctest label '{name}', which no test "
                                    f"registered in {build_dir} carries")
            else:
                cited_jobs.append(name)
                if name not in jobs:
                    failures.append(f"{where}: cites the continuous-integration job '{name}', "
                                    f"which no workflow defines")

        if not cited_tests:
            raise CannotSee([f"{where}: is gated but names no test, so there is no way to "
                             f"check that anything runs it"])
        if not cited_jobs:
            failures.append(f"{where}: cites {cited_tests[0]!r} but names no "
                            f"continuous-integration job that runs it; a test that exists "
                            f"but never runs proves nothing")
            continue
        if any(failure.startswith(where) for failure in failures):
            continue

        reachable = set()
        for job in cited_jobs:
            for args in selectors[job]:
                reachable |= _selected(build_dir, args, cache)
        if not any(name in reachable for name in cited_tests):
            failures.append(
                f"{where}: the named test(s) {', '.join(repr(n) for n in cited_tests)} exist, "
                f"but no cited job ({', '.join(cited_jobs)}) selects any of them -- existence "
                f"is not execution. Fix the job, or render the cell as argued")

    if failures:
        for failure in failures:
            print(f"RT MATRIX EVIDENCE FAILED: {failure}", file=sys.stderr)
        return 1

    width = max(len(kind) for kind in counts)
    for kind in ("ctest_test", "ctest_label", "ci_job"):
        print(f"  {kind.ljust(width)}  {counts[kind]:3d} resolved")
    print("rt matrix evidence resolved")
    return 0


def main(argv):
    mode = argv[1] if len(argv) > 1 else None
    usage = ("usage: rt_matrix.py render  <data.json> <doc.md>\n"
             "       rt_matrix.py check   <data.json> <doc.md>\n"
             "       rt_matrix.py resolve <data.json> <build_dir> <workflows_dir>")
    if mode in ("render", "check"):
        if len(argv) != 4:
            print(usage, file=sys.stderr)
            return 2
    elif mode == "resolve":
        if len(argv) != 5:
            print(usage, file=sys.stderr)
            return 2
    else:
        print(usage, file=sys.stderr)
        return 2

    data_path = argv[2]
    try:
        data = _load(data_path)
        _validate(data)
        if mode == "render":
            return render(data, argv[3])
        if mode == "check":
            return check(data, argv[3])
        return resolve(data, argv[3], argv[4])
    except SchemaError as exc:
        print(f"RT MATRIX SCHEMA ERROR: {data_path} cannot be rendered:", file=sys.stderr)
        for problem in exc.problems:
            print(f"  {problem}", file=sys.stderr)
        return 2
    except CannotSee as exc:
        print("RT MATRIX CANNOT RESOLVE: the build state cannot answer this; "
              "refusing to report a verdict:", file=sys.stderr)
        for problem in exc.problems:
            print(f"  {problem}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"RT MATRIX ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
