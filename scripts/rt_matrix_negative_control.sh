#!/usr/bin/env bash
# Prove that the real-time documentation drift check is able to fail.
#
# The published real-time safety matrix and the determinism page carry tables
# rendered from checked-in data files, and CI runs `rt_matrix.py check` to go
# red when a table and its data file disagree. That check is only worth
# something if it can go red. A check that cannot fail, trusted as evidence, is
# worse than no check at all: it launders an unverified claim as a verified one,
# and every claim resting on it inherits a confidence nobody earned.
#
# So the check is not assumed to discriminate. It is made to, on every run, for
# both documents:
#
#   1. an untouched document passes             -- the check is not simply red
#   2. a load-bearing mutation inside a
#      generated region fails                   -- the check is not simply green
#   3. the failure names the mutated row, and
#      only that row                            -- it discriminated rather than
#                                                 merely died
#   4. the working tree is never mutated        -- copies only; nothing to restore
#   5. the checker carries no bypass hook       -- nothing proven here was proven
#                                                 by a gate weakened to allow it
#
# The mutations flip a rendered evidence class, promote a non-"yes" reasoned
# exclusion to a yes-adjacent string, or corrupt a cited evidence name, always
# inside the generator's markers. Never trailing whitespace and never a region
# outside the markers: the checker does not compare either, so a control built on
# them would pass for the wrong reason and prove nothing at all.
#
# Property 3 is deliberately not a grep over all of stderr. The checker prints a
# unified diff, which quotes the mutated line verbatim -- so the row name appears
# in the output no matter which row was touched. The assertion reads the
# "rows differing" report alone, and demands it list exactly the mutated row.
#
#   scripts/rt_matrix_negative_control.sh
set -euo pipefail

cd "$(dirname "$0")/.."

CHECKER="scripts/rt_matrix.py"
MATRIX_DATA="scripts/rt_matrix_cells.json"
MATRIX_DOC="docs/rt-safety-matrix.md"
TIERS_DATA="scripts/rt_matrix_tiers.json"
TIERS_DOC="docs/determinism.md"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

failures=0
pass() { printf '  PASS  %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1"; failures=$((failures + 1)); }

# --- Property 1: an untouched document passes --------------------------------
#
# Without this the control proves noise rather than discrimination: a checker
# that failed on everything would satisfy property 2 while being useless.
prove_clean_passes() {
    local data="$1" doc="$2" label="$3"
    local copy="$WORK/clean.md" rc=0

    cp "$doc" "$copy"
    python3 "$CHECKER" check "$data" "$copy" >/dev/null 2>&1 || rc=$?

    if [ "$rc" -eq 0 ]; then
        pass "$label: the unmutated document passes the drift check"
    else
        fail "$label: the unmutated document does not pass (exit $rc); the control cannot distinguish drift from noise"
    fi
}

# --- Properties 2, 3 and 4: a load-bearing mutation goes red on its own row ---
#
# The mutation is applied to a copy. The tree is read, never written.
prove_mutation_fails() {
    local data="$1" doc="$2" region="$3" row="$4" needle="$5" replacement="$6" label="$7"
    local copy="$WORK/mutated.md" err="$WORK/stderr.txt" rc=0

    if ! python3 - "$doc" "$copy" "$region" "$row" "$needle" "$replacement" <<'PY'
import sys

src, dst, region, row, needle, replacement = sys.argv[1:7]
lines = open(src).read().split("\n")

begin, end = f"<!-- BEGIN GENERATED: {region} -->", f"<!-- END GENERATED: {region} -->"
if begin not in lines or end not in lines:
    sys.exit(f"the document declares no generated region named '{region}'")
i, j = lines.index(begin), lines.index(end)

# The row is located by its label and only inside the markers, so a mutation can
# never land on the hand-written prose the generator does not own.
hits = [k for k in range(i + 1, j)
        if lines[k].startswith(f"| {row} ") or lines[k].startswith(f"| `{row}` ")]
if len(hits) != 1:
    sys.exit(f"expected exactly one row '{row}' inside the region, found {len(hits)}")
k = hits[0]

if needle not in lines[k]:
    sys.exit(f"the row '{row}' no longer contains the text this control mutates: {needle!r}")
lines[k] = lines[k].replace(needle, replacement, 1)

open(dst, "w").write("\n".join(lines))
PY
    then
        fail "$label: the mutation could not be applied"
        return
    fi

    # A mutation that changed nothing would leave the checker green and the
    # control would report a green checker as a failure -- correct, but for an
    # unhelpful reason. Say which it was.
    if cmp -s "$doc" "$copy"; then
        fail "$label: the mutation changed no bytes; it cannot test anything"
        return
    fi

    python3 "$CHECKER" check "$data" "$copy" >/dev/null 2>"$err" || rc=$?

    if [ "$rc" -ne 1 ]; then
        fail "$label: a mutated cell exits $rc, not 1; the drift check does not catch this edit"
        return
    fi

    if python3 - "$err" "$row" <<'PY'
import re
import sys

err, expected = sys.argv[1], sys.argv[2]
reports = [ln for ln in open(err).read().split("\n") if "rows differing:" in ln]
if len(reports) != 1:
    sys.exit(f"expected exactly one 'rows differing' report, found {len(reports)}")

# Read the report alone. The unified diff below it quotes the mutated line, so a
# grep over the whole of stderr would match whatever row was touched and would
# assert nothing.
listed = re.findall(r"`([^`]+)`", reports[0].split("rows differing:", 1)[1])
if listed != [expected]:
    sys.exit(f"the failure names {listed}, not exactly ['{expected}']")
PY
    then
        pass "$label: exits 1 and names exactly \`$row\`"
    else
        fail "$label: exits 1, but does not name exactly \`$row\` -- a failure that does not discriminate could be a crash"
    fi
}

# --- Property 5: the checker carries no bypass hook ---------------------------
#
# The control proves the gate can fail. That proof is void if the gate carries a
# way to be told to pass -- then the property demonstrated here is a property of
# this invocation rather than of the gate CI runs.
#
# The scan reads the parsed source, not its text: it looks at identifiers the
# checker actually executes and at the command-line flags it accepts, and
# ignores comments and prose. A text scan would fire on any docstring that
# merely discusses a bypass, and a control that cried wolf would be turned off.
# The honest limit: a hook built from string comparisons this scan does not
# model would not be caught. It catches every shape a bypass has actually taken.
prove_no_bypass_hook() {
    if python3 - "$CHECKER" <<'PY'
import ast
import sys

path = sys.argv[1]
tree = ast.parse(open(path).read(), filename=path)

FORBIDDEN = ("getenv", "environ", "skip_check", "no_verify", "force_pass", "bypass")
FORBIDDEN_FLAGS = ("skip", "no-verify", "bypass", "force-pass")

found = []
for node in ast.walk(tree):
    names = []
    if isinstance(node, ast.Name):
        names = [node.id]
    elif isinstance(node, ast.Attribute):
        names = [node.attr]
    elif isinstance(node, ast.keyword) and node.arg:
        names = [node.arg]
    elif isinstance(node, ast.arg):
        names = [node.arg]
    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
        # Only flag-shaped literals. Prose never starts with a double dash, so
        # this closes the command-line hole without firing on an error message.
        if node.value.startswith("--"):
            flag = node.value.lower()
            found += [f"the command-line flag '{node.value}'"
                      for stem in FORBIDDEN_FLAGS if stem in flag]
        continue
    for name in names:
        found += [f"the name '{name}'" for stem in FORBIDDEN if stem in name.lower()]

if found:
    for item in sorted(set(found)):
        print(f"    {path} carries {item}", file=sys.stderr)
    sys.exit(1)
PY
    then
        pass "the drift check carries no hook that could have been told to pass"
    else
        fail "the drift check carries a bypass hook; nothing this control proves is a property of the gate CI runs"
    fi
}

echo "proving the real-time documentation drift check can fail:"

prove_clean_passes "$MATRIX_DATA" "$MATRIX_DOC" "safety matrix"
prove_clean_passes "$TIERS_DATA" "$TIERS_DOC" "determinism tiers"

# A cell's evidence class is the claim's strength. Downgrading a gated cell to
# argued is the edit a reader would most want caught -- and the reverse, an
# argued cell quietly promoted to gated, is the edit that would export a
# guarantee nothing stands behind.
prove_mutation_fails "$MATRIX_DATA" "$MATRIX_DOC" "drivers" "step_budget_solver" \
    '**yes** *(gated)*' 'yes *(argued)*' \
    "safety matrix: a gated cell hand-downgraded to argued"

prove_mutation_fails "$TIERS_DATA" "$TIERS_DOC" "tiers" "cross-instantiation numeric agreement" \
    '**yes** *(gated)*' 'yes *(argued)*' \
    "determinism tiers: a gated tier hand-downgraded to argued"

# An evidence name is the whole of a gated claim: the cell says a named check
# defends it, so a name that no longer matches the data file is a citation to
# something the reader cannot follow.
prove_mutation_fails "$MATRIX_DATA" "$MATRIX_DOC" "policies" "projected_gn" \
    'alloc_gate_projected_gn' 'alloc_gate_projected_gn_renamed_away' \
    "safety matrix: a cited gate name corrupted in place"

# A reasoned exclusion is a non-"yes" verdict with no gate behind it. The edit a
# reader would most want caught is one that hand-promotes it to a yes-adjacent
# string -- exporting an intrinsic bound the policy does not have. The render path
# for the new verdict must be defended like a downgraded class or a corrupted name.
prove_mutation_fails "$MATRIX_DATA" "$MATRIX_DOC" "policies" "lm" \
    'not intrinsically bounded *(reasoned exclusion)*' '**yes** *(gated)*' \
    "safety matrix: a reasoned-exclusion cell hand-promoted to a yes"

prove_mutation_fails "$TIERS_DATA" "$TIERS_DOC" "tiers" "cross-instantiation numeric agreement" \
    'fixed-N path reproduces the dynamic-N trajectory' \
    'fixed-N path reproduces the dynamic-N trajectry' \
    "determinism tiers: a cited test name corrupted in place"

prove_no_bypass_hook

echo
if [ "$failures" -ne 0 ]; then
    echo "the drift check is NOT proven able to fail: $failures of the control's properties do not hold"
    echo "until this passes, a green drift check is not evidence that the published tables match their data files"
    exit 1
fi

echo "the drift check discriminates: both documents pass untouched, and a mutated"
echo "cell goes red naming its own row, with no way to tell the checker to pass"
