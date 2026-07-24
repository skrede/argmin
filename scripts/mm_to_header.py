#!/usr/bin/env python3
"""Convert convex quadratic programs in QPS (MPS + QUADOBJ) form into committed
C++ data headers, offline, using only the Python standard library.

Why an offline converter instead of a QPS reader wired into the test build:
a QPS parser living in the test tree is a file-format dependency that is dead
weight the moment conversion is done, a fuzz surface on developer-supplied
input, and a second code path that can disagree with whatever produced the
numbers. The problems this project exercises do not change between checkouts,
so the honest representation is the converted data itself -- plain aggregate
arrays a reviewer can read at the commit -- with the parser kept out of the
build entirely. This mirrors the project's committed-baseline posture: the
generated headers are the artifact; this script is the tool that made them and
is never invoked by CMake or by any test.

The selection rule -- not a hardcoded roster -- is the commitment. A problem is
converted only if it survives, measured against the actual file:

    n <= 128                 (the fixed-N inline ceiling for double: an n x n
                              condensed-KKT member must fit Eigen's default
                              131072-byte stack-allocation limit, 8 * n^2)
    A is dense-representable  (n * m <= 100000 stored entries, so the dense
                              copy is trivially committable)

Any published-dimension roster is only a starting point; this script verifies
n and m against the parsed file and drops whatever fails the rule.

QPS is fixed-position MPS with a QUADOBJ section storing the lower triangle of
the objective Hessian P. The mapping to the solver's  l <= A x <= u  form:

    ROWS      one free (N) row is the linear objective; L/G/E rows are
              constraints.
    COLUMNS   entries in the objective row are q; entries in constraint rows
              are A.
    RHS       per-row right-hand side b.
    RANGES    turns a one-sided row two-sided (see row_bounds below).
    BOUNDS    variable bounds; the MPS default is 0 <= x, +inf above. Each
              variable with a finite bound contributes an identity row to A so
              box bounds live in the same  l <= A x <= u  system.
    QUADOBJ   lower triangle of P; the upper triangle is its mirror.

Usage:
    mm_to_header.py <qps_dir> <out_dir> [--optima optima.json]

<qps_dir>   directory scanned (non-recursively) for *.qps / *.QPS files.
<out_dir>   receives one <name>.h per surviving problem, plus an aggregating
            mm_problems.h and an X-macro mm_problems.inc for registration.
--optima    optional JSON object mapping problem name -> published optimal
            objective value; emitted into the header when present.
"""

import os
import re
import sys
import json
import argparse

INF = float("inf")


class QpsError(Exception):
    pass


def _tokens(line):
    # MPS is column-oriented, but every standard file is also whitespace-
    # separable; token splitting reads the real files and stays readable.
    return line.split()


def parse_qps(path):
    name = None
    obj_row = None
    row_type = {}
    row_order = []
    col_order = []
    q = {}
    a_entries = {}
    rhs = {}
    ranges = {}
    bnd_lo = {}
    bnd_up = {}
    bnd_seen = set()
    quad = {}
    section = None
    in_int = False

    with open(path, "r") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("*"):
                continue
            if not line[0].isspace():
                head = _tokens(line)
                key = head[0].upper()
                if key == "NAME":
                    name = head[1] if len(head) > 1 else os.path.splitext(
                        os.path.basename(path))[0]
                    section = "NAME"
                elif key in ("ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS",
                             "QUADOBJ", "QMATRIX", "QSECTION", "ENDATA"):
                    section = "QUADOBJ" if key in ("QMATRIX", "QSECTION") else key
                else:
                    section = key
                continue

            tok = _tokens(line)
            if section == "ROWS":
                rtype, rname = tok[0].upper(), tok[1]
                row_type[rname] = rtype
                if rtype == "N":
                    if obj_row is None:
                        obj_row = rname
                else:
                    row_order.append(rname)
            elif section == "COLUMNS":
                if len(tok) >= 3 and tok[1].upper() == "'MARKER'":
                    in_int = tok[2].upper().strip("'") == "INTORG"
                    continue
                col = tok[0]
                if col not in q:
                    q[col] = 0.0
                    col_order.append(col)
                pairs = tok[1:]
                for i in range(0, len(pairs) - 1, 2):
                    rname, val = pairs[i], float(pairs[i + 1])
                    if rname == obj_row:
                        q[col] += val
                    elif rname in row_type:
                        a_entries[(rname, col)] = a_entries.get((rname, col), 0.0) + val
            elif section == "RHS":
                pairs = tok[1:]
                for i in range(0, len(pairs) - 1, 2):
                    rhs[pairs[i]] = float(pairs[i + 1])
            elif section == "RANGES":
                pairs = tok[1:]
                for i in range(0, len(pairs) - 1, 2):
                    ranges[pairs[i]] = float(pairs[i + 1])
            elif section == "BOUNDS":
                btype = tok[0].upper()
                var = tok[2]
                bnd_seen.add(var)
                val = float(tok[3]) if len(tok) > 3 else 0.0
                if btype == "LO":
                    bnd_lo[var] = val
                elif btype == "UP":
                    bnd_up[var] = val
                elif btype == "FX":
                    bnd_lo[var] = val
                    bnd_up[var] = val
                elif btype == "FR":
                    bnd_lo[var] = -INF
                    bnd_up[var] = INF
                elif btype == "MI":
                    bnd_lo[var] = -INF
                elif btype == "PL":
                    bnd_up[var] = INF
                else:
                    raise QpsError("unsupported bound type %s in %s" % (btype, path))
            elif section == "QUADOBJ":
                ci, cj, val = tok[0], tok[1], float(tok[2])
                quad[(ci, cj)] = quad.get((ci, cj), 0.0) + val

    if obj_row is None:
        raise QpsError("no objective (N) row in %s" % path)
    if not col_order:
        raise QpsError("no columns in %s" % path)

    n = len(col_order)
    col_index = {c: j for j, c in enumerate(col_order)}

    p = [[0.0] * n for _ in range(n)]
    for (ci, cj), val in quad.items():
        i, j = col_index[ci], col_index[cj]
        p[i][j] = val
        p[j][i] = val

    qv = [q[c] for c in col_order]

    rows = []  # (a_row, l, u)
    for rname in row_order:
        a_row = [a_entries.get((rname, c), 0.0) for c in col_order]
        b = rhs.get(rname, 0.0)
        rng = ranges.get(rname)
        lo, up = row_bounds(row_type[rname], b, rng)
        rows.append((a_row, lo, up))

    for var in col_order:
        lo = bnd_lo.get(var, 0.0)
        up = bnd_up.get(var, INF)
        if lo == -INF and up == INF:
            continue
        e = [0.0] * n
        e[col_index[var]] = 1.0
        rows.append((e, lo, up))

    return {
        "name": name or os.path.splitext(os.path.basename(path))[0],
        "n": n,
        "P": p,
        "q": qv,
        "rows": rows,
    }


def row_bounds(rtype, b, rng):
    if rtype == "E":
        if rng is None:
            return b, b
        return (b, b + rng) if rng >= 0.0 else (b + rng, b)
    if rtype == "L":
        return (b - abs(rng) if rng is not None else -INF), b
    if rtype == "G":
        return b, (b + abs(rng) if rng is not None else INF)
    raise QpsError("unexpected row type %s" % rtype)


def survives_rule(problem):
    n = problem["n"]
    m = len(problem["rows"])
    return n <= 128 and n * m <= 100000


def sanitize(name):
    return re.sub(r"[^0-9A-Za-z]", "_", name).lower()


def fmt(value):
    if value == INF:
        return "inf"
    if value == -INF:
        return "-inf"
    return "%.17g" % value


def emit_header(problem, optima):
    ident = sanitize(problem["name"])
    n = problem["n"]
    rows = problem["rows"]
    m = len(rows)
    guard = "HPP_GUARD_ARGMIN_TESTS_UNIT_MM_DATA_%s_H" % ident.upper()

    p_flat = [fmt(problem["P"][i][j]) for i in range(n) for j in range(n)]
    q_flat = [fmt(v) for v in problem["q"]]
    a_flat = [fmt(rows[i][0][j]) for i in range(m) for j in range(n)]
    l_flat = [fmt(rows[i][1]) for i in range(m)]
    u_flat = [fmt(rows[i][2]) for i in range(m)]

    opt = optima.get(problem["name"])
    has_opt = "true" if opt is not None else "false"
    opt_val = fmt(opt) if opt is not None else "0.0"

    out = []
    out.append("#ifndef %s" % guard)
    out.append("#define %s" % guard)
    out.append("")
    out.append("// Generated by scripts/mm_to_header.py -- do not edit by hand.")
    out.append("// Convex QP  min 0.5 x^T P x + q^T x  s.t.  l <= A x <= u.")
    out.append("// P is stored dense row-major (symmetric); A is dense row-major.")
    out.append("")
    out.append("#include <array>")
    out.append("#include <limits>")
    out.append("")
    out.append("namespace argmin::mm_data::%s" % ident)
    out.append("{")
    out.append("inline constexpr double inf = std::numeric_limits<double>::infinity();")
    out.append("inline constexpr int n = %d;" % n)
    out.append("inline constexpr int m = %d;" % m)
    out.append("inline constexpr bool has_optimum = %s;" % has_opt)
    out.append("inline constexpr double optimum = %s;" % opt_val)
    out.append("inline constexpr char label[] = \"%s\";" % problem["name"])
    out.append(_array("P", n * n, p_flat))
    out.append(_array("q", n, q_flat))
    out.append(_array("A", m * n, a_flat))
    out.append(_array("l", m, l_flat))
    out.append(_array("u", m, u_flat))
    out.append("}")
    out.append("")
    out.append("#endif")
    out.append("")
    return ident, "\n".join(out)


def _array(sym, count, values):
    body = ", ".join(values)
    return "inline constexpr std::array<double, %d> %s = { %s };" % (count, sym, body)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("qps_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--optima", default=None)
    args = ap.parse_args(argv)

    optima = {}
    if args.optima and os.path.exists(args.optima):
        with open(args.optima, "r") as handle:
            optima = json.load(handle)

    os.makedirs(args.out_dir, exist_ok=True)
    sources = sorted(
        f for f in os.listdir(args.qps_dir) if f.lower().endswith(".qps"))
    if not sources:
        raise SystemExit("no .qps files under %s" % args.qps_dir)

    kept = []
    for fname in sources:
        problem = parse_qps(os.path.join(args.qps_dir, fname))
        if not survives_rule(problem):
            sys.stderr.write(
                "skip %s (n=%d m=%d fails n<=128 / n*m<=100000)\n"
                % (problem["name"], problem["n"], len(problem["rows"])))
            continue
        ident, text = emit_header(problem, optima)
        with open(os.path.join(args.out_dir, ident + ".h"), "w") as handle:
            handle.write(text)
        kept.append(ident)
        sys.stderr.write(
            "keep %s -> %s.h (n=%d m=%d)\n"
            % (problem["name"], ident, problem["n"], len(problem["rows"])))

    if not kept:
        raise SystemExit("no problem survived the selection rule")

    write_manifest(args.out_dir, kept)
    sys.stderr.write("converted %d problem(s)\n" % len(kept))
    return 0


def write_manifest(out_dir, idents):
    with open(os.path.join(out_dir, "mm_problems.inc"), "w") as handle:
        handle.write("// Generated by scripts/mm_to_header.py -- do not edit.\n")
        for ident in idents:
            handle.write("MM_PROBLEM(%s)\n" % ident)

    guard = "HPP_GUARD_ARGMIN_TESTS_UNIT_MM_DATA_MM_PROBLEMS_H"
    lines = ["#ifndef %s" % guard, "#define %s" % guard, ""]
    lines.append("// Generated by scripts/mm_to_header.py -- do not edit by hand.")
    lines.append("// Aggregates every converted problem header; pair with")
    lines.append("// mm_problems.inc for X-macro registration.")
    lines.append("")
    for ident in idents:
        lines.append("#include \"%s.h\"" % ident)
    lines.append("")
    lines.append("#endif")
    lines.append("")
    with open(os.path.join(out_dir, "mm_problems.h"), "w") as handle:
        handle.write("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
