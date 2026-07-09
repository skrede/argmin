#!/usr/bin/env python3
"""Validate disabled-default sweep evidence before defaults or registers move."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
from pathlib import Path


DEFAULT_REQUIRED_ITEMS = [
    "soc",
    "penalty_factor",
    "fast_mode_retirement",
    "multiplier_reest_every_k",
    "filter_default_grids",
    "nlopt_cobyla",
]

REQUIRED_FIELDS = [
    "item",
    "grid_id",
    "selected_value",
    "old_value",
    "correctness_witness",
    "provenance_id",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", newline="") as f:
            data_lines = [
                line for line in f if line.strip() and not line.lstrip().startswith("#")
            ]
    except OSError as exc:
        raise ValueError(f"{path}: cannot read: {exc}") from exc
    if not data_lines:
        raise ValueError(f"{path}: missing CSV header")
    reader = csv.DictReader(data_lines)
    if reader.fieldnames is None:
        raise ValueError(f"{path}: missing CSV header")
    missing = [field for field in REQUIRED_FIELDS if field not in reader.fieldnames]
    if missing:
        raise ValueError(f"{path}: missing required header(s): {', '.join(missing)}")
    return [dict(row) for row in reader]


def require_present(row: dict[str, str], field: str, context: str) -> None:
    if not row.get(field, "").strip():
        raise ValueError(f"{context}: missing {field}")


def parse_finite_number(value: str, context: str) -> None:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{context}: selected_value is not numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{context}: selected_value must be finite")


def row_expects_numeric(row: dict[str, str]) -> bool:
    marker = row.get("selected_numeric", "").strip().lower()
    if marker in {"1", "true", "yes"}:
        return True
    if marker in {"0", "false", "no"}:
        return False
    item = row.get("item", "")
    return item in {"soc", "penalty_factor", "multiplier_reest_every_k"}


def load_publish_summary_evidence(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("solver") != "nlopt_cobyla":
                    continue
                disposition = row.get("row_disposition", "")
                if disposition == "included":
                    return True
                if disposition == "excluded" and row.get("exclusion_reason", ""):
                    return True
    except OSError as exc:
        raise ValueError(f"{path}: cannot read publish summary: {exc}") from exc
    return False


def validate_register(path: Path,
                      required_items: list[str],
                      publish_summary: Path | None = None) -> None:
    rows = read_csv_rows(path)
    seen: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows, start=2):
        item = row.get("item", "").strip()
        context = f"{path}:{index}:{item or '<missing item>'}"
        require_present(row, "item", context)
        require_present(row, "grid_id", context)
        require_present(row, "selected_value", context)
        require_present(row, "old_value", context)
        require_present(row, "correctness_witness", context)
        require_present(row, "provenance_id", context)
        if item in seen and item in required_items:
            raise ValueError(f"{context}: duplicate required item {item!r}")
        if row_expects_numeric(row):
            parse_finite_number(row["selected_value"], context)
        seen[item] = row

    missing = [item for item in required_items if item not in seen]
    if "nlopt_cobyla" in missing and load_publish_summary_evidence(publish_summary):
        missing.remove("nlopt_cobyla")
    if missing:
        raise ValueError(f"{path}: missing required item(s): {', '.join(missing)}")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def valid_register_text(include_filter: bool = True,
                        include_nlopt: bool = True,
                        duplicate_soc: bool = False,
                        missing_provenance: bool = False) -> str:
    rows = [
        ["soc", "penalty_factor_x_soc_max_iterations", "0", "0",
         "reference gate passed", "prov"],
        ["penalty_factor", "penalty_factor", "0.01", "0",
         "reference gate passed", "prov"],
        ["fast_mode_retirement", "mode", "retire_fast", "keep_fast",
         "fast row rejected when correctness failed", "prov"],
        ["multiplier_reest_every_k", "multiplier_reest_every_k", "1", "1",
         "correctness_ok emitted for every row", "prov"],
    ]
    if include_filter:
        rows.append([
            "filter_default_grids", "filter_envelope+filter_trsqp", "1e-5;1e-5",
            "1e-5;1e-5", "both filter grids passed correctness gates", "prov",
        ])
    if include_nlopt:
        rows.append([
            "nlopt_cobyla", "publication_summary", "included", "disabled",
            "publication summary contains nlopt_cobyla evidence", "prov",
        ])
    if duplicate_soc:
        rows.append(["soc", "duplicate", "0", "0", "duplicate", "prov"])
    if missing_provenance:
        rows[0][-1] = ""

    header = ",".join(REQUIRED_FIELDS) + "\n"
    return header + "\n".join(",".join(row) for row in rows) + "\n"


def publish_summary_text(include_nlopt: bool = True) -> str:
    header = (
        "solver,library,problem,class,dimension,seed,mode,solver_iters,"
        "f_evals,g_evals,c_evals,J_evals,wall_time_us,final_objective,"
        "known_optimum,accuracy,constraint_violation,status,row_disposition,"
        "cap_status,exclusion_reason,solve_wall_time_us,end_to_end_wall_time_us,"
        "provenance_id\n"
    )
    if not include_nlopt:
        return header
    return header + (
        "nlopt_cobyla,nlopt,hs021,inequality_bound_constrained,2,42,"
        "publication,1,1,0,1,0,1,-99.96,-99.96,0,0,converged,"
        "included,none,,1,1,prov\n"
    )


def expect_failure(path: Path,
                   required_items: list[str],
                   publish_summary: Path | None,
                   needle: str) -> None:
    try:
        validate_register(path, required_items, publish_summary)
    except ValueError as exc:
        if needle in str(exc):
            return
        raise ValueError(f"expected {needle!r}, got {exc}") from exc
    raise ValueError(f"expected validation failure containing {needle!r}")


def run_self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="argmin-disabled-register-") as tmp_s:
        tmp = Path(tmp_s)
        valid = tmp / "valid.csv"
        missing_filter = tmp / "missing-filter.csv"
        missing_nlopt = tmp / "missing-nlopt.csv"
        duplicate = tmp / "duplicate.csv"
        missing_prov = tmp / "missing-provenance.csv"
        summary = tmp / "publish_summary.csv"
        empty_summary = tmp / "empty_publish_summary.csv"

        write_text(valid, valid_register_text())
        write_text(missing_filter, valid_register_text(include_filter=False))
        write_text(missing_nlopt, valid_register_text(include_nlopt=False))
        write_text(duplicate, valid_register_text(duplicate_soc=True))
        write_text(missing_prov, valid_register_text(missing_provenance=True))
        write_text(summary, publish_summary_text(include_nlopt=True))
        write_text(empty_summary, publish_summary_text(include_nlopt=False))

        validate_register(valid, DEFAULT_REQUIRED_ITEMS, summary)
        validate_register(missing_nlopt, DEFAULT_REQUIRED_ITEMS, summary)
        expect_failure(missing_filter, DEFAULT_REQUIRED_ITEMS, summary,
                       "filter_default_grids")
        expect_failure(missing_nlopt, DEFAULT_REQUIRED_ITEMS, empty_summary,
                       "nlopt_cobyla")
        expect_failure(duplicate, DEFAULT_REQUIRED_ITEMS, summary, "duplicate")
        expect_failure(missing_prov, DEFAULT_REQUIRED_ITEMS, summary,
                       "missing provenance_id")
    print("verify_disabled_register: self-test passed", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("register", nargs="?", type=Path)
    parser.add_argument("--publish-summary", type=Path)
    parser.add_argument("--required-item", action="append", dest="required_items")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()
    if args.register is None:
        print("verify_disabled_register: register path required", file=sys.stderr)
        return 2

    required_items = args.required_items or DEFAULT_REQUIRED_ITEMS
    try:
        validate_register(args.register, required_items, args.publish_summary)
    except ValueError as exc:
        print(f"verify_disabled_register: {exc}", file=sys.stderr)
        return 1
    print(f"verify_disabled_register: {args.register} covers "
          f"{len(required_items)} required item(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
