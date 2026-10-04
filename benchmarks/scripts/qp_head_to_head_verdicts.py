#!/usr/bin/env python3
"""Compute the pre-registered verdicts of the head-to-head QP benchmark from one recorded run.

Reads one run directory (manifest.json plus the per-cohort, per-repetition tick CSVs it lists),
refuses any run that differs from the pre-registration, and writes a JSON verdict record and
Markdown tables at the paths given by --output-json and --output-md.

Exit status: 0 on success, 3 when the input is refused (stderr carries a line starting with
"REFUSED:"), 2 on a usage error.

--schema-only validates a smoke run (format, domains, hashes and manifest; no completeness) and
prints SCHEMA OK. --advisory accepts a run whose clock state is not fixed and stamps the output as
not publishable. --print-preregistration and --print-manifest-template print the pre-registered
tables and the manifest template the harness fills in.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qp_head_to_head import preregistration as prereg
from qp_head_to_head.aggregate import summarize
from qp_head_to_head.dense_rules import dense_default
from qp_head_to_head.records import RefusedInput, RunData, load_run
from qp_head_to_head.render import preregistration_tables, render_json, render_markdown, report_sections
from qp_head_to_head.report import group_reports
from qp_head_to_head.sparse_rules import block_structure, osqp_replacement


def compute_outputs(run: RunData) -> tuple[str, str]:
    summaries = summarize(run)
    verdicts = {
        "schema": prereg.VERDICTS_SCHEMA,
        "run_id": run.run_id,
        "recorded": run.recorded,
        "block_structure": block_structure(summaries, run.refused_families),
        "dense_default": dense_default(summaries, run.refused_families),
        "osqp_replacement": osqp_replacement(summaries, run.refused_families),
        **report_sections(run, summaries, group_reports(run)),
    }
    return render_json(verdicts), render_markdown(verdicts)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--advisory", action="store_true")
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--expect-preregistration-commit")
    parser.add_argument("--print-preregistration", action="store_true")
    parser.add_argument("--print-manifest-template", action="store_true")
    args = parser.parse_args()
    printing = args.print_preregistration or args.print_manifest_template
    if not printing and args.run_dir is None:
        parser.error("--run-dir is required")
    if not printing and not args.schema_only and (args.output_json is None or args.output_md is None):
        parser.error("--output-json and --output-md are required")
    return args


def main() -> int:
    args = _arguments()
    if args.print_preregistration:
        sys.stdout.write(preregistration_tables())
        return 0
    if args.print_manifest_template:
        sys.stdout.write(json.dumps(prereg.manifest_template(), sort_keys=True, indent=2) + "\n")
        return 0
    try:
        run = load_run(args.run_dir, advisory=args.advisory, schema_only=args.schema_only,
                       expect_preregistration_commit=args.expect_preregistration_commit)
    except RefusedInput as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 3
    if args.schema_only:
        print("SCHEMA OK")
        return 0
    json_text, markdown_text = compute_outputs(run)
    args.output_json.write_text(json_text, encoding="utf-8", newline="\n")
    args.output_md.write_text(markdown_text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
