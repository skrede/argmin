"""Markdown tables of every pre-registered value, the text frozen into the documentation."""

from __future__ import annotations

from . import preregistration as prereg


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none"
    return repr(value) if isinstance(value, float) else str(value)


def _fields(fields: dict) -> str:
    return ", ".join(f"`{name} {_value(value)}`" for name, value in fields.items())


def _constants() -> list[str]:
    riccati = prereg.TERMINAL_RICCATI
    rows = [
        ["Block-structure ratio threshold (at least)", str(prereg.BLOCK_STRUCTURE_RATIO_THRESHOLD),
         "BLOCK_STRUCTURE_RATIO_THRESHOLD"],
        ["OSQP replacement ratio limit (at most)", str(prereg.OSQP_REPLACEMENT_RATIO_LIMIT),
         "OSQP_REPLACEMENT_RATIO_LIMIT"],
        ["Long-horizon minimum N", str(prereg.LONG_HORIZON_MIN_N), "LONG_HORIZON_MIN_N"],
        *[[f"Pass threshold, {tier} tier (KKT, inclusive)", _value(value), "TIERS"]
          for tier, value in prereg.TIERS.items()],
        ["Repetitions", str(prereg.REPETITIONS), "REPETITIONS"],
        ["Timed ticks per sequence", str(prereg.STEPS_PER_SEQUENCE), "STEPS_PER_SEQUENCE"],
        ["Fixed-regime untimed warm resolves", str(prereg.FIXED_WARMUP_RESOLVES), "FIXED_WARMUP_RESOLVES"],
        ["Fixed-regime timed resolves", str(prereg.FIXED_TIMED_RESOLVES), "FIXED_TIMED_RESOLVES"],
        ["Dual convention probe tolerance", _value(prereg.CONVENTION_PROBE_TOLERANCE), "CONVENTION_PROBE_TOLERANCE"],
        ["Reference solution KKT maximum", _value(prereg.REFERENCE_SOLUTION_KKT_MAX), "REFERENCE_SOLUTION_KKT_MAX"],
        ["Binding row tolerance", _value(prereg.BINDING_ROW_TOLERANCE), "BINDING_ROW_TOLERANCE"],
        ["Binding gate: median binding fraction (at least)", _value(prereg.BINDING_MEDIAN_FRACTION_MIN),
         "BINDING_MEDIAN_FRACTION_MIN"],
        ["Binding gate: state-row binding step fraction (at least)",
         _value(prereg.BINDING_STATE_ROW_STEP_FRACTION_MIN), "BINDING_STATE_ROW_STEP_FRACTION_MIN"],
        ["OSQP adaptive rho interval (iterations)", str(prereg.OSQP_ADAPTIVE_RHO_INTERVAL),
         "OSQP_ADAPTIVE_RHO_INTERVAL"],
        ["Legacy terminal weight", f"{_value(prereg.LEGACY_TERMINAL_WEIGHT)} I", "LEGACY_TERMINAL_WEIGHT"],
        ["Long-horizon terminal weight", f"P_inf(Q + rho I, R + rho I), {_fields(riccati)}", "TERMINAL_RICCATI"],
        ["Display significant figures (round half to even)", str(prereg.DISPLAY_SIGNIFICANT_FIGURES),
         "DISPLAY_SIGNIFICANT_FIGURES"],
        ["Fail classes of a non-passing tick", ", ".join(prereg.FAIL_CLASSES[1:]), "FAIL_CLASSES"],
        ["Arm allowed an in-tick feasible-start fallback", prereg.FALLBACK_ARM, "FALLBACK_ARM"],
        ["Run id pattern", prereg.RUN_ID_PATTERN.pattern, "RUN_ID_PATTERN"],
        ["Manifest schema", prereg.MANIFEST_SCHEMA, "MANIFEST_SCHEMA"],
        ["Verdicts schema", prereg.VERDICTS_SCHEMA, "VERDICTS_SCHEMA"],
    ]
    return markdown_table(["Quantity", "Value", "Constant"], [[q, v, f"`{c}`"] for q, v, c in rows])


def _problem_set() -> list[str]:
    cells = [[prereg.cell_id(nx, nu, N), str(nx), str(nu), str(N),
              "P_inf" if N >= prereg.LONG_HORIZON_MIN_N else f"{_value(prereg.LEGACY_TERMINAL_WEIGHT)} I",
              "dense, sparse", ", ".join(prereg.REGIMES)] for nx, nu, N in prereg.mpc_cells()]
    n, m = prereg.DENSE_ONLY_DIMENSIONS
    cells.append([prereg.DENSE_ONLY_PROBLEM, "-", "-", "-", f"n {n}, m {m}", "dense", "fixed"])
    variants = [[name, _fields(fields)] for name, fields in
                (("nominal", prereg.NOMINAL_VARIANT), ("tight", prereg.TIGHT_VARIANT))]
    return [*markdown_table(["Cell", "nx", "nu", "N", "Terminal weight", "Cohorts", "Regimes"], cells), "",
            *markdown_table(["Variant", "Parameters"], variants)]


def _arms() -> list[str]:
    rows = [[arm.cohort, f"`{arm.name}`", arm.role, arm.dual_kind, ", ".join(arm.tiers),
             "every cell" if all(prereg.expressible(arm.name, N) for _, _, N in prereg.mpc_cells())
             else "long-horizon cells only"]
            for arm in prereg.ARMS]
    settings = [[f"`{arm}`", tier, _fields(fields)]
                for arm, tiers in prereg.ARM_SETTINGS.items() for tier, fields in tiers.items()]
    return [*markdown_table(["Cohort", "Arm", "Role", "Dual kind", "Tiers", "Expressible on"], rows), "",
            *markdown_table(["Arm", "Tier", "Overridden settings"], settings)]


def _pins_and_build() -> list[str]:
    pins = [["argmin", f"`{prereg.ARGMIN_SHA}`"],
            *[[name, f"`{sha}`"] for name, sha in prereg.COMPETITOR_PINS.items()]]
    build = [[key, f"`{_value(value)}`"] for key, value in prereg.BUILD_POLICY.items()]
    return [*markdown_table(["Component", "Commit"], pins), "",
            *markdown_table(["Build setting", "Value"], build)]


def preregistration_tables() -> str:
    sections = (("Constants", _constants()), ("Problem set", _problem_set()), ("Arms", _arms()),
                ("Pins and build policy", _pins_and_build()),
                ("Record format", ["`" + ", ".join(prereg.CSV_COLUMNS) + "`"]))
    lines = []
    for title, body in sections:
        lines += [f"### {title}", "", *body, ""]
    return "\n".join(lines)
