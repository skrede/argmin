"""Validation of a run manifest against the pre-registration.

Returns the families whose tight variant the binding gate refused; raises RefusedInput on the
first manifest that differs from what was pre-registered.
"""

from __future__ import annotations

import math
import re

from . import preregistration as prereg

_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RefusedInput(Exception):
    pass


def _refuse(key: str, message: str) -> None:
    raise RefusedInput(f"manifest.json: {key}: {message}")


def _same(a: object, b: object) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(map(_same, a, b))
    return type(a) is type(b) and a == b


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _require(manifest: dict, key: str, kind: type) -> object:
    value = manifest[key]
    if not isinstance(value, kind) or isinstance(value, bool) is not (kind is bool):
        _refuse(key, f"expected a {kind.__name__}, found {type(value).__name__}")
    return value


def _check_identity(manifest: dict, expect_commit: str | None) -> None:
    if manifest["schema"] != prereg.MANIFEST_SCHEMA:
        _refuse("schema", f"expected {prereg.MANIFEST_SCHEMA!r}, found {manifest['schema']!r}")
    run_id = manifest["run_id"]
    if not isinstance(run_id, str) or not prereg.RUN_ID_PATTERN.match(run_id):
        _refuse("run_id", f"{run_id!r} does not match {prereg.RUN_ID_PATTERN.pattern}")
    for key in ("started_utc", "finished_utc"):
        if not _require(manifest, key, str):
            _refuse(key, "empty")
    for key in ("preregistration_commit", "ctrlpp_harness_sha"):
        if not _SHA1.match(_require(manifest, key, str)):
            _refuse(key, "not a full commit SHA")
    if expect_commit is not None and manifest["preregistration_commit"] != expect_commit:
        _refuse("preregistration_commit", f"{manifest['preregistration_commit']} is not {expect_commit}")
    if _require(manifest, "ctrlpp_harness_dirty", bool):
        _refuse("ctrlpp_harness_dirty", "the harness working tree was dirty")
    _require(manifest, "host", dict)


def _check_pins_and_build(manifest: dict) -> None:
    if manifest["argmin_sha"] != prereg.ARGMIN_SHA:
        _refuse("argmin_sha", f"{manifest['argmin_sha']!r} is not {prereg.ARGMIN_SHA}")
    if not _same(manifest["competitor_pins"], prereg.COMPETITOR_PINS):
        _refuse("competitor_pins", "differ from the pre-registered pins")
    build = _require(manifest, "build", dict)
    for key, value in prereg.BUILD_POLICY.items():
        if key in ("march", "release_flags"):
            continue
        if not _same(build.get(key), value):
            _refuse("build", f"{key} is {build.get(key)!r}, expected {value!r}")
    if not isinstance(build.get("compiler_version"), str):
        _refuse("build", "compiler_version is not recorded")
    required = [prereg.BUILD_POLICY["march"], *prereg.BUILD_POLICY["release_flags"].split()]
    for key in ("c_flags", "cxx_flags"):
        flags = build.get(key)
        tokens = flags.split() if isinstance(flags, str) else []
        missing = [flag for flag in required if flag not in tokens]
        if missing:
            _refuse("build", f"{key} lacks {' '.join(missing)}")


def _check_settings(manifest: dict) -> None:
    if not _same(manifest["constants"], prereg.manifest_constants()):
        _refuse("constants", "differ from the pre-registered constants")
    settings = _require(manifest, "arm_settings", dict)
    if settings.keys() != prereg.ARM_SETTINGS.keys():
        _refuse("arm_settings", "the arm set differs from the registry")
    for arm, tiers in prereg.ARM_SETTINGS.items():
        recorded = settings[arm]
        if not isinstance(recorded, dict) or recorded.keys() != tiers.keys():
            _refuse("arm_settings", f"{arm}: the tier set differs from the registry")
        for tier, fields in tiers.items():
            for field, value in fields.items():
                if not isinstance(recorded[tier], dict) or not _same(recorded[tier].get(field), value):
                    _refuse("arm_settings", f"{arm} {tier} {field} differs from {value!r}")
    names = {arm.name for arm in prereg.ARMS}
    update_api = _require(manifest, "update_api", dict)
    if update_api.keys() != names or any(value not in prereg.UPDATE_APIS for value in update_api.values()):
        _refuse("update_api", f"needs one of {prereg.UPDATE_APIS} for every arm")


def _check_expressibility(manifest: dict) -> None:
    entries = _require(manifest, "expressibility", list)
    fields = ("cohort", "arm", "cell", "expressible")
    if not all(isinstance(entry, dict) and entry.keys() == set(fields) for entry in entries):
        _refuse("expressibility", f"every entry needs exactly {', '.join(fields)}")
    recorded = [tuple(entry[field] for field in fields) for entry in entries]
    expected = {tuple(entry[field] for field in fields) for entry in prereg.expressibility_matrix()}
    if len(recorded) != len(set(recorded)) or set(recorded) != expected:
        _refuse("expressibility", "differs from the registry-derived matrix")


def _check_probes_and_weights(manifest: dict) -> None:
    probes = _require(manifest, "probes", dict)
    probed = dict.fromkeys(arm.name for arm in prereg.ARMS if arm.role != "report_only" and arm.dual_kind == "native")
    for arm in probed:
        probe = probes.get(arm)
        if not isinstance(probe, dict) or probe.get("passed") is not True:
            _refuse("probes", f"{arm}: no passed convention probe")
        error = probe.get("max_dual_error")
        if not _is_number(error) or error > prereg.CONVENTION_PROBE_TOLERANCE:
            _refuse("probes", f"{arm}: max_dual_error {error!r} exceeds {prereg.CONVENTION_PROBE_TOLERANCE}")
    weights = _require(manifest, "terminal_weights", dict)
    cells = {prereg.cell_id(*cell) for cell in prereg.LONG_HORIZON_CELLS}
    if weights.keys() != cells or not all(isinstance(v, str) and _SHA256.match(v) for v in weights.values()):
        _refuse("terminal_weights", "needs a SHA-256 for each long-horizon cell")


def _family_passes(family: str, record: object, cells: list[str]) -> bool:
    if not isinstance(record, dict) or not isinstance(record.get("cells"), dict):
        _refuse("binding_gate", f"{family}: no per-cell figures")
    if record["cells"].keys() != set(cells) or not isinstance(record.get("family_passed"), bool):
        _refuse("binding_gate", f"{family}: needs figures for {', '.join(cells)} and family_passed")
    passes = True
    for cell in cells:
        figures = record["cells"][cell]
        median = figures.get("median_binding_fraction") if isinstance(figures, dict) else None
        state = figures.get("max_state_row_step_fraction") if isinstance(figures, dict) else None
        if not all(_is_number(x) and 0.0 <= x <= 1.0 for x in (median, state)):
            _refuse("binding_gate", f"{family} {cell}: figures must be fractions")
        passes &= median >= prereg.BINDING_MEDIAN_FRACTION_MIN
        passes &= state >= prereg.BINDING_STATE_ROW_STEP_FRACTION_MIN
    if record["family_passed"] is not passes:
        _refuse("binding_gate", f"{family}: family_passed disagrees with its figures")
    return passes


def _refused_families(manifest: dict) -> tuple[str, ...]:
    gate = _require(manifest, "binding_gate", dict)
    families = {prereg.family_id(nx, nu): [] for nx, nu in prereg.families()}
    for nx, nu, N in prereg.mpc_cells():
        families[prereg.family_id(nx, nu)].append(prereg.cell_id(nx, nu, N))
    if gate.keys() != families.keys():
        _refuse("binding_gate", f"needs exactly the families {', '.join(families)}")
    return tuple(f for f, cells in families.items() if not _family_passes(f, gate[f], cells))


def check_manifest(manifest: object, *, advisory: bool, schema_only: bool,
                   expect_preregistration_commit: str | None) -> tuple[str, ...]:
    if not isinstance(manifest, dict):
        raise RefusedInput("manifest.json: not a JSON object")
    missing = [key for key in prereg.MANIFEST_KEYS if key not in manifest]
    if missing:
        _refuse("keys", f"missing {', '.join(missing)}")
    _check_identity(manifest, expect_preregistration_commit)
    _check_pins_and_build(manifest)
    _check_settings(manifest)
    _check_expressibility(manifest)
    _check_probes_and_weights(manifest)
    refused = _refused_families(manifest)
    files = _require(manifest, "csv_files", dict)
    if not files or not all(isinstance(v, str) and _SHA256.match(v) for v in files.values()):
        _refuse("csv_files", "needs a SHA-256 for every listed CSV file")
    if manifest["clock_state"] not in prereg.CLOCK_STATES:
        _refuse("clock_state", f"{manifest['clock_state']!r} is not one of {prereg.CLOCK_STATES}")
    if manifest["clock_state"] != "fixed" and not (advisory or schema_only):
        _refuse("clock_state", f"{manifest['clock_state']!r}: a recorded run needs a fixed clock (or --advisory)")
    return refused
