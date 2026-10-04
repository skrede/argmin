"""Every pre-registered value of the head-to-head QP benchmark.

The record format, the problem set, the arm registry, the per-tier arm settings, the pins and the
decision thresholds are fixed here before any recorded run exists; every other module reads them
from this one and never restates a number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

from .arm_settings import ARM_SETTINGS

ARGMIN_SHA = "ca7ec854a02dc90f0b423cec52bda9a4f306c43a"

COMPETITOR_PINS = {
    "osqp": "236713ce9a56c182ac3230d52108f952afce1523",
    "qpoases": "680f18c8ef0018a120e1604b769f056e8368df97",
    "daqp": "29a8987d8b9a596c1ab6acf84c17d38b068b73bc",
    "hpipm": "16a8db59f595bda642f460aa0f62e308f4b76db3",
    "blasfeo": "9628393214623b29e5dd2a8784f7423edbc93e04",
    "tinympc": "023f36bd5b27267e1a96ed47a774d120b5707f13",
    "proxsuite": "b93d7778ffc3299d84b5cb0851022a29bf24a596",
}

BUILD_POLICY = {
    "compiler": "gcc",
    "build_type": "Release",
    "march": "-march=x86-64-v3",
    "release_flags": "-O3 -DNDEBUG",
    "blasfeo_target": "X64_INTEL_HASWELL",
    "blasfeo_la": "HIGH_PERFORMANCE",
    "blasfeo_mf": "PANELMAJ",
    "hpipm_target": "AVX",
    "daqp_faster_math": False,
    "daqp_native_optimization": False,
}

COHORTS = ("dense", "sparse")
LEGACY_CELLS = ((2, 1, 10), (4, 2, 10), (4, 2, 20), (8, 3, 20), (8, 3, 30), (12, 4, 30))
LONG_HORIZON_CELLS = ((4, 2, 50), (4, 2, 100), (8, 3, 50), (8, 3, 100))
LONG_HORIZON_MIN_N = 50
DENSE_ONLY_PROBLEM = "dense_mpc_shaped"
DENSE_ONLY_DIMENSIONS = (10, 5)

VARIANTS = ("nominal", "tight")
NOMINAL_VARIANT = {"input_limit": 0.5, "state_limit": None, "tracking_target": 0.0, "x0_amplitude": 5.0}
TIGHT_VARIANT = {"input_limit": 0.2, "state_limit": 1.0, "tracking_target": 2.0, "x0_amplitude": 1.0}
LEGACY_TERMINAL_WEIGHT = 10.0
TERMINAL_RICCATI = {"rho": 1.0, "max_iterations": 1000, "gain_tolerance": 1e-5}

REGIMES = ("fixed", "closed_loop", "churning")
TIERS = {"mpc": 1e-3, "high": 1e-6}
REPETITIONS = 5
STEPS_PER_SEQUENCE = 60
FIXED_WARMUP_RESOLVES = 20
FIXED_TIMED_RESOLVES = 60


@dataclass(frozen=True, slots=True)
class ArmSpec:
    name: str
    cohort: str
    role: str
    dual_kind: str
    tiers: tuple[str, ...]


_BOTH_TIERS = ("mpc", "high")
_MPC_ONLY = ("mpc",)

ARMS = (
    ArmSpec("argmin_dense_admm", "dense", "candidate", "native", _BOTH_TIERS),
    ArmSpec("argmin_dense_active_set", "dense", "candidate", "native", _BOTH_TIERS),
    ArmSpec("qpoases", "dense", "competitor", "native", _BOTH_TIERS),
    ArmSpec("daqp", "dense", "competitor", "native", _BOTH_TIERS),
    ArmSpec("proxqp_dense", "dense", "competitor", "native", _BOTH_TIERS),
    ArmSpec("osqp", "dense", "competitor", "native", _BOTH_TIERS),
    ArmSpec("argmin_dense_admm_polished", "dense", "report_only", "native", _MPC_ONLY),
    ArmSpec("argmin_sparse_admm", "sparse", "candidate", "native", _BOTH_TIERS),
    ArmSpec("osqp", "sparse", "competitor", "native", _BOTH_TIERS),
    ArmSpec("proxqp_sparse", "sparse", "competitor", "native", _BOTH_TIERS),
    ArmSpec("hpipm", "sparse", "reference", "native", _BOTH_TIERS),
    ArmSpec("tinympc", "sparse", "reference", "recovered", _BOTH_TIERS),
    ArmSpec("argmin_sparse_admm_polished", "sparse", "report_only", "native", _MPC_ONLY),
    ArmSpec("osqp_polished", "sparse", "report_only", "native", _MPC_ONLY),
)

DENSE_CANDIDATE_ARMS = ("argmin_dense_admm", "argmin_dense_active_set")
SPARSE_CANDIDATE_ARM = "argmin_sparse_admm"
REPLACEMENT_INCUMBENT_ARM = "osqp"
STRUCTURE_REFERENCE_ARMS = ("hpipm", "tinympc")
FALLBACK_ARM = "argmin_dense_active_set"
UPDATE_APIS = ("separate", "fused")

BLOCK_STRUCTURE_RATIO_THRESHOLD = Fraction(3)
OSQP_REPLACEMENT_RATIO_LIMIT = Fraction(5, 4)
CONVENTION_PROBE_TOLERANCE = 1e-6
REFERENCE_SOLUTION_KKT_MAX = 1e-9
BINDING_ROW_TOLERANCE = 1e-6
BINDING_MEDIAN_FRACTION_MIN = 0.10
BINDING_STATE_ROW_STEP_FRACTION_MIN = 0.50
OSQP_ADAPTIVE_RHO_INTERVAL = 25

CSV_COLUMNS = (
    "run_id", "cohort", "cell", "variant", "nx", "nu", "N", "n", "m", "regime", "tier", "arm",
    "polish", "rep", "seed", "step", "setup_ns", "update_ns", "solve_ns", "tick_ns", "condense_ns",
    "instr_setup", "instr_update", "instr_solve", "instr_tick", "iterations", "kkt_stat",
    "kkt_prim", "kkt_comp", "kkt_max", "native_status", "pass", "fail_class", "allocs_tick",
    "dual_kind", "fallback", "binding_frac", "churn_frac",
)
FAIL_CLASSES = ("none", "inaccurate", "max_iter", "infeasible_claim", "error")
DUAL_KINDS = ("native", "recovered")
RUN_ID_PATTERN = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{10}$")
CSV_FILE_PATTERN = re.compile(r"^(dense|sparse)_rep([0-9]+)\.csv(\.gz)?$")
DISPLAY_SIGNIFICANT_FIGURES = 3

MANIFEST_SCHEMA = "qp-head-to-head-manifest/1"
VERDICTS_SCHEMA = "qp-head-to-head-verdicts/1"
MANIFEST_KEYS = (
    "schema", "run_id", "started_utc", "finished_utc", "preregistration_commit", "argmin_sha",
    "ctrlpp_harness_sha", "ctrlpp_harness_dirty", "competitor_pins", "build", "host", "clock_state",
    "constants", "arm_settings", "update_api", "expressibility", "binding_gate", "probes",
    "terminal_weights", "csv_files",
)
CLOCK_STATES = ("fixed", "unfixed")


def cell_id(nx: int, nu: int, N: int) -> str:
    return f"nx{nx}_nu{nu}_N{N}"


def family_id(nx: int, nu: int) -> str:
    return f"nx{nx}_nu{nu}"


def mpc_cells() -> tuple[tuple[int, int, int], ...]:
    return LEGACY_CELLS + LONG_HORIZON_CELLS


def families() -> tuple[tuple[int, int], ...]:
    return tuple(dict.fromkeys((nx, nu) for nx, nu, _ in mpc_cells()))


def cohort_arms(cohort: str) -> tuple[ArmSpec, ...]:
    return tuple(arm for arm in ARMS if arm.cohort == cohort)


def expressible(arm: str, N: int) -> bool:
    return arm != "tinympc" or N >= LONG_HORIZON_MIN_N


def expressibility_matrix() -> list[dict]:
    entries = []
    for arm in ARMS:
        for nx, nu, N in mpc_cells():
            entries.append({"cohort": arm.cohort, "arm": arm.name, "cell": cell_id(nx, nu, N),
                            "expressible": expressible(arm.name, N)})
        if arm.cohort == "dense":
            entries.append({"cohort": "dense", "arm": arm.name, "cell": DENSE_ONLY_PROBLEM, "expressible": True})
    return entries


def manifest_constants() -> dict:
    return {
        "repetitions": REPETITIONS,
        "steps_per_sequence": STEPS_PER_SEQUENCE,
        "fixed_warmup_resolves": FIXED_WARMUP_RESOLVES,
        "fixed_timed_resolves": FIXED_TIMED_RESOLVES,
        "tiers": dict(TIERS),
        "convention_probe_tolerance": CONVENTION_PROBE_TOLERANCE,
        "reference_solution_kkt_max": REFERENCE_SOLUTION_KKT_MAX,
        "binding_row_tolerance": BINDING_ROW_TOLERANCE,
        "binding_median_fraction_min": BINDING_MEDIAN_FRACTION_MIN,
        "binding_state_row_step_fraction_min": BINDING_STATE_ROW_STEP_FRACTION_MIN,
        "nominal_variant": dict(NOMINAL_VARIANT),
        "tight_variant": dict(TIGHT_VARIANT),
        "legacy_terminal_weight": LEGACY_TERMINAL_WEIGHT,
        "terminal_riccati": dict(TERMINAL_RICCATI),
        "osqp_adaptive_rho_interval": OSQP_ADAPTIVE_RHO_INTERVAL,
    }


def manifest_template() -> dict:
    template = dict.fromkeys(MANIFEST_KEYS)
    flags = f"{BUILD_POLICY['release_flags']} {BUILD_POLICY['march']}"
    template.update({
        "schema": MANIFEST_SCHEMA,
        "argmin_sha": ARGMIN_SHA,
        "competitor_pins": dict(COMPETITOR_PINS),
        "build": {**BUILD_POLICY, "compiler_version": None, "c_flags": flags, "cxx_flags": flags},
        "constants": manifest_constants(),
        "arm_settings": {arm: {tier: dict(fields) for tier, fields in tiers.items()}
                         for arm, tiers in ARM_SETTINGS.items()},
        "expressibility": expressibility_matrix(),
    })
    return template
