#!/usr/bin/env python3
"""Dolan-More performance-profile post-processor for publish_bench artifacts.

Consumes a single ``publish_bench`` run directory and produces per
``(problem_class, tau, metric)`` performance profiles in PNG + SVG form
plus an aggregated ``publish_report.md`` that lists the per-solver
solved-fraction tables.

Input directory layout (produced by the publish_bench binary):

    {run_dir}/publish_summary.csv
    {run_dir}/traces/{solver}_{problem}_seed{N}.csv

Output directory layout (default ``{run_dir}/profiles``):

    {output_dir}/publish_report.md
    {output_dir}/{class}_{metric}_tau{em04..em12}.png
    {output_dir}/{class}_{metric}_tau{em04..em12}.svg

Method:

For every triple (solver s, problem p, seed) the script reads the matching
trace CSV and computes the first-hit unit cost ``t_tau(s, p)`` -- the
smallest value of the cost metric (wall-clock microseconds or problem-level
function evaluations) at which

    |f_best - f_star| <= tau * max(1, |f_star|)

is achieved (Dolan and More, "Benchmarking Optimization Software with
Performance Profiles", Mathematical Programming 91(2), 2002, Section 2).
``t_tau`` is set to positive infinity if the tolerance is never reached on
the trace. Constrained and bound-constrained rows additionally require finite
trace feasibility no larger than the publication feasibility threshold.

Across the seed sweep the per-(solver, problem, tau, metric) values are
reduced to the median (matching the configured median-of-N protocol used by
the run wrapper), and Dolan-More performance profiles are then computed in
log2 space:

    r(s, p) = t_tau(s, p) / min_s t_tau(s, p)
    rho_s(theta) = #{p : log2(r(s, p)) <= theta} / n_p

with unsolved cells (NaN / inf) excluded from the numerator. The empirical
CDF curve rho_s is plotted vs theta, one curve per solver, one figure per
``(problem_class, metric, tau)`` cell. A markdown table of the final
solved-fraction (rho_s at the maximum theta) accompanies each figure in
``publish_report.md``.

The set of problem classes is discovered at runtime from the ``class``
column of ``publish_summary.csv`` -- the underlying classification enum is
a bitmask over seven atomic flags producing combined strings such as
``inequality_bound_constrained`` or ``global_inequality`` whose presence in
any given run depends on which problems the registry iterated. Hardcoding
a static class tuple would silently drop combined-class profiles.

For the published per-step Dolan-More profile the cohort is filtered to
the HS suite (the ``application``-flagged shapes -- n in [30, 120] -- are
dropped) and to the named per-step solver subset
``{kraft_slsqp_accurate, tr_sqp_accurate, tr_sqp_fast, nlopt_slsqp}``
(reduced from five entries after the line-search SQP family was
empirically collapsed to single-mode; tr_sqp retains both modes);
the cohort is intentionally limited to paired per-problem measurements.

Reference: E. D. Dolan and J. J. More, "Benchmarking Optimization Software
with Performance Profiles", Mathematical Programming 91(2):201-213, 2002.
"""

from __future__ import annotations

import sys
import math
import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TAU_GRID = (1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
METRICS = ("wall_us", "f_evals")
EPS_FEAS_DEFAULT = 1e-8

# PROBLEM_CLASSES is discovered at runtime from publish_summary.csv -- do not
# hardcode. The underlying problem_class enum is a bitmask combining seven
# atomic flags (unconstrained, bound_constrained, inequality, equality,
# mixed, global, application) into strings like
# "inequality_bound_constrained" via to_string(); the exact set present in
# any run depends on which problem_registry problems were iterated. The
# `application`-flagged shapes (n in [30, 120]) are excluded from the
# Dolan-More cohort -- see HS_SUITE_EXCLUDE_TOKEN.


# Dolan-More assumes a homogeneous problem cohort; mixing small-n HS
# textbook problems (n <= 6) with application-shaped problems
# (n in [30, 120]) inflates the cohort heterogeneity that the method
# was not designed for. Application cells appear in publish_summary.csv
# (their regression gates are enforced separately by the regression
# checker) but do not appear in the Dolan-More figures.
HS_SUITE_EXCLUDE_TOKEN = "application"


# Per-step Dolan-More cohort governance: argmin has paired
# per-HS-problem-microseconds-per-step numbers only against NLopt at argmin's
# problem sizes. Ipopt, KNITRO, SNOPT, and scipy SLSQP publish aggregate
# figures that are not in the same comparison frame as argmin's per-HS
# measurements. The PROHIBITED_IN_PER_STEP set is enforced as a hard assertion
# at script entry; the assertion is non-bypassable.
PER_STEP_VALID_SOLVERS = frozenset({
    # argmin internal solvers (paired publish_bench measurements). The
    # line-search SQP family is single-mode after the empirical _fast
    # collapse; the trust-region SQP family (tr_sqp + filter_trsqp) retains
    # both modes since per-mode dispatch is a published API surface and
    # per-mode comparison is meaningful. The filter_trsqp entries inherit
    # the Wachter & Biegler 2005/2006 filter-LS lineage paired with the
    # Byrd-Omojokun 1987 composite step per Lalee, Nocedal & Plantenga 1998.
    "kraft_slsqp_accurate",
    "nw_sqp_accurate",
    "filter_slsqp_accurate",
    "filter_nw_sqp_accurate",
    "tr_sqp_fast", "tr_sqp_accurate",
    "filter_trsqp_fast", "filter_trsqp_accurate",
    # NLopt (the only external library with paired per-step
    # per-HS-problem numbers at argmin's problem sizes).
    "nlopt_slsqp", "nlopt_mma", "nlopt_lbfgs", "nlopt_bobyqa",
    "nlopt_crs2", "nlopt_isres",
})

PROHIBITED_IN_PER_STEP = frozenset({
    "ipopt", "knitro", "snopt", "scipy_slsqp",
})

# The named profile cohort for the argmin per-step Dolan-More figures
# (kraft_slsqp as the sole line-search representative after the _fast
# collapse; tr_sqp + filter_trsqp at parity on both modes, paired against
# NLopt SLSQP as the external reference). The filter_trsqp entries follow
# the Fletcher & Leyffer 2002 filter-acceptance + Byrd-Omojokun 1987
# composite step lineage per Lalee, Nocedal & Plantenga 1998, with the
# Wachter & Biegler 2005/2006 filter-LS refinement.
PROFILE_SOLVERS = frozenset({
    "kraft_slsqp_accurate",
    "tr_sqp_accurate",
    "tr_sqp_fast",
    "filter_trsqp_accurate",
    "filter_trsqp_fast",
    "nlopt_slsqp",
})


def _assert_no_prohibited_solvers(summary: pd.DataFrame) -> None:
    """Refuse to build a per-step Dolan-More profile when the plot cohort
    contains a CUTEst-aggregate-only solver.

    The guard runs against the post-cohort-filter dataframe -- i.e., whatever
    rows are about to be plotted. In default mode the named-subset cohort
    excludes every prohibited solver by construction and the guard is a no-op.
    In ``--no-cohort-filter`` diagnostic mode the cohort widens to every solver
    present in the HS-suite-filtered input; if a prohibited solver is in that
    widened cohort the guard exits 2 so a stray ipopt / knitro / snopt /
    scipy_slsqp row cannot reach a published profile.
    """

    if "solver" not in summary.columns:
        return
    prohibited_present = sorted(
        set(summary["solver"].astype(str).unique())
        .intersection(PROHIBITED_IN_PER_STEP))
    if not prohibited_present:
        return
    sys.stderr.write(
        "dm_profile: refusing to build per-step Dolan-More profile because "
        "publish_summary.csv contains prohibited solvers: "
        f"{prohibited_present}.\n"
        "Per the argmin honest-position on per-step performance, only NLopt "
        "has paired per-HS-problem-microseconds-per-step numbers in the same "
        "comparison frame as argmin's publish_bench cells. Ipopt, KNITRO, "
        "SNOPT, and scipy SLSQP publish CUTEst-aggregate figures that are "
        "not reportable alongside per-HS comparisons.\n"
        "Use a paired per-problem comparator cohort for this profile.\n")
    sys.exit(2)


def _apply_cohort_filters(summary: pd.DataFrame,
                          apply_solver_subset: bool) -> pd.DataFrame:
    """Apply the HS-suite + solver-subset filters to the loaded summary.

    The HS-suite filter excludes ``problem_class::application`` rows
    unconditionally. The solver-subset filter restricts to
    ``PROFILE_SOLVERS``; it is skipped when ``apply_solver_subset`` is
    ``False`` (diagnostic mode via ``--no-cohort-filter``).
    """

    before_rows = len(summary)
    if "class" in summary.columns:
        mask = ~summary["class"].astype(str).str.contains(
            HS_SUITE_EXCLUDE_TOKEN, na=False)
        filtered = summary[mask].copy()
    else:
        filtered = summary.copy()
    sys.stderr.write(
        f"dm_profile: HS-suite filter dropped "
        f"{before_rows - len(filtered)} application-shaped rows; "
        f"profile cohort = {len(filtered)} rows.\n")

    if apply_solver_subset:
        filtered = filtered[
            filtered["solver"].astype(str).isin(PROFILE_SOLVERS)].copy()
        if filtered.empty:
            sys.stderr.write(
                "dm_profile: no rows match the per-step profile cohort "
                f"{sorted(PROFILE_SOLVERS)}; nothing to plot.\n")
            sys.exit(1)
        present = sorted(set(filtered["solver"].astype(str).unique()))
        sys.stderr.write(
            f"dm_profile: profile cohort solvers = {present}\n")
    else:
        sys.stderr.write(
            "dm_profile: --no-cohort-filter active; solver-subset filter "
            "skipped. The HS-suite filter and the prohibited-solver "
            "assertion remain in force.\n")

    return filtered


def _safe_tau_token(tau: float) -> str:
    """Produce a filesystem-safe token for a floating-point tolerance.

    ``f"{tau:.0e}"`` yields strings like ``"1e-08"`` -- the minus sign and
    plus sign are remapped so the result is safe across Windows / shell
    completions / pandas string parsing alike.
    """

    return f"{tau:.0e}".replace("-", "m").replace("+", "")


def _is_constrained_class(problem_class: str) -> bool:
    cls = str(problem_class)
    return cls not in ("unconstrained", "global")


def _clamp_work_units(value: float) -> float:
    if not math.isfinite(value):
        return float("inf")
    return max(1.0, value)


def compute_t_tau(trace_df: pd.DataFrame,
                  f_star: float,
                  tau: float,
                  metric: str,
                  *,
                  constrained: bool = False,
                  eps_feas: float = EPS_FEAS_DEFAULT) -> float:
    """First-hit metric value for objective accuracy and optional feasibility.

    Returns ``float("inf")`` when the tolerance is never met on this trace.
    """

    required_columns = {"f_best", metric}
    if constrained:
        required_columns.add("cv")
    missing = sorted(required_columns.difference(trace_df.columns))
    if missing:
        raise ValueError(f"trace missing required columns: {missing}")

    threshold = tau * max(1.0, abs(f_star))
    diffs = (trace_df["f_best"] - f_star).abs()
    hit_mask = diffs <= threshold
    if constrained:
        cv = pd.to_numeric(trace_df["cv"], errors="coerce")
        hit_mask = hit_mask & np.isfinite(cv) & (cv <= eps_feas)
    hits = trace_df[hit_mask]
    if hits.empty:
        return float("inf")
    return _clamp_work_units(float(hits[metric].iloc[0]))


def performance_profile(t_tau_matrix: pd.DataFrame,
                        solvers: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute the Dolan-More performance profile in log2 ratio space.

    ``t_tau_matrix`` is indexed by problem (rows) and solver (columns).
    Returns a (theta_grid, rho_per_solver) pair where ``rho_per_solver[s]``
    is the empirical CDF over ``log2(r(s, p))`` for solver ``s``.
    """

    row_min = t_tau_matrix.min(axis=1, skipna=True).map(_clamp_work_units)
    r = t_tau_matrix.div(row_min, axis=0)
    log2_r = np.log2(r.replace([np.inf, -np.inf], np.nan))

    finite_max = log2_r.to_numpy(dtype=float)
    finite_max = finite_max[np.isfinite(finite_max)]
    upper = float(finite_max.max()) + 1.0 if finite_max.size > 0 else 1.0

    thetas = np.linspace(0.0, upper, 200)
    n_p = len(t_tau_matrix.index)

    rho: dict[str, np.ndarray] = {}
    for s in solvers:
        column = log2_r[s].to_numpy(dtype=float)
        # Unsolved cells (NaN / inf) are counted as not satisfying any theta;
        # they stay out of the numerator at every theta.
        finite = column[np.isfinite(column)]
        if n_p == 0:
            rho[s] = np.zeros_like(thetas)
            continue
        rho[s] = np.array(
            [float((finite <= th).sum()) / float(n_p) for th in thetas])
    return thetas, rho


def plot_profile(thetas: np.ndarray,
                 rho: dict[str, np.ndarray],
                 title: str,
                 output_png: Path,
                 output_svg: Path) -> None:
    """Render one DM performance-profile figure in PNG + SVG form."""

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for solver in sorted(rho.keys()):
        ax.plot(thetas, rho[solver], label=solver, linewidth=1.4)
    ax.set_xlabel(r"$\log_2(r(s, p))$")
    ax.set_ylabel(r"$\rho_s(\theta)$")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize="small", ncol=2)
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    fig.savefig(output_svg)
    plt.close(fig)


def _disambiguate_solver_problem(stem_no_seed: str,
                                 known_pairs: set[tuple[str, str]]) -> tuple[str, str] | None:
    """Split ``"{solver}_{problem}"`` into its components.

    Both solver and problem names contain underscores, so the split point is
    ambiguous. The function tries every interior split index and returns the
    first one that matches a ``(solver, problem)`` pair seen in the summary
    CSV. Returns ``None`` if no split matches; the caller handles fallback.
    """

    parts = stem_no_seed.split("_")
    for i in range(1, len(parts)):
        candidate = ("_".join(parts[:i]), "_".join(parts[i:]))
        if candidate in known_pairs:
            return candidate
    return None


def _load_summary(run_dir: Path) -> pd.DataFrame:
    summary_path = run_dir / "publish_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing summary CSV: {summary_path}")
    summary = pd.read_csv(summary_path)
    _require_columns(summary, {
        "solver", "problem", "seed", "class", "known_optimum",
        "row_disposition",
    }, summary_path)
    return summary


def _require_columns(df: pd.DataFrame,
                     columns: set[str],
                     path: Path | str) -> None:
    missing = sorted(columns.difference(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")


def _collect_t_tau_rows(run_dir: Path,
                        summary: pd.DataFrame,
                        tau_grid: tuple[float, ...],
                        eps_feas: float) -> pd.DataFrame:
    """Walk every trace file and emit one row per (trace, tau, metric)."""

    traces_dir = run_dir / "traces"
    if not traces_dir.is_dir():
        raise FileNotFoundError(f"missing traces directory: {traces_dir}")

    known_pairs: set[tuple[str, str]] = set(
        zip(summary["solver"].astype(str),
            summary["problem"].astype(str)))
    summary_by_key: dict[tuple[str, str, int], dict[str, object]] = {}
    for record in summary.to_dict("records"):
        disposition = str(record.get("row_disposition", ""))
        if disposition and disposition != "included":
            continue
        summary_by_key[(
            str(record["solver"]),
            str(record["problem"]),
            int(record["seed"]),
        )] = record

    rows: list[dict[str, object]] = []
    for trace_path in sorted(traces_dir.glob("*.csv")):
        stem = trace_path.stem
        try:
            stem_no_seed, seed_token = stem.rsplit("_seed", 1)
            seed = int(seed_token)
        except ValueError:
            print(f"dm_profile: skipping unparseable filename {trace_path.name}",
                  file=sys.stderr)
            continue

        pair = _disambiguate_solver_problem(stem_no_seed, known_pairs)
        if pair is None:
            # Trace file's (solver, problem) is not in the cohort-filtered
            # summary. Skip silently -- this is the expected path for traces
            # whose solver was excluded by the named-cohort filter (e.g.,
            # ipopt traces when the default PROFILE_SOLVERS cohort is active).
            continue

        solver, problem = pair
        summary_row = summary_by_key.get((solver, problem, seed))
        if summary_row is None:
            print(f"dm_profile: no included summary row for {trace_path.name}; skipping",
                  file=sys.stderr)
            continue

        f_star = float(summary_row["known_optimum"])
        constrained = _is_constrained_class(str(summary_row["class"]))
        try:
            tdf = pd.read_csv(trace_path)
        except Exception as exc:
            print(f"dm_profile: read error on {trace_path.name}: {exc}",
                  file=sys.stderr)
            continue
        if tdf.empty:
            continue

        for tau in tau_grid:
            for metric in METRICS:
                if metric not in tdf.columns:
                    continue
                t_tau = compute_t_tau(tdf, f_star, tau, metric,
                                      constrained=constrained,
                                      eps_feas=eps_feas)
                rows.append({
                    "solver": solver,
                    "problem": problem,
                    "seed": seed,
                    "tau": tau,
                    "metric": metric,
                    "t_tau": t_tau,
                })
    return pd.DataFrame(rows)


def _write_self_test_csv(run_dir: Path) -> None:
    traces_dir = run_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "publish_summary.csv").write_text(
        "solver,library,problem,class,dimension,seed,mode,solver_iters,"
        "f_evals,g_evals,c_evals,J_evals,wall_time_us,final_objective,"
        "known_optimum,accuracy,constraint_violation,status,row_disposition,"
        "cap_status,exclusion_reason,solve_wall_time_us,end_to_end_wall_time_us,"
        "provenance_id\n"
        "solver_a,argmin,constrained,inequality,1,42,publication,1,"
        "1,0,0,0,1,0.0,0.0,0.0,1e-9,converged,included,"
        "none,,1,1,selftest\n"
        "solver_a,argmin,unconstrained,unconstrained,1,42,publication,1,"
        "1,0,0,0,1,0.0,0.0,0.0,0.0,converged,included,"
        "none,,1,1,selftest\n")
    (traces_dir / "solver_a_constrained_seed42.csv").write_text(
        "iter,f_evals,g_evals,c_evals,J_evals,wall_us,"
        "f_current,f_best,accuracy,cv,step_norm,kkt_residual\n"
        "0,0,0,0,0,0,0.0,0.0,0.0,1e-5,0.0,0.0\n"
        "1,0,0,0,0,0,0.0,0.0,0.0,nan,0.0,0.0\n")
    (traces_dir / "solver_a_unconstrained_seed42.csv").write_text(
        "iter,f_evals,g_evals,c_evals,J_evals,wall_us,"
        "f_current,f_best,accuracy,cv,step_norm,kkt_residual\n"
        "0,0,0,0,0,0,0.0,0.0,0.0,nan,0.0,0.0\n")


def _run_self_test() -> int:
    failures: list[str] = []
    constrained = pd.DataFrame({
        "f_best": [0.0, 0.0, 0.0],
        "cv": [1e-5, float("nan"), 1e-9],
        "f_evals": [0, 0, 2],
        "wall_us": [0, 0, 2],
    })
    if math.isfinite(compute_t_tau(constrained.iloc[:1], 0.0, 1e-8, "f_evals",
                                   constrained=True, eps_feas=1e-8)):
        failures.append("objective-accurate infeasible constrained trace hit")
    if math.isfinite(compute_t_tau(constrained.iloc[1:2], 0.0, 1e-8, "f_evals",
                                   constrained=True, eps_feas=1e-8)):
        failures.append("non-finite constrained feasibility hit")
    clamped = compute_t_tau(constrained.iloc[2:3], 0.0, 1e-8, "f_evals",
                            constrained=True, eps_feas=1e-8)
    if clamped < 1.0:
        failures.append("finite constrained work unit was not clamped")

    unconstrained = pd.DataFrame({
        "f_best": [0.0],
        "f_evals": [0],
        "wall_us": [0],
    })
    if compute_t_tau(unconstrained, 0.0, 1e-8, "f_evals",
                     constrained=False, eps_feas=1e-8) != 1.0:
        failures.append("unconstrained objective hit failed or was not clamped")

    with tempfile.TemporaryDirectory(prefix="argmin-dm-selftest-") as tmp:
        run_dir = Path(tmp)
        _write_self_test_csv(run_dir)
        summary = _load_summary(run_dir)
        rows = _collect_t_tau_rows(run_dir, summary, (1e-8,), EPS_FEAS_DEFAULT)
        constrained_rows = rows[
            (rows["problem"] == "constrained")
            & (rows["metric"] == "f_evals")]
        if constrained_rows.empty or math.isfinite(float(constrained_rows["t_tau"].iloc[0])):
            failures.append("fixture constrained infeasible trace produced a hit")
        unconstrained_rows = rows[
            (rows["problem"] == "unconstrained")
            & (rows["metric"] == "f_evals")]
        if unconstrained_rows.empty or float(unconstrained_rows["t_tau"].iloc[0]) != 1.0:
            failures.append("fixture unconstrained trace did not hit at clamped unit")

    for failure in failures:
        print(f"dm_profile self-test: FAIL: {failure}", file=sys.stderr)
    if failures:
        return 1
    print("dm_profile self-test: PASS", file=sys.stderr)
    return 0


def _emit_profiles(t_tau_med: pd.DataFrame,
                   problem_classes: tuple[str, ...],
                   tau_grid: tuple[float, ...],
                   output_dir: Path,
                   dry_run: bool) -> tuple[list[str], int, int]:
    """Emit one PNG + SVG + markdown block per (class, metric, tau) cell.

    Returns ``(report_lines, png_count, svg_count)``.
    """

    report_lines: list[str] = []
    png_count = 0
    svg_count = 0

    for metric in METRICS:
        for cls in problem_classes:
            for tau in tau_grid:
                subset = t_tau_med[
                    (t_tau_med["metric"] == metric)
                    & (t_tau_med["class"] == cls)
                    & (t_tau_med["tau"] == tau)
                ]
                if subset.empty:
                    continue
                pivot = subset.pivot_table(
                    index="problem",
                    columns="solver",
                    values="t_tau",
                    aggfunc="median")
                if pivot.empty or pivot.shape[1] == 0:
                    continue

                solvers = list(pivot.columns)
                thetas, rho = performance_profile(pivot, solvers)

                tau_token = _safe_tau_token(tau)
                stem = f"{cls}_{metric}_tau{tau_token}"
                png_path = output_dir / f"{stem}.png"
                svg_path = output_dir / f"{stem}.svg"
                title = f"DM profile -- class={cls}, metric={metric}, tau={tau:.0e}"

                if dry_run:
                    print(f"dm_profile: would write {png_path} and {svg_path}",
                          file=sys.stderr)
                else:
                    plot_profile(thetas, rho, title, png_path, svg_path)
                    png_count += 1
                    svg_count += 1

                final_solved = sorted(
                    ((s, float(rho[s][-1])) for s in solvers),
                    key=lambda item: item[1],
                    reverse=True)
                report_lines.append(f"### {cls} -- metric={metric}, tau={tau:.0e}")
                report_lines.append("")
                report_lines.append("| solver | solved fraction |")
                report_lines.append("|--------|-----------------|")
                for solver, frac in final_solved:
                    report_lines.append(f"| {solver} | {frac:.3f} |")
                report_lines.append("")
    return report_lines, png_count, svg_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Compute Dolan-More performance profiles from a "
                     "publish_bench run and emit PNG + SVG + markdown."))
    parser.add_argument("--run-dir", type=Path,
                        help="Directory containing publish_summary.csv and traces/.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: {run_dir}/profiles).")
    parser.add_argument("--tau-grid", nargs="+", type=float,
                        default=list(TAU_GRID),
                        help="Tolerance grid (default: 1e-4 1e-6 1e-8 1e-10 1e-12).")
    parser.add_argument("--eps-feas", type=float, default=EPS_FEAS_DEFAULT,
                        help="Absolute feasibility threshold for constrained trace hits.")
    parser.add_argument("--solvers", nargs="+", default=None,
                        help="Restrict to this solver list (default: all).")
    parser.add_argument("--classes", nargs="+", default=None,
                        help=("Restrict to this problem-class list "
                              "(intersection with the runtime-discovered set; "
                              "default: all discovered classes)."))
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip file writes; print intended output paths.")
    parser.add_argument("--no-cohort-filter", action="store_true",
                        help=("Diagnostic use only: bypass the named-subset "
                              "profile cohort filter and plot every solver "
                              "present in publish_summary.csv. The HS-suite "
                              "class filter and the prohibited-solver "
                              "assertion remain in force."))
    parser.add_argument("--self-test", action="store_true",
                        help="Run built-in feasibility and zero-work checks.")
    args = parser.parse_args()

    if args.self_test:
        return _run_self_test()
    if args.run_dir is None:
        parser.error("--run-dir is required unless --self-test is used")

    run_dir: Path = args.run_dir
    output_dir: Path = args.output_dir if args.output_dir is not None else run_dir / "profiles"
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_summary(run_dir)

    # HS-suite filter (excluding application-shaped cells) + named profile
    # cohort filter. The HS-suite filter applies unconditionally; the
    # named-subset filter respects --no-cohort-filter.
    summary = _apply_cohort_filters(
        summary, apply_solver_subset=not args.no_cohort_filter)

    # Honest-position guard: prohibited per-step solvers in the post-filter
    # cohort terminate the run before any plotting or report writing. Default
    # mode's named cohort excludes them by construction, so this is a no-op
    # there; --no-cohort-filter mode triggers the guard if a prohibited
    # solver survives into the plot cohort.
    _assert_no_prohibited_solvers(summary)

    if args.solvers:
        summary = summary[summary["solver"].isin(args.solvers)].copy()
        if summary.empty:
            print("dm_profile: --solvers filter eliminated every row",
                  file=sys.stderr)
            return 1

    # Runtime discovery: the live publish_summary.csv's `class` column carries
    # the actual to_string() output of the problem_class bitmask for each
    # iterated problem. Sorting gives deterministic plot / filename ordering
    # across runs.
    discovered = [c for c in summary["class"].unique()
                  if isinstance(c, str) and c]
    problem_classes: tuple[str, ...] = tuple(sorted(discovered))
    if args.classes:
        keep = set(args.classes)
        problem_classes = tuple(c for c in problem_classes if c in keep)
    print(f"dm_profile: discovered {len(problem_classes)} problem classes: "
          f"{list(problem_classes)}",
          file=sys.stderr)

    t_tau_long = _collect_t_tau_rows(run_dir, summary, tuple(args.tau_grid),
                                     args.eps_feas)
    if t_tau_long.empty:
        print("dm_profile: no usable trace rows; nothing to emit",
              file=sys.stderr)
        return 1

    # Median across the seed sweep per (solver, problem, tau, metric).
    t_tau_med = (
        t_tau_long
        .groupby(["solver", "problem", "tau", "metric"], as_index=False)
        ["t_tau"].median())

    class_by_problem = (
        summary.groupby("problem")["class"].first().to_dict())
    t_tau_med["class"] = t_tau_med["problem"].map(class_by_problem)

    report_lines: list[str] = []
    report_lines.append("# Performance-profile Report")
    report_lines.append("")
    report_lines.append(
        f"Discovered {len(problem_classes)} problem classes: "
        f"{list(problem_classes)}")
    report_lines.append("")
    report_lines.append(
        "Each table below lists the solved fraction at the maximum theta "
        "for the corresponding Dolan-More profile cell.")
    report_lines.append("")

    cell_lines, png_count, svg_count = _emit_profiles(
        t_tau_med, problem_classes, tuple(args.tau_grid), output_dir, args.dry_run)
    report_lines.extend(cell_lines)

    if not args.dry_run:
        report_path = output_dir / "publish_report.md"
        report_path.write_text("\n".join(report_lines) + "\n")
        print(f"dm_profile: done -> {output_dir} "
              f"(png={png_count}, svg={svg_count})",
              file=sys.stderr)
    else:
        print(f"dm_profile: dry-run complete (would emit png={png_count}, "
              f"svg={svg_count})",
              file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
