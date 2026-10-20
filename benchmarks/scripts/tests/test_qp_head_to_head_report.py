import dataclasses
import json
import math
from fractions import Fraction

import pytest

from qp_head_to_head import preregistration as prereg
from qp_head_to_head.records import RunData, Tick
from qp_head_to_head.report import display_time_us, group_reports, percentile_linear
from qp_head_to_head_verdicts import compute_outputs

DENSE = ("dense", "nx2_nu1_N10", "nominal", "fixed", "mpc", "argmin_dense_active_set")
SPARSE = ("sparse", "nx2_nu1_N10", "nominal", "fixed", "mpc", "osqp")
BASE = dict(run_id="20261004T120000Z-0123456789", nx=2, nu=1, N=10, n=10, m=10, polish=0, rep=0, seed=-1,
            step=1, setup_ns=-1, update_ns=100, solve_ns=300, tick_ns=400, condense_ns=50, instr_setup=-1,
            instr_update=1000, instr_solve=3000, instr_tick=4000, iterations=7, kkt_stat=1e-9, kkt_prim=1e-9,
            kkt_comp=1e-9, kkt_max=1e-9, native_status="solved", passed=1, fail_class="none", allocs_tick=0,
            dual_kind="native", fallback=0, binding_frac=0.25, churn_frac=-1.0)


def tick(key=DENSE, **fields):
    names = ("cohort", "cell", "variant", "regime", "tier", "arm")
    return Tick(**{**BASE, **dict(zip(names, key)), **fields})


def failing(key=DENSE, **fields):
    return tick(key, **{"passed": 0, "fail_class": "inaccurate", "kkt_max": 1.0, **fields})


def run_of(*ticks, manifest=None, refused=()):
    return RunData(run_id=BASE["run_id"], manifest=manifest or {}, ticks=tuple(ticks), inputs=(),
                   refused_families=refused, recorded=True)


def report(*ticks, key=DENSE):
    return group_reports(run_of(*ticks))[key]


def test_percentile_linear_interpolates_exactly():
    assert percentile_linear([4, 1, 3, 2], Fraction(1, 4)) == Fraction(7, 4)
    assert percentile_linear([4, 1, 3, 2], Fraction(3, 4)) == Fraction(13, 4)
    assert percentile_linear([9], Fraction(1, 4)) == percentile_linear([9], Fraction(3, 4)) == 9
    assert percentile_linear([], Fraction(1, 4)) is None


@pytest.mark.parametrize("ns, shown", [(1235, "1.24"), (1245, "1.24"), (1255, "1.26")])
def test_display_time_rounds_half_to_even_in_microseconds(ns, shown):
    assert display_time_us(Fraction(ns)) == shown


def test_times_and_instructions_come_from_passing_ticks_only():
    result = report(tick(step=0, setup_ns=5000, instr_setup=9000, tick_ns=400), tick(step=1, tick_ns=600),
                    failing(step=0, setup_ns=1, tick_ns=1, update_ns=1, solve_ns=1, condense_ns=1,
                            instr_tick=1))
    assert result.median_setup_ns == 5000 and result.median_instr_setup == 9000
    assert (result.p25_tick_ns, result.p75_tick_ns) == (450, 550)
    assert (result.median_update_ns, result.median_solve_ns, result.median_condense_ns) == (100, 300, 50)
    assert result.median_instr_tick == 4000


def test_counts_and_extremes_cover_every_tick():
    result = report(tick(iterations=3, allocs_tick=2), tick(iterations=4, fallback=1),
                    failing(iterations=40, allocs_tick=5, fallback=1), failing(fail_class="max_iter"))
    assert (result.iterations_min, result.iterations_median, result.iterations_max) == (3, Fraction(11, 2), 40)
    assert result.worst_kkt_max == 1.0 and result.allocs_max == 5 and result.fallback_count == 2
    assert result.fail_classes == {"inaccurate": 1, "max_iter": 1, "infeasible_claim": 0, "error": 0}


def test_unavailable_sentinels_are_not_values():
    result = report(tick(instr_tick=-1, instr_update=-1, allocs_tick=-1), tick(instr_tick=-1, allocs_tick=-1),
                    key=DENSE)
    assert result.median_instr_tick is None and result.median_instr_update == 1000
    assert result.allocs_max is None
    sparse = report(tick(SPARSE, condense_ns=-1), key=SPARSE)
    assert sparse.median_condense_ns is None and sparse.fallback_count == 0


def test_a_group_without_a_passing_tick_has_no_time():
    result = report(failing(), failing(kkt_max=math.nan))
    assert result.p25_tick_ns is None and result.median_update_ns is None
    assert math.isnan(result.worst_kkt_max)


def test_near_threshold_count_is_one_ulp_either_side():
    threshold = prereg.TIERS["mpc"]
    above = math.nextafter(threshold, math.inf)
    below = math.nextafter(threshold, -math.inf)
    near = [tick(kkt_max=threshold), tick(kkt_max=below), failing(kkt_max=above)]
    far = [tick(kkt_max=math.nextafter(below, -math.inf)), failing(kkt_max=math.nextafter(above, math.inf))]
    assert report(*near, *far).near_threshold_count == 3


def _outputs(*ticks, **run):
    json_text, markdown_text = compute_outputs(run_of(*ticks, **run))
    return json.loads(json_text), markdown_text


def test_summary_shows_the_split_and_marks_unavailable_values():
    verdicts, markdown = _outputs(tick(step=0, setup_ns=5000), tick(tick_ns=1245), failing(fallback=1),
                                  tick(SPARSE, condense_ns=-1, instr_tick=-1, allocs_tick=-1))
    entries = {tuple(entry[name] for name in ("cohort", "cell", "variant", "regime", "tier", "arm")): entry
               for entry in verdicts["summary"]}
    dense, sparse = entries[DENSE], entries[SPARSE]
    assert dense["median_tick_ns"] == "1645/2" and dense["median_tick_us"] == "0.822"
    assert dense["fallback_count"] == 1 and sparse["fallback_count"] == 0
    assert sparse["median_condense_ns"] is None and sparse["median_instr_tick_display"] == "n/a"
    assert sparse["allocs_max_display"] == "n/a"
    empty = entries[("sparse", "nx2_nu1_N10", "tight", "fixed", "mpc", "osqp")]
    assert empty["p25_tick_ns"] is None and empty["p25_tick_us"] == "no passing tick"
    assert markdown.index("## Group summary") < markdown.index("## Block-structured storage")
    assert "| nx2_nu1_N10 | nominal | `argmin_dense_active_set` | 2/3 | 0.822 |" in markdown


def test_exclusions_inputs_and_preregistration_sections():
    gate = {"nx2_nu1": {"cells": {"nx2_nu1_N10": {"median_binding_fraction": 0.05,
                                                  "max_state_row_step_fraction": 0.2}}, "family_passed": False}}
    verdicts, _ = _outputs(manifest={"binding_gate": gate}, refused=("nx2_nu1",))
    assert verdicts["excluded"]["refused_tight_families"] == [{"family": "nx2_nu1", **gate["nx2_nu1"]}]
    not_expressible = verdicts["excluded"]["not_expressible"]
    assert {entry["arm"] for entry in not_expressible} == {"tinympc"} and len(not_expressible) == 6
    assert verdicts["preregistration"]["thresholds"]["osqp_replacement_ratio_at_most"] == "5/4"
    assert verdicts["preregistration"]["argmin_sha"] == prereg.ARGMIN_SHA
    assert verdicts["inputs"] == []


def test_report_reads_ticks_in_any_order():
    ticks = [tick(tick_ns=ns, iterations=ns // 100, kkt_max=ns * 1e-12) for ns in (400, 900, 500, 700)]
    assert group_reports(run_of(*ticks)) == group_reports(run_of(*reversed(ticks)))
    assert dataclasses.asdict(report(*ticks))["iterations_median"] == 6
