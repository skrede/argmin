from fractions import Fraction

import pytest

from qp_head_to_head import preregistration as prereg
from qp_head_to_head.aggregate import GroupSummary, display_ratio, exact_median
from qp_head_to_head.dense_rules import dense_default
from qp_head_to_head.records import expected_combinations
from qp_head_to_head.render import preregistration_tables
from qp_head_to_head.sparse_rules import block_structure, osqp_replacement

MEDIANS = {"argmin_sparse_admm": 3000, "osqp": 2500, "hpipm": 1000, "tinympc": 2000,
           "argmin_dense_admm": 1500, "argmin_dense_active_set": 1400}


def group(median, n_pass=300, n_ticks=300):
    return GroupSummary(n_ticks, n_pass, Fraction(n_pass, n_ticks), exact_median([median] * n_pass))


def summaries(overrides=(), refused_families=()):
    table = {key: group(MEDIANS.get(key[5], 1000)) for key in expected_combinations(refused_families)}
    for match, value in overrides:
        for key in table:
            if all(part is None or part == actual for part, actual in zip(match, key)):
                table[key] = value
    return table


def sparse(cell=None, variant=None, arm=None):
    return ("sparse", cell, variant, "closed_loop", "mpc", arm)


def dense(regime, tier, cell=None, variant=None, arm=None):
    return ("dense", cell, variant, regime, tier, arm)


def long_horizon_ratios(at_50, at_100):
    return [(sparse(prereg.cell_id(nx, nu, N), arm="argmin_sparse_admm"), group(at_50 if N == 50 else at_100))
            for nx, nu, N in prereg.LONG_HORIZON_CELLS]


def reasons(rule, key="failing_conditions"):
    return [(entry.get("cell") or entry.get("family"), entry.get("variant"), entry["reasons"]) for entry in rule[key]]


def test_block_structure_is_go_at_exactly_three_with_growth():
    rule = block_structure(summaries(long_horizon_ratios(3000, 4000)), ())
    assert rule["outcome"] == "GO"
    assert {cell["ratio"] for cell in rule["cells"]} == {Fraction(3), Fraction(4)}
    assert [family["grows"] for family in rule["families"]] == [True] * 4


def test_block_structure_ratio_just_below_three_is_no_go():
    overrides = [*long_horizon_ratios(3000, 4000), (sparse("nx4_nu2_N50", "nominal", "argmin_sparse_admm"), group(2999))]
    rule = block_structure(summaries(overrides), ())
    assert rule["outcome"] == "NO-GO"
    assert reasons(rule) == [("nx4_nu2_N50", "nominal", ["ratio_below_threshold"])]


def test_block_structure_ratio_just_above_three_is_go():
    assert block_structure(summaries(long_horizon_ratios(3001, 4000)), ())["outcome"] == "GO"


def test_block_structure_equal_ratios_do_not_count_as_growth():
    overrides = [*long_horizon_ratios(3000, 4000), (sparse("nx8_nu3_N100", "tight", "argmin_sparse_admm"), group(3000))]
    rule = block_structure(summaries(overrides), ())
    assert rule["outcome"] == "NO-GO"
    assert reasons(rule) == [("nx8_nu3", "tight", ["no_growth"])]


def test_block_structure_needs_a_reference_passing_every_tick():
    overrides = [*long_horizon_ratios(3000, 4000), (sparse("nx8_nu3_N50", "nominal", "hpipm"), group(1000, 299)),
                 (sparse("nx8_nu3_N50", "nominal", "tinympc"), group(1000, 0))]
    rule = block_structure(summaries(overrides), ())
    assert rule["outcome"] == "NO-GO"
    assert reasons(rule) == [("nx8_nu3_N50", "nominal", ["no_qualifying_reference"]),
                             ("nx8_nu3", "nominal", ["no_growth"])]


@pytest.mark.parametrize("tinympc, best", [(900, "tinympc"), (1000, "hpipm"), (1100, "hpipm")])
def test_block_structure_picks_the_faster_reference_and_breaks_ties_by_registry_order(tinympc, best):
    rule = block_structure(summaries([(sparse(arm="tinympc"), group(tinympc))]), ())
    assert {cell["best_reference"] for cell in rule["cells"]} == {best}


def test_block_structure_fails_a_cell_where_argmin_never_passes():
    overrides = [*long_horizon_ratios(3000, 4000), (sparse("nx4_nu2_N100", "tight", "argmin_sparse_admm"), group(1, 0))]
    rule = block_structure(summaries(overrides), ())
    assert rule["outcome"] == "NO-GO"
    assert reasons(rule)[0] == ("nx4_nu2_N100", "tight", ["argmin_no_passing_tick"])


def test_block_structure_drops_a_refused_tight_family_but_keeps_its_nominal_variant():
    refused = ("nx4_nu2",)
    overrides = [*long_horizon_ratios(3000, 4000), (sparse("nx4_nu2_N50", "tight", "argmin_sparse_admm"), group(1, 0))]
    rule = block_structure(summaries(overrides, refused), refused)
    assert rule["outcome"] == "GO"
    assert [(f["family"], f["variant"]) for f in rule["families"]] == [
        ("nx4_nu2", "nominal"), ("nx8_nu3", "nominal"), ("nx8_nu3", "tight")]
    low = [*overrides, (sparse("nx4_nu2_N50", "nominal", "argmin_sparse_admm"), group(2999))]
    assert reasons(block_structure(summaries(low, refused), refused)) == [
        ("nx4_nu2_N50", "nominal", ["ratio_below_threshold"])]


def test_block_structure_without_a_complete_family_is_not_testable(monkeypatch):
    table = summaries(long_horizon_ratios(3000, 4000))
    monkeypatch.setattr(prereg, "LONG_HORIZON_CELLS", ((4, 2, 50), (8, 3, 50)))
    rule = block_structure(table, ())
    assert rule["outcome"] == "NO-GO"
    assert reasons(rule) == [(None, None, ["growth_not_testable"])]
    assert "growth not testable" in rule["failing_conditions"][0]["message"]


def outcome(rule, regime="fixed", tier="mpc"):
    return next(entry for entry in rule if (entry["regime"], entry["tier"]) == (regime, tier))


def test_dense_default_emits_six_outcomes_in_registry_order():
    rule = dense_default(summaries(), ())
    assert [(entry["regime"], entry["tier"]) for entry in rule] == [
        ("fixed", "mpc"), ("fixed", "high"), ("closed_loop", "mpc"), ("closed_loop", "high"),
        ("churning", "mpc"), ("churning", "high")]


def test_dense_default_single_qualifier_is_the_default():
    overrides = [(dense("fixed", "mpc", "nx4_nu2_N20", "tight", "argmin_dense_admm"), group(10, 299))]
    entry = outcome(dense_default(summaries(overrides), ()))
    assert (entry["outcome"], entry["default_arm"], entry["qualifying"]) == (
        "default", "argmin_dense_active_set", ["argmin_dense_active_set"])
    assert outcome(dense_default(summaries(overrides), ()), "fixed", "high")["qualifying"] == list(
        prereg.DENSE_CANDIDATE_ARMS)


def test_dense_default_arm_winning_every_cell_is_the_default():
    entry = outcome(dense_default(summaries([(dense("fixed", "mpc", arm="argmin_dense_admm"), group(1000))]), ()))
    assert (entry["outcome"], entry["default_arm"], entry["ties"]) == ("default", "argmin_dense_admm", [])
    assert len(entry["wins"]["argmin_dense_admm"]) == 21


def test_dense_default_split_wins_are_a_portfolio_with_the_boundary():
    overrides = [(dense("closed_loop", "high", "nx2_nu1_N10", arm="argmin_dense_admm"), group(1000))]
    entry = outcome(dense_default(summaries(overrides), ()), "closed_loop", "high")
    assert (entry["outcome"], entry["default_arm"], entry["indistinguishable"]) == ("portfolio", None, False)
    assert entry["wins"]["argmin_dense_admm"] == [{"cell": "nx2_nu1_N10", "variant": "nominal"},
                                                  {"cell": "nx2_nu1_N10", "variant": "tight"}]
    assert len(entry["wins"]["argmin_dense_active_set"]) == 18


def test_dense_default_tie_awards_the_cell_to_neither_arm():
    overrides = [(dense("churning", "mpc", "nx12_nu4_N30", "tight", "argmin_dense_admm"), group(1400))]
    entry = outcome(dense_default(summaries(overrides), ()), "churning", "mpc")
    assert (entry["outcome"], entry["default_arm"]) == ("default", "argmin_dense_active_set")
    assert entry["ties"] == [{"cell": "nx12_nu4_N30", "variant": "tight"}]
    assert len(entry["wins"]["argmin_dense_active_set"]) == 19


def test_dense_default_all_ties_are_an_indistinguishable_portfolio():
    entry = outcome(dense_default(summaries([(dense("fixed", "mpc", arm="argmin_dense_admm"), group(1400))]), ()))
    assert (entry["outcome"], entry["indistinguishable"], len(entry["ties"])) == ("portfolio", True, 21)


def test_dense_default_without_a_qualifier_is_no_default():
    overrides = [(dense("fixed", "mpc", "dense_mpc_shaped", arm="argmin_dense_admm"), group(10, 0)),
                 (dense("fixed", "mpc", "nx8_nu3_N30", "nominal", "argmin_dense_active_set"), group(10, 1))]
    entry = outcome(dense_default(summaries(overrides), ()))
    assert (entry["outcome"], entry["default_arm"], entry["qualifying"]) == ("no-default", None, [])
    failures = {c["arm"]: c["pass_rate_failures"] for c in entry["candidates"]}
    assert failures["argmin_dense_admm"] == [{"cell": "dense_mpc_shaped", "variant": "nominal", "pass_rate": 0}]
    assert failures["argmin_dense_active_set"][0]["pass_rate"] == Fraction(1, 300)


def test_dense_default_counts_the_dense_only_problem_in_the_fixed_regime_only():
    rule = dense_default(summaries(), ("nx2_nu1",))
    for entry in rule:
        cells = [(c["cell"], c["variant"]) for c in entry["cells"]]
        assert (("dense_mpc_shaped", "nominal") in cells) is (entry["regime"] == "fixed")
        assert ("nx2_nu1_N10", "tight") not in cells
        assert len(cells) == (20 if entry["regime"] == "fixed" else 19)


def replacement(argmin=None, osqp=None, cell="nx4_nu2_N20", variant="nominal"):
    overrides = [(sparse(arm="argmin_sparse_admm"), group(1200)), (sparse(arm="osqp"), group(1000))]
    if argmin is not None:
        overrides.append((sparse(cell, variant, "argmin_sparse_admm"), argmin))
    if osqp is not None:
        overrides.append((sparse(cell, variant, "osqp"), osqp))
    return osqp_replacement(summaries(overrides), ())


@pytest.mark.parametrize("median, satisfied", [(1250, True), (1249, True), (1251, False)])
def test_osqp_replacement_limit_is_inclusive_and_exact(median, satisfied):
    rule = replacement(group(median))
    assert rule["outcome"] == ("replace" if satisfied else "keep")
    if not satisfied:
        assert reasons(rule, "failing_cells") == [("nx4_nu2_N20", "nominal", ["ratio"])]
        assert "1251/1000" in rule["failing_cells"][0]["message"]


def test_osqp_replacement_needs_argmin_to_pass_every_tick():
    rule = replacement(group(1200, 299))
    assert rule["outcome"] == "keep"
    assert reasons(rule, "failing_cells") == [("nx4_nu2_N20", "nominal", ["pass_rate"])]
    assert "pass rate 299/300" in rule["failing_cells"][0]["message"]


def test_osqp_replacement_names_both_failing_conditions():
    rule = replacement(group(1300, 150))
    assert reasons(rule, "failing_cells") == [("nx4_nu2_N20", "nominal", ["pass_rate", "ratio"])]


def test_osqp_without_a_passing_tick_satisfies_the_cell_when_argmin_passes_all():
    rule = replacement(osqp=group(1, 0))
    assert rule["outcome"] == "replace"
    cell = next(c for c in rule["cells"] if (c["cell"], c["variant"]) == ("nx4_nu2_N20", "nominal"))
    assert (cell["osqp_no_passing_tick"], cell["ratio"], cell["satisfied"]) == (True, None, True)


def test_argmin_without_a_passing_tick_fails_the_replacement():
    rule = replacement(group(1, 0))
    assert reasons(rule, "failing_cells") == [("nx4_nu2_N20", "nominal", ["pass_rate", "ratio"])]


def test_exact_median():
    assert exact_median([3, 1, 2]) == 2
    assert exact_median([2, 1]) == Fraction(3, 2)
    assert exact_median([7]) == 7
    assert exact_median([]) is None


@pytest.mark.parametrize("value, text", [
    (Fraction(1235, 1000), "1.24"), (Fraction(1245, 1000), "1.24"), (Fraction(1255, 1000), "1.26"),
    (Fraction(2999, 1000), "3.00"), (Fraction(3), "3.00"), (Fraction(1, 3), "0.333"), (Fraction(12345), "12300"),
])
def test_display_ratio_rounds_half_to_even_at_three_significant_figures(value, text):
    assert display_ratio(value) == text


def test_preregistration_tables_are_deterministic_and_complete():
    text = preregistration_tables()
    assert text == preregistration_tables()
    for pin in (prereg.ARGMIN_SHA, *prereg.COMPETITOR_PINS.values()):
        assert pin in text
    assert "| OSQP replacement ratio limit (at most) | 5/4 |" in text
    assert "| Block-structure ratio threshold (at least) | 3 |" in text
    assert "x86-64-v3" in text
