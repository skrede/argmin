// Direct truth-table and lifecycle tests for the filter acceptance
// substrate. These exercise filter_set in isolation (no full solve):
// the Pareto-dominance rule with its asymmetric envelope margins, the
// maximum-violation ceiling, the add/prune lifecycle, and the reset
// (clear) semantics.
//
// Reference: Wachter & Biegler (2006), "On the implementation of an
//            interior-point filter line-search algorithm for large-scale
//            nonlinear programming", Section 2.3, eqs. (6), (8);
//            Fletcher & Leyffer (2002), "Nonlinear programming without a
//            penalty function", Math. Program. 91:239-269.

#include "argmin/detail/filter_acceptance.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using argmin::detail::filter_set;

// A trial (f, h) is dominated by entry (f_j, h_j) -- and therefore
// rejected -- iff BOTH envelope conditions hold (W&B 2006 eq. (6)):
//     f >= f_j - gamma_f * h_j
//     h >= (1 - gamma_h) * h_j
// With a single entry (f_j = 1.0, h_j = 2.0) and margins
// gamma_f = gamma_h = 0.1 the thresholds are:
//     f-threshold: 1.0 - 0.1 * 2.0 = 0.8
//     h-threshold: (1 - 0.1) * 2.0 = 1.8
// so a trial is acceptable iff (f < 0.8) OR (h < 1.8), provided its
// violation stays under the ceiling h_max.
TEST_CASE("filter dominance truth table with envelope margins",
          "[filter_acceptance]")
{
    filter_set<double> filter(0.1, 0.1);
    filter.initialize(100.0);
    filter.add(1.0, 2.0);

    // Both thresholds crossed strictly -> dominated -> rejected.
    CHECK_FALSE(filter.is_acceptable(0.9, 1.9));
    // f below its threshold -> not dominated -> acceptable.
    CHECK(filter.is_acceptable(0.7, 1.9));
    // h below its threshold -> not dominated -> acceptable.
    CHECK(filter.is_acceptable(0.9, 1.7));
    // Boundary: both conditions hold with equality (>= is inclusive) ->
    // rejected.
    CHECK_FALSE(filter.is_acceptable(0.8, 1.8));
    // Strictly dominated in both legs.
    CHECK_FALSE(filter.is_acceptable(5.0, 5.0));
    // Origin dominates the entry in neither leg (f and h both below
    // threshold) -> acceptable.
    CHECK(filter.is_acceptable(0.0, 0.0));
}

TEST_CASE("filter h_max ceiling rejects excess violation regardless of f",
          "[filter_acceptance]")
{
    filter_set<double> filter(0.1, 0.1);
    filter.initialize(3.0);

    // Empty filter, violation above the ceiling -> rejected even for an
    // arbitrarily good objective.
    CHECK_FALSE(filter.is_acceptable(-1.0e6, 3.5));
    // Same objective, violation just under the ceiling -> acceptable.
    CHECK(filter.is_acceptable(-1.0e6, 2.9));
    // Boundary is inclusive: h == h_max is not "> h_max", so acceptable.
    CHECK(filter.is_acceptable(0.0, 3.0));
}

TEST_CASE("filter add prunes dominated entries and shrinks the frontier",
          "[filter_acceptance]")
{
    filter_set<double> filter(0.1, 0.1);
    filter.initialize(100.0);

    filter.add(2.0, 2.0);
    CHECK(filter.size() == 1);

    // (1.0, 1.0) dominates (2.0, 2.0) in the add() sense
    // (e.objective >= f AND e.violation >= h), so the older entry is
    // pruned and the frontier stays size 1.
    filter.add(1.0, 1.0);
    CHECK(filter.size() == 1);

    // (0.5, 3.0) trades objective for violation -- it dominates neither
    // (1.0,1.0) nor vice versa -- so both remain on the frontier.
    filter.add(0.5, 3.0);
    CHECK(filter.size() == 2);
}

// Reset semantics. clear() is the per-run reset used when a solver starts
// a fresh optimization from a stored filter_set. A cleared filter must
// behave like a freshly constructed one: the previous run's violation
// ceiling h_max must NOT carry over.
//
// RED against the current substrate: clear() drops the entries but
// retains h_max_, so a trial whose violation exceeds the *prior* run's
// ceiling is still rejected after the reset. The acceptance check below
// therefore fails until clear() also restores the default ceiling.
TEST_CASE("filter reset must not retain the previous run's h_max ceiling",
          "[filter_acceptance]")
{
    filter_set<double> filter(0.1, 0.1);
    const double fresh_ceiling = filter.h_max();  // default construction value

    filter.initialize(3.0);  // tighten the ceiling for the first run
    filter.add(1.0, 1.0);
    filter.clear();          // reset for a new run

    // A fresh filter has the default ceiling; a trial with violation 5.0
    // (above the retired 3.0 ceiling but far below the default) belongs
    // in an empty filter and must be accepted.
    CHECK(filter.h_max() == Approx(fresh_ceiling));
    CHECK(filter.is_acceptable(0.0, 5.0));
}
