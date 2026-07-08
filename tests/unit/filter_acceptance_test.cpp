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
// ceiling h_max and entry floor theta_min must NOT carry over.
//
// This case shipped RED against the pre-fix substrate (clear() dropped
// the entries but retained h_max_) under a [!shouldfail] disposition;
// the fix landed -- clear() now restores the default-construction
// ceiling and floor -- and the instrument fired green, so the tag is
// removed and the case asserts the corrected semantics.
TEST_CASE("filter reset must not retain the previous run's h_max ceiling",
          "[filter_acceptance]")
{
    filter_set<double> filter(0.1, 0.1);
    const double fresh_ceiling = filter.h_max();  // default construction value

    filter.initialize(3.0, 0.5);  // tighten ceiling + raise floor, run 1
    filter.add(1.0, 1.0);
    filter.clear();               // reset for a new run

    // A fresh filter has the default ceiling; a trial with violation 5.0
    // (above the retired 3.0 ceiling but far below the default) belongs
    // in an empty filter and must be accepted.
    CHECK(filter.h_max() == Approx(fresh_ceiling));
    CHECK(filter.is_acceptable(0.0, 5.0));

    // The retired run's entry floor must not survive either: a fresh
    // add() of an exactly-feasible pair stores violation 0, so its
    // h-domination is vacuous again (the default un-floored behavior)
    // rather than floored at the prior run's 0.5.
    CHECK(filter.theta_min() == 0.0);
    filter.add(0.5, 0.0);
    CHECK_FALSE(filter.is_acceptable(1.0, 0.4));
}

// theta_min entry floor (Wachter & Biegler 2006 Section 4:
// theta_min = 1e-4 * max{1, theta(x_0)}). add() stores
// {f, max(h, theta_min)} so h-domination is never vacuous: an
// exactly-feasible visited point can no longer turn the filter into a
// permanent monotone-f gate.
//
// Truth table with gamma_f = gamma_h = 0.1, theta_min = 1e-4, and an
// exactly-feasible entry (f_j, h_j) = (1.0, 0.0) stored floored as
// (1.0, 1e-4). W&B 2006 eq. (6) thresholds on the FLOORED entry:
//     f-threshold: 1.0 - 0.1 * 1e-4 = 0.99999
//     h-threshold: (1 - 0.1) * 1e-4 = 9.0e-5
// so a trial is acceptable iff (f < 0.99999) OR (h < 9.0e-5).
TEST_CASE("filter theta_min floor keeps h-domination non-vacuous",
          "[filter_acceptance]")
{
    filter_set<double> filter(0.1, 0.1);
    filter.initialize(100.0, 1e-4);
    filter.add(1.0, 0.0);  // exactly-feasible entry, stored as (1.0, 1e-4)

    // (i) A strictly-feasible descent trial with tiny violation is
    // acceptable on the h leg even with a WORSE objective -- pre-floor
    // this trial was rejected (h-domination vacuous against h_j = 0,
    // f-domination at any f >= f_j), the permanent monotone-f gate.
    CHECK(filter.is_acceptable(1.5, 5e-7));
    // Same escape at exactly-zero trial violation, arbitrary f.
    CHECK(filter.is_acceptable(1e6, 0.0));
    // Above the floored h-threshold with a dominated objective ->
    // rejected (the floor does not disable dominance).
    CHECK_FALSE(filter.is_acceptable(1.0, 9.1e-5));
    // Above the floored h-threshold but below the f-threshold ->
    // acceptable on the f leg.
    CHECK(filter.is_acceptable(0.5, 9.1e-5));
}

// (ii) The floor must not weaken dominance for entries whose violation
// already exceeds theta_min: max(h, theta_min) = h there, so the
// stored entry -- and every eq. (6) verdict against it -- is identical
// to the un-floored table (same rows as the base truth-table case).
TEST_CASE("filter theta_min floor leaves entries above the floor intact",
          "[filter_acceptance]")
{
    filter_set<double> filter(0.1, 0.1);
    filter.initialize(100.0, 1e-4);
    filter.add(1.0, 2.0);  // h = 2.0 > theta_min: stored unchanged

    CHECK_FALSE(filter.is_acceptable(0.9, 1.9));
    CHECK(filter.is_acceptable(0.7, 1.9));
    CHECK(filter.is_acceptable(0.9, 1.7));
    CHECK_FALSE(filter.is_acceptable(0.8, 1.8));
    CHECK_FALSE(filter.is_acceptable(5.0, 5.0));
    CHECK(filter.is_acceptable(0.0, 0.0));
}

// (iii) Default theta_min = 0 (single-argument initialize) reproduces
// the un-floored table bit-for-bit -- the line-search filter family
// (filter_slsqp / filter_nw_sqp) passes no floor and must be
// behaviorally untouched until its own hardening phase wires one.
// In particular the pre-floor vacuous-h-domination behavior of an
// h = 0 entry is preserved under the zero default.
TEST_CASE("filter default zero floor reproduces the un-floored table",
          "[filter_acceptance]")
{
    filter_set<double> filter(0.1, 0.1);
    filter.initialize(100.0);  // theta_min defaults to 0
    filter.add(1.0, 0.0);      // stored un-floored as (1.0, 0.0)

    // h-domination vacuous (h_trial >= 0 always): acceptance
    // degenerates to strict monotone-f, exactly the legacy verdicts.
    CHECK_FALSE(filter.is_acceptable(1.5, 5e-7));
    CHECK_FALSE(filter.is_acceptable(1.0, 0.0));
    CHECK(filter.is_acceptable(0.9, 5e-7));

    // Base truth-table rows unchanged under the zero default.
    filter.clear();
    filter.initialize(100.0);
    filter.add(1.0, 2.0);
    CHECK_FALSE(filter.is_acceptable(0.9, 1.9));
    CHECK(filter.is_acceptable(0.7, 1.9));
    CHECK(filter.is_acceptable(0.9, 1.7));
    CHECK_FALSE(filter.is_acceptable(0.8, 1.8));
}

// Bounded-capacity regression pins. The filter's (objective, violation)
// frontier is capped at filter_set::capacity so its hot-path add() never
// grows the underlying vector past the initialize()-time reservation
// (steady-state allocation-free) and the worst-case entry count is bounded.
// The capacity value and the drop heuristic were selected by an empirical
// high-water sweep across the constrained HS suite (measured worst case 29
// entries; the shipped cap of 256 is never reached on the suite).

// (iv) Steady-state no-realloc: pushing strictly non-dominating entries up
// to the capacity does not reallocate the backing vector. std::vector's
// capacity() is observed directly (this is EIGEN-independent) -- the
// reservation is taken once at initialize() and the pointer/capacity stay
// put across every add() below the cap.
TEST_CASE("filter bounded capacity: no reallocation below the cap",
          "[filter_acceptance]")
{
    filter_set<double> filter(0.0, 0.0);
    filter.initialize(1e9);

    // The reservation is the fixed capacity, taken once.
    CHECK(filter.capacity_reserved() >= filter_set<double>::capacity);
    const std::size_t reserved = filter.capacity_reserved();

    // A strictly increasing-f, increasing-h stream is mutually
    // non-dominating: no entry prunes another, so the frontier grows by one
    // per add() until the cap. Fill to exactly the capacity.
    for(std::size_t i = 0; i < filter_set<double>::capacity; ++i)
    {
        const double x = static_cast<double>(i);
        filter.add(x, x + 1.0);
        // No add() below the cap may reallocate.
        CHECK(filter.capacity_reserved() == reserved);
    }
    CHECK(filter.size() == filter_set<double>::capacity);
}

// (v) At-capacity behavior: once full, admitting a new entry that prunes
// nothing drops the maximal-violation (loosest) frontier entry, keeps the
// size at the cap, and does not reallocate.
TEST_CASE("filter bounded capacity: at-capacity drops the loosest entry",
          "[filter_acceptance]")
{
    constexpr std::size_t cap = filter_set<double>::capacity;
    filter_set<double> filter(0.0, 0.0);
    filter.initialize(1e9);

    // Fill to capacity with a genuine non-dominated (Pareto) frontier:
    // entry i is (cap - i, i + 1), so f decreases as h increases and no add
    // prunes another. The maximal-violation entry is the lowest-f corner
    // (1, cap), which uniquely guards the low-objective / high-violation
    // region.
    for(std::size_t i = 0; i < cap; ++i)
    {
        const double f = static_cast<double>(cap - i);
        const double h = static_cast<double>(i + 1);
        filter.add(f, h);
    }
    const std::size_t reserved = filter.capacity_reserved();
    REQUIRE(filter.size() == cap);

    const double cap_d = static_cast<double>(cap);

    // The loosest (max-violation) corner (1, cap) uniquely guards (1.5, cap):
    // no other entry has both f <= 1.5 and h <= cap. Rejected while present.
    CHECK_FALSE(filter.is_acceptable(1.5, cap_d));

    // Admit a high-objective, low-violation point. It dominates no stored
    // entry (its f exceeds every entry's), so pruning removes nothing, the
    // frontier is full, and the at-capacity policy fires: drop the loosest
    // (max-violation) entry (1, cap).
    filter.add(cap_d + 50.0, 0.5);

    // Size stays pinned at the cap; no reallocation.
    CHECK(filter.size() == cap);
    CHECK(filter.capacity_reserved() == reserved);

    // The dropped loosest entry (1, cap) no longer guards: (1.5, cap) is now
    // acceptable (no surviving entry dominates it).
    CHECK(filter.is_acceptable(1.5, cap_d));

    // The freshly admitted entry is present and guards as expected.
    CHECK_FALSE(filter.is_acceptable(cap_d + 50.0, 0.5));
}

// (vi) Dominance pruning is unchanged below the cap: a new point that
// dominates existing entries prunes them, and the frontier stays a
// non-dominated set exactly as before the bound was added.
TEST_CASE("filter bounded capacity: dominance pruning unchanged below the cap",
          "[filter_acceptance]")
{
    filter_set<double> filter(0.0, 0.0);
    filter.initialize(1e9);

    filter.add(5.0, 5.0);
    filter.add(4.0, 6.0);
    filter.add(6.0, 4.0);
    CHECK(filter.size() == 3);

    // (3, 3) dominates all three stored entries (lower f AND lower h than
    // each): the pruning erases them and only (3, 3) remains.
    filter.add(3.0, 3.0);
    CHECK(filter.size() == 1);

    // The surviving frontier is exactly (3, 3): a trial dominated by it is
    // rejected, and any trial below either leg is accepted.
    CHECK_FALSE(filter.is_acceptable(3.0, 3.0));
    CHECK(filter.is_acceptable(2.9, 3.0));
    CHECK(filter.is_acceptable(3.0, 2.9));
}
