#ifndef HPP_GUARD_ARGMIN_DETAIL_FILTER_ACCEPTANCE_H
#define HPP_GUARD_ARGMIN_DETAIL_FILTER_ACCEPTANCE_H

// Filter acceptance test and switching condition for filter-based SQP.
//
// The filter maintains a set of pairs (f, h) representing previously
// rejected combinations of objective value and constraint violation.
// A trial point is acceptable if it is not dominated by any filter entry
// and does not exceed the maximum constraint violation threshold.
//
// The switching condition distinguishes f-type iterations (near-feasible,
// aim for objective decrease) from h-type iterations (infeasible, aim
// for constraint violation decrease).
//
// Reference: Wachter & Biegler 2006, "On the implementation of an
//            interior-point filter line-search algorithm for large-scale
//            nonlinear programming", Section 2.3, eqs. (6), (8), (14);
//            Fletcher & Leyffer 2002, "Nonlinear programming without a
//            penalty function", Math. Program. 91:239-269;
//            N&W Section 15.5 (filter methods).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace argmin::detail
{

#ifdef ARGMIN_FILTER_CAPACITY_SWEEP
// Sweep-only instrumentation: a process-global configuration + high-water
// record that lets an out-of-tree capacity/drop-policy sweep drive the real
// filter policies (whose filter_set is a private state member) without any
// production API surface. Compiled ONLY when ARGMIN_FILTER_CAPACITY_SWEEP is
// defined (the capacity sweep driver); the shipping library never sees this.
//
//   capacity        -- entry cap enforced by filter_set::add(); set huge to
//                      measure the natural (unbounded) high-water.
//   overflow_policy  -- 0 reject the incoming trial (do not augment),
//                       1 drop the maximal-violation entry, 2 FIFO ring
//                       (drop the oldest entry).
//   highwater        -- maximum entries_.size() observed since the last reset.
//   evictions        -- number of at-capacity drops/rejects since reset.
struct filter_sweep_state_t
{
    std::size_t capacity{1u << 30};
    int overflow_policy{0};
    std::size_t highwater{0};
    std::size_t evictions{0};
};

inline filter_sweep_state_t& filter_sweep_state()
{
    static filter_sweep_state_t state;
    return state;
}
#endif

// A single filter entry: an (objective, violation) pair that has been
// recorded as unacceptable. Trial points dominated by any entry are
// rejected.
template <typename Scalar = double>
struct filter_entry
{
    Scalar objective;
    Scalar violation;
};

// Filter set maintaining the non-dominated frontier of rejected
// (objective, violation) pairs.
//
// Reference: Wachter & Biegler 2006, Section 2.3.
template <typename Scalar = double>
class filter_set
{
public:
    // Construct a filter_set with explicit envelope parameters.
    //
    // gamma_f and gamma_h are the asymmetric envelope margins for
    // filter dominance (Wachter & Biegler 2006 Section 2.3, eq. 6).
    // Defaults 1e-5 / 1e-5 preserve the v0.2.1 behaviour; per-policy
    // tuning is exposed via filter_*_policy::options_type.
    //
    // argmin variant: independent gamma_f and gamma_h (Wachter &
    //                 Biegler 2006 Section 2.3); Fletcher-Leyffer 2002
    //                 Section 5 originally presents a single shared
    //                 margin; rationale: HS043 geometry suggests
    //                 asymmetric defaults.
    //
    // Reference: Wachter & Biegler 2006 Section 2.3, eq. (6);
    //            Fletcher & Leyffer 2002 Section 5.
    explicit filter_set(Scalar gamma_f = Scalar(1e-5),
                        Scalar gamma_h = Scalar(1e-5))
        : gamma_f_(gamma_f), gamma_h_(gamma_h) {}

    // Bounded (objective, violation) frontier capacity. add() caps the
    // frontier at this many entries so its push_back never grows the vector
    // past the initialize()-time reservation -- steady-state allocation-free
    // -- and the worst-case entry count is bounded for the real-time claim.
    //
    // The value is chosen by an empirical sweep of the filter high-water
    // count across the constrained Hock-Schittkowski suite on every filter
    // policy (the line-search and both trust-region modes): dominance pruning
    // already keeps the frontier small, with a measured worst case of 29
    // entries. 256 sits at roughly 8x that high-water with round headroom,
    // so the cap is never reached on the suite and convergence is identical
    // to an unbounded frontier by measurement; it also matches the historical
    // reservation, leaving the memory footprint unchanged
    // (256 * 16 B = 4 KB on x86_64 with double).
    static constexpr std::size_t capacity = 256;

    // Initialize the filter with a maximum constraint violation ceiling
    // and an optional violation floor applied to stored entries.
    //
    // Callers set h_max = 1e4 * max(1, h_0) where h_0 is the initial
    // constraint violation, and theta_min = 1e-4 * max(1, h_0) when the
    // consuming policy opts into the floored-entry variant. The zero
    // default preserves the un-floored behavior bit-for-bit for callers
    // that pass only h_max.
    //
    // Pre-reserves the underlying entry vector to the bounded frontier
    // capacity so that hot-path add() calls never trigger geometric vector
    // growth (which would allocate). Combined with the at-capacity policy in
    // add(), this makes the filter both steady-state allocation-free and
    // worst-case bounded -- the last unbounded container on the real-time
    // filter-SQP path. See the capacity constant for the empirical basis.
    //
    // Reference: Wachter & Biegler 2006 Section 4 (theta_max and
    //            theta_min scaling against max{1, theta(x_0)}).
    void initialize(Scalar h_max, Scalar theta_min = Scalar(0))
    {
        entries_.clear();
#ifdef ARGMIN_FILTER_CAPACITY_SWEEP
        const std::size_t reserve_to = filter_sweep_state().capacity;
        if(entries_.capacity() < reserve_to)
            entries_.reserve(reserve_to);
#else
        if(entries_.capacity() < capacity)
            entries_.reserve(capacity);
#endif
        h_max_ = h_max;
        theta_min_ = theta_min;
    }

    // Re-set the envelope on an existing filter_set without
    // re-construction. Use case: a policy holds filter_set as a state
    // member and threads options into it during init().
    //
    // Reference: Wachter & Biegler 2006 Section 2.3, eq. (6).
    void set_envelope(Scalar gamma_f, Scalar gamma_h)
    {
        gamma_f_ = gamma_f;
        gamma_h_ = gamma_h;
    }

    // Test whether a trial point (f_trial, h_trial) is acceptable to
    // the filter. Returns true if the point is not dominated by any
    // existing entry and does not exceed h_max.
    //
    // A trial point is dominated by entry (f_j, h_j) when BOTH:
    //   f_trial >= f_j - gamma_f * h_j
    //   h_trial >= (1 - gamma_h) * h_j
    //
    // Reference: Wachter & Biegler 2006, eq. (6).
    bool is_acceptable(Scalar f_trial, Scalar h_trial) const
    {
        if(h_trial > h_max_)
            return false;

        for(const auto& entry : entries_)
        {
            bool f_dominated = f_trial >= entry.objective - gamma_f_ * entry.violation;
            bool h_dominated = h_trial >= (Scalar(1) - gamma_h_) * entry.violation;
            if(f_dominated && h_dominated)
                return false;
        }
        return true;
    }

    // Add a point to the filter, pruning any entries dominated by the
    // new point. An existing entry (f_j, h_j) is dominated if
    // f_j >= f AND h_j >= h.
    //
    // The stored violation is floored at theta_min_ so h-domination is
    // never vacuous: an exactly-feasible point enters the filter with
    // violation theta_min_, leaving trials with
    // h_trial < (1 - gamma_h) * theta_min_ acceptable on the h leg.
    // Without the floor an h = 0 entry makes the h-domination condition
    // hold for every trial, degenerating acceptance to strict monotone-f
    // anchored at any exactly-feasible visited point (a permanent gate
    // with no escape at Maratos-type stalls). The default theta_min_ = 0
    // reproduces the un-floored behavior exactly.
    //
    // Reference: Wachter & Biegler 2006 Section 2.3 (theta_min gating)
    //            and Section 4 (theta_min = 1e-4 * max{1, theta(x_0)}).
    void add(Scalar f, Scalar h)
    {
        const Scalar h_floored = std::max(h, theta_min_);
        entries_.erase(
            std::remove_if(entries_.begin(), entries_.end(),
                           [f, h_floored](const filter_entry<Scalar>& e) {
                               return e.objective >= f
                                      && e.violation >= h_floored;
                           }),
            entries_.end());

        // At-capacity policy: when the pruned frontier is already full, drop
        // the maximal-violation (loosest) entry before admitting the new one,
        // so the tight near-feasible guards that discriminate near the
        // optimum survive. On the swept suite the cap is never reached, so
        // this is a worst-case safeguard for pathological inputs; under
        // forced starvation it is the empirically least-disruptive heuristic
        // (fewer at-capacity events and fewer perturbed cells than either
        // rejecting the trial or a FIFO ring).
#ifdef ARGMIN_FILTER_CAPACITY_SWEEP
        const std::size_t cap = filter_sweep_state().capacity;
        const int overflow_policy = filter_sweep_state().overflow_policy;
#else
        constexpr std::size_t cap = capacity;
        constexpr int overflow_policy = 1;
#endif
        if(entries_.size() >= cap)
        {
#ifdef ARGMIN_FILTER_CAPACITY_SWEEP
            ++filter_sweep_state().evictions;
#endif
            if(overflow_policy == 0)
            {
                return; // reject the incoming trial: frontier unchanged
            }
            else if(overflow_policy == 1)
            {
                auto worst = std::max_element(
                    entries_.begin(), entries_.end(),
                    [](const filter_entry<Scalar>& a,
                       const filter_entry<Scalar>& b) {
                        return a.violation < b.violation;
                    });
                if(worst != entries_.end())
                    entries_.erase(worst);
            }
            else
            {
                entries_.erase(entries_.begin()); // FIFO ring: drop oldest
            }
        }
        entries_.push_back({f, h_floored});
#ifdef ARGMIN_FILTER_CAPACITY_SWEEP
        {
            auto& st = filter_sweep_state();
            if(entries_.size() > st.highwater)
                st.highwater = entries_.size();
        }
#endif
    }

    bool empty() const { return entries_.empty(); }

    std::uint16_t size() const
    {
        return static_cast<std::uint16_t>(entries_.size());
    }

    // Currently reserved backing-store capacity (entries). Exposes the
    // underlying vector's reservation so a steady-state no-reallocation
    // invariant can be observed directly, independent of Eigen.
    std::size_t capacity_reserved() const { return entries_.capacity(); }

    // Reset to the freshly-constructed state: entries dropped, the
    // violation ceiling restored to the default-construction value, and
    // the entry floor back to zero. A cleared filter must not retain
    // the previous run's h_max ceiling or theta_min floor -- callers
    // that need run-specific values re-initialize() after clear().
    void clear()
    {
        entries_.clear();
        h_max_ = Scalar(1e4);
        theta_min_ = Scalar(0);
    }

    Scalar h_max() const { return h_max_; }

    Scalar theta_min() const { return theta_min_; }

private:
    std::vector<filter_entry<Scalar>> entries_{};
    Scalar h_max_{Scalar(1e4)};
    Scalar theta_min_{Scalar(0)};
    Scalar gamma_f_{Scalar(1e-5)};
    Scalar gamma_h_{Scalar(1e-5)};
};

// Switching condition: determines whether the current iteration is
// f-type (optimality-seeking) or h-type (feasibility-seeking).
//
// Returns true (f-type) when all three conditions hold:
//   1. h_k <= delta * h_max           (near-feasible)
//   2. grad_f_dot_p < 0               (descent direction for f)
//   3. (-grad_f_dot_p)^s_f > delta * h_k^s_h  (sufficient predicted f-decrease)
//
// Reference: Wachter & Biegler 2006, eq. (14), Section 2.3.1.
template <typename Scalar>
bool is_f_type_iteration(Scalar h_k, Scalar grad_f_dot_p, Scalar h_max,
                         Scalar delta = Scalar(1e-4),
                         Scalar s_f = Scalar(2.3),
                         Scalar s_h = Scalar(1.0))
{
    if(h_k > delta * h_max)
        return false;
    if(grad_f_dot_p >= Scalar(0))
        return false;
    return std::pow(-grad_f_dot_p, s_f) > delta * std::pow(h_k, s_h);
}

// Wachter-Biegler switching-condition parameters. Literature-fixed
// defaults from W-B 2006 Section 4:
//   delta = 1, s_theta = 1.1, s_phi = 2.3, eta_phi = 1e-4.
//
// Reference: Wachter & Biegler 2006, Section 4 (default constants);
//            eq. (19) (switching condition); eq. (20) (Armijo-type
//            sufficient objective reduction).
template <typename Scalar = double>
struct wb_switching_params
{
    Scalar delta{Scalar(1)};
    Scalar s_theta{Scalar(1.1)};
    Scalar s_phi{Scalar(2.3)};
    Scalar eta_phi{Scalar(1e-4)};
};

// Verdict of the W-B filter acceptance with switching-condition
// classification.
//
//   accept  -- trial passes the filter memory check AND the
//              current-iterate margin (eq. 18).
//   f_type  -- switching condition (eq. 19) and the sufficient-
//              objective-reduction condition (eq. 20) both hold;
//              a genuine objective-driven step.
//   augment -- the filter must be augmented with the ACCEPTED step
//              (Algorithm A step A-7: augment iff the accepted step
//              is not f-type). Always false on rejection.
struct wb_acceptance_verdict
{
    bool accept;
    bool f_type;
    bool augment;
};

// Wachter-Biegler filter acceptance with switching-condition
// classification, trust-region form (unit step, alpha = 1 in eqs.
// 19/20 -- the composite step is accepted or rejected whole; radius
// management replaces backtracking).
//
// Acceptance requires BOTH:
//   1. Filter memory check (eq. 6 margins + h_max ceiling) against
//      the stored filter (Algorithm A step A-5.3).
//   2. Current-iterate margin (eq. 18): sufficient progress vs the
//      CURRENT iterate on at least one leg,
//          h_trial <= (1 - gamma_h) * h_current
//       or f_trial <= f_current - gamma_f * h_current.
//      Without this margin no monotone quantity exists across
//      f-type accepts (which do not augment the filter) and cycling
//      through stale filter gaps is possible.
//
// Classification (consumed by the caller's augmentation rule A-7):
//   switching (eq. 19, alpha = 1, gated by the theta_min floor of
//   W-B Section 2.3):
//       h_current <= theta_min
//       AND grad_f_dot_p < 0
//       AND (-grad_f_dot_p)^s_phi > delta * h_current^s_theta
//   sufficient reduction (eq. 20, alpha = 1):
//       f_trial <= f_current + eta_phi * grad_f_dot_p
//   f_type = switching AND sufficient reduction. f-type accepted
//   steps leave the filter unchanged (they are reversible and must
//   not pollute the filter); every other accepted step augments it.
//
// grad_f_dot_p is the DIRECTIONAL objective-model term grad_f^T p of
// eq. (19) -- an f-model quantity, not a Lagrangian-model reduction.
// For second-order-corrected trials the caller passes the ORIGINAL
// step's grad_f^T p (W-B Section 2.4: the switching test and the
// right-hand side of eq. 20 keep the uncorrected direction).
//
// Reference: Wachter & Biegler 2006, Math. Programming 106:25-57,
//            Section 2.3 eqs. (18)-(20), Algorithm A steps A-5.3,
//            A-5.4, A-7; Section 4 (default constants).
template <typename Scalar>
wb_acceptance_verdict wb_switching_acceptance(
    const filter_set<Scalar>& filter,
    Scalar f_current, Scalar h_current,
    Scalar f_trial, Scalar h_trial,
    Scalar grad_f_dot_p,
    Scalar gamma_f, Scalar gamma_h,
    Scalar theta_min,
    const wb_switching_params<Scalar>& params = {})
{
    const bool filter_ok = filter.is_acceptable(f_trial, h_trial);

    const bool current_ok =
        h_trial <= (Scalar(1) - gamma_h) * h_current
        || f_trial <= f_current - gamma_f * h_current;

    const bool switching =
        h_current <= theta_min
        && grad_f_dot_p < Scalar(0)
        && std::pow(-grad_f_dot_p, params.s_phi)
               > params.delta * std::pow(h_current, params.s_theta);

    const bool sufficient =
        f_trial <= f_current + params.eta_phi * grad_f_dot_p;

    const bool accept = filter_ok && current_ok;
    const bool f_type = switching && sufficient;
    return {accept, f_type, accept && !f_type};
}

}

#endif
