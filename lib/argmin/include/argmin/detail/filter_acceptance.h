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
#include <cstdint>
#include <vector>

namespace argmin::detail
{

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
    // Initialize the filter with a maximum constraint violation ceiling.
    //
    // Callers set h_max = 1e4 * max(1, h_0) where h_0 is the initial
    // constraint violation.
    //
    // Reference: Wachter & Biegler 2006, eq. (8).
    void initialize(Scalar h_max)
    {
        entries_.clear();
        h_max_ = h_max;
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
    void add(Scalar f, Scalar h)
    {
        entries_.erase(
            std::remove_if(entries_.begin(), entries_.end(),
                           [f, h](const filter_entry<Scalar>& e) {
                               return e.objective >= f && e.violation >= h;
                           }),
            entries_.end());
        entries_.push_back({f, h});
    }

    bool empty() const { return entries_.empty(); }

    std::uint16_t size() const
    {
        return static_cast<std::uint16_t>(entries_.size());
    }

    void clear() { entries_.clear(); }

    Scalar h_max() const { return h_max_; }

private:
    std::vector<filter_entry<Scalar>> entries_{};
    Scalar h_max_{Scalar(1e4)};
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

}

#endif
