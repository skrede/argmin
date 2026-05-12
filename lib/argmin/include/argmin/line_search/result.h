#ifndef HPP_GUARD_ARGMIN_LINE_SEARCH_RESULT_H
#define HPP_GUARD_ARGMIN_LINE_SEARCH_RESULT_H

#include <cstddef>

namespace argmin
{

// Result of a line search computation.
//
// Returned by all line search algorithms (armijo, strong_wolfe, etc.).
// The caller checks `success` to determine whether the step satisfies
// the requested conditions.

template <typename Scalar = double>
struct line_search_result
{
    Scalar alpha{};
    Scalar value{};
    int evaluations{0};
    bool success{false};

    // Per-line-search diagnostics for telemetry. Defaulted-member
    // sub-struct so future fields (e.g., inf_eval_count) can be added
    // without touching existing call sites. Mirrors the
    // step_result::solver_diagnostics defaulted-sub-struct pattern at
    // result/step_result.h.
    //
    // Reference: Ipopt IpIpoptCalculatedQuantities::f_or_grad_returned_nan
    //            (NaN detection model; argmin variant is Armijo-only —
    //            non-finite phi observed at the line search becomes a
    //            backtrack trigger rather than a propagated tainted
    //            value, surfaced via this counter to callers).
    struct ls_diagnostics
    {
        std::size_t nan_eval_count{0};
    };
    ls_diagnostics diagnostics{};
};

}

#endif
