#ifndef HPP_GUARD_ARGMIN_OPTIONS_QP_OPTIONS_H
#define HPP_GUARD_ARGMIN_OPTIONS_QP_OPTIONS_H

#include <cstdint>

namespace argmin
{

// Active-set QP solver parameters.
//
// Used by the line-search SQP policies (kraft_slsqp, nw_sqp, filter_slsqp,
// filter_nw_sqp) and any future helper consuming the same QP subproblem
// shape.
//
// The brace-initialized literals here are the accurate-mode fallback used
// for any default-constructed qp_options{} instance. Per-policy per-mode
// defaults are surfaced through each policy's options_type via a designated
// initializer of the form
//   qp_options qp{
//       .max_iterations = default_qp_max_iterations,
//       .tolerance      = default_qp_tolerance,
//   };
// where `default_qp_*` are static-constexpr members on the policy that
// depend on the closed-set Mode NTTP. Direct-field reads (no value_or
// indirection) at every consumer site.
//
// Reference: NLopt slsqp.c lsq_ / lsei_ iter-cap convention
//            (production-library precedent for the 200-iter / 1e-12-tol
//            accurate-mode fallback);
//            Kraft 1988 DFVLR-FB 88-28 Section 3.2 (QP cast accuracy via
//            acc^2 convergence threshold).
struct qp_options
{
    std::uint16_t max_iterations{200};
    double tolerance{1e-12};
};

}

#endif
