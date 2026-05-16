#ifndef HPP_GUARD_ARGMIN_SOLVER_SQP_MODE_H
#define HPP_GUARD_ARGMIN_SOLVER_SQP_MODE_H

#include <cstdint>

namespace argmin
{

// argmin variant: closed-set algorithm-variant selector consumed by BOTH
//                 the line-search SQP family (kraft_slsqp, nw_sqp,
//                 filter_slsqp, filter_nw_sqp) and the trust-region SQP
//                 family (tr_sqp). `accurate` preserves baseline per-knob
//                 defaults; `fast` selects looser-tolerance / dead-branch-
//                 stripped per-knob defaults sized to a per-call NMPC-style
//                 budget. The cross-cutting convergence tolerances
//                 (default_gradient_tolerance, default_step_tolerance_rel,
//                 default_feasibility_tolerance) take the same per-mode
//                 values across both families so a downstream consumer
//                 racing line-search vs trust-region policies under
//                 basic_solver_group sees a uniform tolerance contract.
//                 The family-specific knobs differ in semantics:
//
//                 Line-search SQP `fast`: second-order-correction skip,
//                 tighter Armijo iter cap, BFGS-skip on non-positive
//                 curvature, smaller per-step budgets — sized for
//                 NMPC-style per-call wall-time budgets.
//
//                 Line-search SQP `accurate`: full second-order correction,
//                 larger Armijo iter cap, Powell-damped BFGS, tighter
//                 convergence tolerances — sized for IK-style accuracy
//                 budgets.
//
//                 Trust-region SQP `fast`: smaller Steihaug-CG inner-
//                 iteration cap, flatter Dembo-Eisenstat-Steihaug
//                 forcing sequence, BFGS-skip on non-positive curvature.
//                 No second-order-correction, Armijo, or merit-penalty
//                 analog — the trust-region family has none of these.
//
//                 Trust-region SQP `accurate`: larger Steihaug-CG inner-
//                 iteration cap, tighter Eisenstat-Walker forcing
//                 sequence, Powell-damped BFGS, tighter convergence
//                 tolerances.
//
//                 Trust-region ratio thresholds (eta_1, eta_2) and radius
//                 update factors (shrink, expand) are uniform across modes
//                 per literature consensus; per-mode dispatch on these
//                 knobs has no published precedent.
//
// Reference: KNITRO commercial NLP solver mode-system (precedent for SQP
//            fast/accurate modes in commercial solvers); the closed-variant
//            convention also matches argmin's solver/alternative/cmaes/,
//            gcmma/, isres/ idiom of treating algorithmic variants as
//            separate types.
//            Line-search family: Kraft 1988 DFVLR-FB 88-28; Nocedal and
//            Wright 2e Chapter 18 (line-search SQP framework).
//            Trust-region family: Nocedal and Wright 2e Section 18.5
//            Algorithm 18.4 (Byrd-Omojokun composite step); Lalee,
//            Nocedal, Plantenga 1998 SIAM J. Optim. 8(3):682-706;
//            Conn, Gould, Toint 2000 MOS-SIAM Trust-Region Methods
//            Chapters 7, 12, 17; Nocedal and Wright 2e Section 4.1
//            (universal trust-region ratio test and radius update).
//            Forcing-sequence dispatch: Nocedal and Wright 2e Section 7.3
//            (truncated CG); Eisenstat and Walker 1996 SIAM J. Sci.
//            Comput. 17(1):16-32 (adaptive forcing for inexact Newton);
//            Dembo, Eisenstat, Steihaug 1982 SIAM J. Numer. Anal.
//            19(2):400-408 (alternative forcing sequence).
enum class sqp_mode : std::uint8_t { accurate, fast };

}

#endif
