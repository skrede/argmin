#ifndef HPP_GUARD_ARGMIN_SOLVER_SQP_MODE_H
#define HPP_GUARD_ARGMIN_SOLVER_SQP_MODE_H

#include <cstdint>

namespace argmin
{

// argmin variant: closed-set algorithm-variant selector for the line-search
//                 SQP family (kraft_slsqp, nw_sqp, filter_slsqp,
//                 filter_nw_sqp; tr_sqp adopts in a future milestone).
//                 `accurate` preserves baseline per-knob defaults; `fast`
//                 selects looser-tolerance / dead-branch-stripped per-knob
//                 defaults sized to a per-call NMPC-style budget.
//
// Reference: KNITRO commercial NLP solver mode-system (precedent for SQP
//            fast/accurate modes in commercial solvers); the closed-variant
//            convention also matches argmin's solver/alternative/cmaes/,
//            gcmma/, isres/ idiom of treating algorithmic variants as
//            separate types.
enum class sqp_mode : std::uint8_t { accurate, fast };

}

#endif
