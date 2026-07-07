#ifndef HPP_GUARD_ARGMIN_RESULT_TIMED_SOLVE_RESULT_H
#define HPP_GUARD_ARGMIN_RESULT_TIMED_SOLVE_RESULT_H

#include "argmin/types.h"
#include "argmin/result/solve_result.h"

#include <chrono>

namespace argmin
{

// Wall-clock-stamped result returned by the time-budget drivers.
//
// Publicly derives from solve_result so it carries every convergence
// diagnostic of the base plus the measured wall_time. The public derivation
// also makes timed_solve_result convertible-to solve_result: a time driver
// therefore satisfies the same nlp_solver / convertible_to<solve_result>
// surface the step driver does, without duplicating the diagnostic fields and
// without dragging <chrono> back into the base result.

template <typename Scalar = double, int N = dynamic_dimension>
struct timed_solve_result : solve_result<Scalar, N>
{
    std::chrono::steady_clock::duration wall_time{};
};

}

#endif
