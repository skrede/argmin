#ifndef HPP_GUARD_ARGMIN_SCHEDULE_TIME_BOXED_SCHEDULE_H
#define HPP_GUARD_ARGMIN_SCHEDULE_TIME_BOXED_SCHEDULE_H

#include "argmin/result/step_result.h"

#include <chrono>
#include <cstddef>

namespace argmin
{

// Time-boxed scheduling policy for basic_solver_group.
//
// Gives each solver a fixed time slice. When the slice is exhausted,
// moves to the next solver. Uses steady_clock (monotonic, not affected
// by wall-clock adjustments).
//
// Like fallback_schedule, select() does not know about solver retirement
// and can return an already-retired index between slice expirations.
// basic_solver_group::step() skip-scans past a retired index; this
// schedule only needs to keep producing its normal time-sliced sequence.

struct time_boxed_schedule
{
    std::chrono::microseconds time_slice{1000};

    void reset()
    {
        current_index_ = 0;
        slice_started_ = false;
    }

    std::size_t select(std::size_t num_solvers)
    {
        auto now = std::chrono::steady_clock::now();

        if(!slice_started_)
        {
            slice_start_ = now;
            slice_started_ = true;
        }

        if(now - slice_start_ >= time_slice)
        {
            current_index_ = (current_index_ + 1) % num_solvers;
            slice_start_ = now;
        }

        return current_index_;
    }

    template <typename Scalar>
    void notify(const step_result<Scalar>&)
    {}

private:
    std::size_t current_index_{0};
    std::chrono::steady_clock::time_point slice_start_{};
    bool slice_started_{false};
};

}

#endif
