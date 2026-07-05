#ifndef HPP_GUARD_ARGMIN_SCHEDULE_FALLBACK_SCHEDULE_H
#define HPP_GUARD_ARGMIN_SCHEDULE_FALLBACK_SCHEDULE_H

#include "argmin/result/step_result.h"

#include <cstddef>

namespace argmin
{

// Fallback scheduling policy for basic_solver_group.
//
// Runs the first solver until it stalls (stall_threshold consecutive
// non-improving steps), then switches to the next solver. If all solvers
// stall, cycles back to the first.
//
// select() does not know about solver retirement -- it can return an
// already-retired index (e.g. the current solver retires via roundoff /
// divergence / convergence before ever stalling out on this schedule's own
// terms). basic_solver_group::step() is responsible for skip-scanning past
// a retired index returned here; this schedule only needs to keep
// producing its normal stall-driven sequence.

struct fallback_schedule
{
    int stall_threshold{10};

    void reset()
    {
        current_index_ = 0;
        stall_counter_ = 0;
    }

    std::size_t select(std::size_t) { return current_index_; }

    template <typename Scalar>
    void notify(const step_result<Scalar>& result)
    {
        notify_impl(result.improved, num_solvers_);
    }

    void set_num_solvers(std::size_t n) { num_solvers_ = n; }

private:
    void notify_impl(bool improved, std::size_t num_solvers)
    {
        if(improved)
        {
            stall_counter_ = 0;
        }
        else
        {
            ++stall_counter_;
            if(stall_counter_ >= stall_threshold)
            {
                current_index_ = (current_index_ + 1) % num_solvers;
                stall_counter_ = 0;
            }
        }
    }

    std::size_t current_index_{0};
    std::size_t num_solvers_{1};
    int stall_counter_{0};
};

}

#endif
