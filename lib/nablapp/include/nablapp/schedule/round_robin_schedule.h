#ifndef HPP_GUARD_NABLAPP_SCHEDULE_ROUND_ROBIN_SCHEDULE_H
#define HPP_GUARD_NABLAPP_SCHEDULE_ROUND_ROBIN_SCHEDULE_H

#include "nablapp/result/step_result.h"

#include <cstddef>

namespace nablapp
{

// Round-robin scheduling policy for basic_solver_group.
//
// Rotates through solvers: step solver 0, then 1, ..., then back to 0.
// Each solver gets exactly one step per round.

struct round_robin_schedule
{
    void reset() { current_index_ = 0; }

    std::size_t select(std::size_t num_solvers)
    {
        std::size_t idx = current_index_;
        current_index_ = (current_index_ + 1) % num_solvers;
        return idx;
    }

    template <typename Scalar>
    void notify(const step_result<Scalar>&)
    {}

private:
    std::size_t current_index_{0};
};

}

#endif
