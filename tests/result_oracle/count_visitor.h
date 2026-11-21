#ifndef HPP_GUARD_ARGMIN_TESTS_RESULT_ORACLE_COUNT_VISITOR_H
#define HPP_GUARD_ARGMIN_TESTS_RESULT_ORACLE_COUNT_VISITOR_H

#include "cases/case_options.h"

#include "oracle/callback_tracer.h"

#include <catch2/catch_test_macros.hpp>

#include <span>
#include <cstdint>
#include <algorithm>
#include <string_view>

namespace argmin::cases
{

// Solves each case of the listed keys over a traced copy of its problem, from construction on, so the reported count
// must include the calls of the policy's init and of the solve loop itself.
struct count_visitor
{
    std::span<const std::string_view> keys;
    std::uint32_t judged;

    template <typename Problem, typename Options, typename MakeSolver, typename MakeSolverFor>
    void operator()(const case_id& id, const Problem& problem, const Options&, const MakeSolver&, const MakeSolverFor& make_solver_for)
    {
        if(std::ranges::find(keys, id.key) == keys.end())
            return;
        const oracle::traced<Problem> traced{problem};
        auto solver = make_solver_for(traced);
        const auto result = solver.solve();
        ++judged;
        INFO(id.key << ' ' << id.problem_class << ' ' << id.problem);
        CHECK(result.function_evaluations == traced.objective_calls());
    }
};

}

#endif
