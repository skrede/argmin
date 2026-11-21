#ifndef HPP_GUARD_ARGMIN_TESTS_RESULT_ORACLE_MEMBERSHIP_VISITOR_H
#define HPP_GUARD_ARGMIN_TESTS_RESULT_ORACLE_MEMBERSHIP_VISITOR_H

#include "cases/case_options.h"

#include "oracle/trajectory_membership.h"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

namespace argmin::cases
{

// Two twins built alike: one stepped to record its trajectory, one solved; the seeded stochastic cases make the twins
// draw the same stream.
struct membership_visitor
{
    template <typename Problem, typename Options, typename MakeSolver>
    void operator()(const case_id& id, const Problem& problem, const Options& options, const MakeSolver& make_solver)
    {
        auto stepped = make_solver();
        const std::vector<oracle::trajectory_row> rows = oracle::record_trajectory(stepped, problem, options.max_iterations);
        auto solved = make_solver();
        const auto result = solved.solve();
        INFO(id.key << ' ' << id.problem_class << ' ' << id.problem);
        CHECK(oracle::membership_failure(result, rows) == std::string{});
    }
};

}

#endif
