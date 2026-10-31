#ifndef HPP_GUARD_ARGMIN_TESTS_RELEASE_SAFETY_WRONG_LENGTH_RESET_PROBE_H
#define HPP_GUARD_ARGMIN_TESTS_RELEASE_SAFETY_WRONG_LENGTH_RESET_PROBE_H

#include "eigen_trap.h"

#include "cases/case_options.h"

#include "argmin/solver/options.h"

#include "argmin/result/status.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <vector>
#include <cstddef>
#include <algorithm>
#include <string_view>

namespace argmin::release_safety
{

template <typename Solver>
solver_status restart(Solver& solver, const Eigen::VectorXd& x0, bool clear)
{
    return clear ? solver.reset_clear(x0) : solver.reset(x0);
}

template <typename Solver>
void check_wrong_length_is_rejected(Solver& solver, Eigen::Index length, bool clear)
{
    const Eigen::VectorXd before = solver.state().x;
    CHECK(restart(solver, Eigen::VectorXd::Constant(length, 0.5), clear) == solver_status::invalid_problem);
    CHECK(solver.status() == solver_status::invalid_problem);
    CHECK(solver.state().x == before);
    CHECK(solver.step().policy_status == solver_status::invalid_problem);
    CHECK(solver.state().x == before);
}

template <typename Solver>
void check_valid_reset_recovers(Solver& solver, const Eigen::VectorXd& start, bool clear)
{
    CHECK(restart(solver, start, clear) == solver_status::running);
    CHECK(solver.status() == solver_status::running);
    CHECK_FALSE(solver.step().policy_status == solver_status::invalid_problem);
}

struct wrong_length_reset_probe
{
    std::vector<std::string_view> visited;

    template <typename Problem, typename MakeSolver>
    void operator()(const cases::case_id& id, const Problem& problem, const solver_options<>&, const MakeSolver& make_solver)
    {
        INFO("case " << id.key << " | " << id.problem_class << " | " << id.problem);
        visited.push_back(id.key);
        auto solver = make_solver();
        const Eigen::VectorXd start = problem.initial_point();
        for(const bool clear : {false, true})
        {
            check_wrong_length_is_rejected(solver, start.size() - 1, clear);
            check_wrong_length_is_rejected(solver, start.size() + 1, clear);
            check_valid_reset_recovers(solver, start, clear);
        }
    }
};

template <std::size_t K>
void check_every_key_visited(const wrong_length_reset_probe& probe, const std::array<std::string_view, K>& keys)
{
    for(const std::string_view key : keys)
    {
        INFO("solver key with no wrong-length reset probe: " << key);
        CHECK(std::ranges::find(probe.visited, key) != probe.visited.end());
    }
}

}

#endif
