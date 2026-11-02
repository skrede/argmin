#ifndef HPP_GUARD_ARGMIN_TESTS_RELEASE_SAFETY_WRONG_LENGTH_START_PROBE_H
#define HPP_GUARD_ARGMIN_TESTS_RELEASE_SAFETY_WRONG_LENGTH_START_PROBE_H

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
void check_start_is_rejected(Solver&& solver)
{
    CHECK(solver.status() == solver_status::invalid_problem);
    CHECK(solver.step().policy_status == solver_status::invalid_problem);
    const auto result = solver.solve();
    CHECK(result.status == solver_status::invalid_problem);
    CHECK(result.iterations == 0);
    CHECK(result.function_evaluations == 0);
}

template <typename Solver>
void check_same_first_step(Solver&& from_dynamic, Solver&& from_own)
{
    CHECK(from_dynamic.status() == from_own.status());
    CHECK(from_dynamic.state().x == from_own.state().x);
    const auto dynamic_step = from_dynamic.step();
    const auto own_step = from_own.step();
    CHECK_FALSE(own_step.policy_status == solver_status::invalid_problem);
    CHECK(dynamic_step.policy_status == own_step.policy_status);
    CHECK(from_dynamic.state().x == from_own.state().x);
}

struct wrong_length_start_probe
{
    std::vector<std::string_view> visited;

    template <typename Problem, typename MakeSolver>
    void operator()(const cases::case_id& id, const Problem& problem, const solver_options<>&, const MakeSolver& make_solver)
    {
        INFO("case " << id.key << " | " << id.problem_class << " | " << id.problem);
        visited.push_back(id.key);
        const Eigen::VectorXd start = problem.initial_point();
        check_start_is_rejected(make_solver(Eigen::VectorXd::Constant(start.size() - 1, 0.5).eval()));
        check_start_is_rejected(make_solver(Eigen::VectorXd::Constant(start.size() + 1, 0.5).eval()));
        check_same_first_step(make_solver(start), make_solver());
    }
};

template <std::size_t K>
void check_every_key_constructed(const wrong_length_start_probe& probe, const std::array<std::string_view, K>& keys)
{
    for(const std::string_view key : keys)
    {
        INFO("solver key with no wrong-length construction probe: " << key);
        CHECK(std::ranges::find(probe.visited, key) != probe.visited.end());
    }
}

}

#endif
