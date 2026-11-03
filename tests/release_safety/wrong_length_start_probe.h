#ifndef HPP_GUARD_ARGMIN_TESTS_RELEASE_SAFETY_WRONG_LENGTH_START_PROBE_H
#define HPP_GUARD_ARGMIN_TESTS_RELEASE_SAFETY_WRONG_LENGTH_START_PROBE_H

#include "eigen_trap.h"

#include "cases/case_options.h"

#include "argmin/solver/options.h"

#include "argmin/result/status.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <new>
#include <array>
#include <memory>
#include <vector>
#include <cstddef>
#include <optional>
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

struct first_step
{
    solver_status status;
    Eigen::VectorXd start;
    std::optional<solver_status> policy_status;
    Eigen::VectorXd x;
};

template <typename Solver>
class placed_solver
{
public:
    template <typename MakeSolver, typename... Start>
    placed_solver(std::array<std::byte, sizeof(Solver)>& storage, const MakeSolver& make_solver, const Start&... start) : solver_{*::new(static_cast<void*>(storage.data())) Solver(make_solver(start...))}
    {
    }

    placed_solver(const placed_solver&) = delete;
    placed_solver& operator=(const placed_solver&) = delete;
    ~placed_solver() { std::destroy_at(&solver_); }

    Solver& get() { return solver_; }

private:
    Solver& solver_;
};

template <typename Solver, typename MakeSolver, typename... Start>
first_step take_first_step(std::array<std::byte, sizeof(Solver)>& storage, const MakeSolver& make_solver, const Start&... start)
{
    placed_solver<Solver> placed{storage, make_solver, start...};
    Solver& solver = placed.get();
    const solver_status status = solver.status();
    const Eigen::VectorXd before = solver.state().x;
    const std::optional<solver_status> policy_status = solver.step().policy_status;
    return {status, before, policy_status, solver.state().x};
}

// Eigen's packet kernels peel by object address, so both solvers are built in one storage to compare bits exactly.
template <typename MakeSolver>
void check_same_first_step(const MakeSolver& make_solver, const Eigen::VectorXd& start)
{
    using solver_type = decltype(make_solver());
    alignas(solver_type) std::array<std::byte, sizeof(solver_type)> storage;
    const first_step from_dynamic = take_first_step<solver_type>(storage, make_solver, start);
    const first_step from_own = take_first_step<solver_type>(storage, make_solver);
    CHECK(from_dynamic.status == from_own.status);
    CHECK(from_dynamic.start == from_own.start);
    CHECK_FALSE(from_own.policy_status == solver_status::invalid_problem);
    CHECK(from_dynamic.policy_status == from_own.policy_status);
    CHECK(from_dynamic.x == from_own.x);
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
        check_same_first_step(make_solver, start);
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
