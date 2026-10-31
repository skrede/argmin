#ifndef HPP_GUARD_ARGMIN_TESTS_RELEASE_SAFETY_ISRES_PROBE_H
#define HPP_GUARD_ARGMIN_TESTS_RELEASE_SAFETY_ISRES_PROBE_H

#include "eigen_trap.h"

#include "argmin/solver/options.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/result/status.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <optional>

namespace argmin::release_safety
{

// f = |x|^2 on the [-5, 5]^n box with `rows` copies of the inactive 10 - sum(x) >= 0. A constraint call handed a
// buffer of any length other than `rows` is counted and writes nothing.
template <std::int32_t N>
struct counted_box
{
    using vector = Eigen::Vector<double, N>;

    static constexpr std::int32_t problem_dimension = N;

    std::int32_t n;
    std::int32_t rows;
    mutable std::int32_t evaluations;
    mutable std::int32_t wrong_buffers;

    std::int32_t dimension() const { return n; }

    std::int32_t num_equality() const { return 0; }

    std::int32_t num_inequality() const { return rows; }

    double value(const vector& x) const
    {
        ++evaluations;
        return x.squaredNorm();
    }

    template <typename Values>
    void constraints(const vector& x, Values& c) const
    {
        ++evaluations;
        if(c.rows() != rows)
        {
            ++wrong_buffers;
            return;
        }
        c.setConstant(10.0 - x.sum());
    }

    vector lower_bounds() const { return vector::Constant(n, -5.0); }

    vector upper_bounds() const { return vector::Constant(n, 5.0); }
};

inline const Eigen::Vector2d inside(0.5, 0.5);

inline counted_box<2> plane(std::int32_t rows)
{
    return {.n = 2, .rows = rows, .evaluations = 0, .wrong_buffers = 0};
}

template <typename Policy>
Policy sized(std::optional<std::uint32_t> population, std::optional<double> parent_fraction)
{
    Policy policy;
    policy.options.seed = 7u;
    policy.options.population_size = population;
    policy.options.parent_fraction = parent_fraction;
    return policy;
}

template <typename Policy, typename Problem>
void check_rejected(const Policy& policy, const Problem& problem, const typename Problem::vector& x0)
{
    step_budget_solver solver{policy, problem, x0, solver_options<>{}};
    CHECK(solver.status() == solver_status::invalid_problem);
    CHECK(solver.step().policy_status == solver_status::invalid_problem);
    CHECK(solver.reset(x0) == solver_status::invalid_problem);
    CHECK(solver.reset_clear(x0) == solver_status::invalid_problem);
    step_budget_solver fresh{policy, problem, x0, solver_options<>{}};
    CHECK(fresh.solve(solver_options<>{}).status == solver_status::invalid_problem);
    CHECK(problem.evaluations == 0);
}

template <typename Policy, typename Problem>
void check_accepted(const Policy& policy, const Problem& problem, const typename Problem::vector& x0)
{
    step_budget_solver solver{policy, problem, x0, solver_options<>{}};
    CHECK(solver.status() == solver_status::running);
    for(std::int32_t k = 0; k < 3; ++k)
        CHECK(solver.step().policy_status != solver_status::invalid_problem);
    CHECK(solver.reset(x0) == solver_status::running);
    CHECK(problem.wrong_buffers == 0);
}

}

#endif
