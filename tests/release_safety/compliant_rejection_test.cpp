#include "eigen_trap.h"

#include "argmin/qp/qp_types.h"
#include "argmin/qp/dense_admm_qp.h"
#include "argmin/qp/sparse_admm_qp.h"
#include "argmin/qp/dense_active_set_qp.h"

#include "argmin/solver/options.h"
#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/solver/alternative/cmaes/repair_l2_penalty_policy.h"
#include "argmin/solver/alternative/cmaes/pwq_reparameterization_policy.h"
#include "argmin/solver/alternative/cmaes/no_repair_adaptive_penalty_policy.h"

#include "argmin/solver/alternative/isres/nlopt_faithful_policy.h"

#include "argmin/result/status.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <cstdint>
#include <utility>
#include <optional>

using namespace argmin;

namespace
{

struct bounded_sphere
{
    Eigen::VectorXd lb;
    Eigen::VectorXd ub;

    static constexpr std::int32_t problem_dimension = dynamic_dimension;

    std::int32_t dimension() const { return static_cast<std::int32_t>(lb.size()); }

    double value(const Eigen::VectorXd& x) const { return x.squaredNorm(); }

    Eigen::VectorXd lower_bounds() const { return lb; }

    Eigen::VectorXd upper_bounds() const { return ub; }
};

struct boxed_half_plane : bounded_sphere
{
    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = x[0] + x[1] - 1.0;
    }

    std::int32_t num_equality() const { return 0; }

    std::int32_t num_inequality() const { return 1; }
};

constexpr std::int32_t capped_population = 8;

using pwq_capped = alternative::cmaes::pwq_reparameterization_policy<double, dynamic_dimension, capped_population>;
using l2_capped = alternative::cmaes::repair_l2_penalty_policy<double, dynamic_dimension, capped_population>;
using no_repair_capped = alternative::cmaes::no_repair_adaptive_penalty_policy<double, dynamic_dimension, capped_population>;

Eigen::VectorXd start()
{
    return Eigen::VectorXd{{0.5, 0.2}};
}

template <typename Problem>
Problem boxed(Eigen::VectorXd lb, Eigen::VectorXd ub)
{
    Problem problem;
    problem.lb = std::move(lb);
    problem.ub = std::move(ub);
    return problem;
}

template <typename Policy, typename Problem>
std::optional<solver_status> first_step_status(const Policy& policy, const Problem& problem)
{
    step_budget_solver solver{policy, problem, start(), solver_options<>{}};
    return solver.step().policy_status;
}

template <typename Policy, typename Problem>
solver_status solve_status(const Policy& policy, const Problem& problem)
{
    step_budget_solver solver{policy, problem, start(), solver_options<>{}};
    return solver.solve().status;
}

// Every row of A is an equality row l = u = 1, so row and equality counts move together.
template <typename Solver>
qp_error dense_rejection(Solver& solver, std::int32_t n, std::int32_t m)
{
    const Eigen::MatrixXd P = Eigen::MatrixXd::Identity(n, n);
    const Eigen::VectorXd q = Eigen::VectorXd::Zero(n);
    const Eigen::MatrixXd A = Eigen::MatrixXd::Ones(m, n);
    const Eigen::VectorXd b = Eigen::VectorXd::Ones(m);
    const auto result = solver.solve(P, q, A, b, b);
    REQUIRE_FALSE(result.has_value());
    return result.error();
}

}

TEST_CASE("sparse ADMM rejects an empty problem with a dimension mismatch", "[release-safety][qp]")
{
    const Eigen::SparseMatrix<double, Eigen::ColMajor> P(0, 0);
    const Eigen::SparseMatrix<double, Eigen::ColMajor> A(0, 0);
    const Eigen::VectorXd empty(0);
    sparse_admm_qp_solver<double> solver;
    const auto result = solver.solve(P, empty, A, empty, empty);
    REQUIRE_FALSE(result.has_value());
    CHECK(result.error() == qp_error::dimension_mismatch);
}

TEST_CASE("dense active-set QP rejects more equality rows than variables as an invalid problem", "[release-safety][qp]")
{
    dense_active_set_qp_solver<double> solver(1, 2);
    CHECK(dense_rejection(solver, 1, 2) == qp_error::invalid_problem);
}

TEMPLATE_TEST_CASE("dense QP solvers reject a problem past their constructed capacity", "[release-safety][qp]", dense_admm_qp_solver<double>, dense_active_set_qp_solver<double>)
{
    TestType rows_capped(2, 1);
    CHECK(dense_rejection(rows_capped, 2, 2) == qp_error::capacity_exceeded);
    TestType dimension_capped(1, 2);
    CHECK(dense_rejection(dimension_capped, 2, 1) == qp_error::capacity_exceeded);
}

TEMPLATE_TEST_CASE("CMA-ES rejects a population past its storage cap as an invalid problem", "[release-safety][cmaes]", pwq_capped, l2_capped, no_repair_capped)
{
    TestType policy;
    policy.options.lambda = 16u;
    policy.options.initial_sigma = 0.5;
    const auto problem = boxed<bounded_sphere>(Eigen::VectorXd::Constant(2, -5.0), Eigen::VectorXd::Constant(2, 5.0));
    CHECK(first_step_status(policy, problem) == solver_status::invalid_problem);
    CHECK(solve_status(policy, problem) == solver_status::invalid_problem);
}

TEST_CASE("nlopt-faithful ISRES rejects a population past its storage cap as an invalid problem", "[release-safety][isres]")
{
    alternative::isres::nlopt_faithful_policy<> policy;
    policy.options.seed = 42u;
    policy.options.population_size = 4000u;
    const auto problem = boxed<boxed_half_plane>(Eigen::VectorXd::Constant(2, -10.0), Eigen::VectorXd::Constant(2, 10.0));
    CHECK(first_step_status(policy, problem) == solver_status::invalid_problem);
}

TEST_CASE("COBYLA rejects an inverted box as an invalid problem", "[release-safety][cobyla]")
{
    const auto problem = boxed<boxed_half_plane>(Eigen::VectorXd{{1.0, -1.0}}, Eigen::VectorXd{{0.0, 1.0}});
    CHECK(first_step_status(cobyla_policy<double>{}, problem) == solver_status::invalid_problem);
    CHECK(solve_status(cobyla_policy<double>{}, problem) == solver_status::invalid_problem);
}
