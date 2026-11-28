#include "argmin/detail/active_set_qp.h"

#include "argmin/qp/dense_active_set_qp.h"

#include "argmin/options/qp_options.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using argmin::detail::qp_status;
using solver_type = argmin::detail::active_set_qp_solver<double>;

namespace
{

// min (x0 - 1)^2 + (x1 - 2.5)^2 over the five rows of the textbook active-set example, from the feasible vertex (2, 0);
// solved by (1.4, 1.7) after several working-set changes.
solver_type::result_type solve_textbook_example(std::uint32_t max_iterations)
{
    Eigen::MatrixXd A_in(5, 2);
    A_in << 1.0, -2.0, -1.0, -2.0, -1.0, 2.0, 1.0, 0.0, 0.0, 1.0;
    solver_type solver(2, 5);
    argmin::qp_options<> opts;
    opts.max_iterations = max_iterations;
    return solver.solve(2.0 * Eigen::MatrixXd::Identity(2, 2), Eigen::VectorXd{{-2.0, -5.0}}, Eigen::MatrixXd(0, 2), Eigen::VectorXd(0),
                        A_in, Eigen::VectorXd{{-2.0, -6.0, -2.0, 0.0, 0.0}}, Eigen::VectorXd{{2.0, 0.0}}, opts);
}

// A start of magnitude 1e9 on the row 0.3 x0 + 0.7 x1 + 1.1 x2 = 0: the step to the minimizer near zero cancels, so the
// landed point misses the row by rounding of order 1e9 epsilon, far above the sqrt(epsilon) floor relative to |x| ~ 0.
Eigen::VectorXd start_on_row()
{
    const Eigen::Vector3d a(0.3, 0.7, 1.1);
    const Eigen::Vector3d direction(1.37, -0.91, 0.53);
    return 1e9 * (direction - a * (a.dot(direction) / a.squaredNorm()));
}

}

TEST_CASE("active_set_qp_solver reports max_iterations only when it reaches its iteration cap", "[qp]")
{
    const auto capped = solve_textbook_example(1);
    CHECK(capped.status == qp_status::max_iterations);
    CHECK(capped.iterations == 1);
    CHECK(capped.primal_feasible);

    const auto solved = solve_textbook_example(200);
    REQUIRE(solved.status == qp_status::optimal);
    CHECK((solved.x - Eigen::Vector2d(1.4, 1.7)).cwiseAbs().maxCoeff() <= 1e-12);
    CHECK(solved.primal_feasible);
}

TEST_CASE("active_set_qp_solver reports unverified for an early stop its terminal check rejects", "[qp]")
{
    solver_type solver(3, 1);
    const auto r = solver.solve(Eigen::MatrixXd::Identity(3, 3).eval(), Eigen::VectorXd::Zero(3).eval(), Eigen::MatrixXd{{0.3, 0.7, 1.1}},
                                Eigen::VectorXd{{0.0}}, Eigen::MatrixXd(0, 3), Eigen::VectorXd(0), start_on_row());
    CHECK(r.status == qp_status::unverified);
    CHECK(r.iterations < 200);
    CHECK_FALSE(r.primal_feasible);
}

TEST_CASE("active_set_qp_solver reports a start it cannot restore as not primal feasible", "[qp]")
{
    solver_type solver(1, 2);
    const auto r = solver.solve(Eigen::MatrixXd::Identity(1, 1).eval(), Eigen::VectorXd::Zero(1).eval(), Eigen::MatrixXd(0, 1),
                                Eigen::VectorXd(0), Eigen::MatrixXd{{1.0}, {-1.0}}, Eigen::VectorXd{{1.0, 0.0}}, Eigen::VectorXd{{0.5}});
    CHECK(r.status == qp_status::infeasible);
    CHECK_FALSE(r.primal_feasible);
}

TEST_CASE("dense_active_set_qp_solver reports solved_inaccurate when its inner solve stops unverified", "[qp]")
{
    argmin::dense_active_set_qp_solver<double> solver(3, 1);
    REQUIRE_FALSE(solver.warm_start(start_on_row(), Eigen::VectorXd(0)).has_value());
    argmin::dense_active_set_qp_options<double> opts;
    opts.tolerance = 1e-12;
    const Eigen::VectorXd zero{{0.0}};
    const auto r = solver.solve(Eigen::MatrixXd::Identity(3, 3), Eigen::VectorXd::Zero(3), Eigen::MatrixXd{{0.3, 0.7, 1.1}}, zero, zero, opts);
    REQUIRE(r.has_value());
    CHECK(r->status == argmin::qp_solve_status::solved_inaccurate);
    CHECK(r->iterations < static_cast<int>(opts.max_iterations));
}
