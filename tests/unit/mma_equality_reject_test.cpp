// MMA / GCMMA / CCSA equality-constraint rejection (release-safety pin).
//
// The separable-approximation policies (mma, gcmma, ccsa_quadratic) are
// inequality-and-box methods: they size their constraint buffers for the
// inequality rows only. A problem's constraints()/constraint_jacobian()
// write num_equality() + num_inequality() rows with the equality rows
// FIRST (the same ordering the augmented Lagrangian reads as
// head(n_eq)/tail(n_ineq)). Handing such a policy an equality-constrained
// problem therefore used to overflow the inequality-sized buffers -- a
// heap out-of-bounds write in a release/NDEBUG build, where the former
// debug-only precondition assert was compiled out.
//
// This case pins the exception-free runtime reject that closes that
// defect: an equality-constrained problem now yields a terminal
// solver_status::invalid_problem instead of undefined behavior. It is a
// deliberate release-config regression -- it MUST pass under NDEBUG (the
// configuration where the overflow used to manifest) and cleanly under
// AddressSanitizer.
//
// The worked problem carries BOTH an equality and an inequality row and
// fills them equality-first without resizing the caller's buffer, exactly
// the shape that overflowed an inequality-sized buffer before the fix.

#include "argmin/solver/mma_policy.h"
#include "argmin/solver/gcmma_policy.h"
#include "argmin/solver/ccsa_quadratic_policy.h"
#include "argmin/solver/basic_solver.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

using namespace argmin;

namespace
{

// 2D problem with one equality and one inequality constraint. The
// constraint writers fill the equality row first and do NOT resize the
// destination -- so an inequality-sized (length-1) buffer would take an
// out-of-bounds write at the inequality index. Post-fix, the policies
// probe with a full-length buffer during init and reject at the first
// step() before any constraint evaluation.
struct eq_and_ineq_problem
{
    static constexpr int problem_dimension = dynamic_dimension;

    int dimension() const { return 2; }
    int num_equality() const { return 1; }
    int num_inequality() const { return 1; }

    double value(const Eigen::VectorXd& x) const
    {
        return x[0] * x[0] + x[1] * x[1];
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        g.resize(2);
        g[0] = 2.0 * x[0];
        g[1] = 2.0 * x[1];
    }

    // Equality row first, inequality row second; caller must pre-size.
    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c[0] = x[0] + x[1] - 1.0;   // equality: x0 + x1 = 1
        c[1] = 2.0 - x[0] - x[1];   // inequality (argmin: >= 0 feasible)
    }

    void constraint_jacobian(const Eigen::VectorXd&, Eigen::MatrixXd& J) const
    {
        J(0, 0) = 1.0;  J(0, 1) = 1.0;    // equality row
        J(1, 0) = -1.0; J(1, 1) = -1.0;   // inequality row
    }

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd{{0.0, 0.0}}; }
    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd{{2.0, 2.0}}; }
};

template <typename Policy>
solver_status first_step_status(const eq_and_ineq_problem& problem,
                                const Eigen::VectorXd& x0)
{
    Policy policy;
    solver_options<> opts;
    auto s = policy.init(problem, x0, opts);
    const auto r = policy.step(s);
    REQUIRE(r.policy_status.has_value());
    return *r.policy_status;
}

template <typename Policy>
solver_status solved_status(const eq_and_ineq_problem& problem,
                            const Eigen::VectorXd& x0)
{
    solver_options<> opts;
    opts.max_iterations = 10;
    basic_solver solver{Policy{}, problem, x0, opts};
    return solver.solve(opts).status;
}

}

TEST_CASE("mma rejects an equality-constrained problem at runtime",
          "[mma][reject][release-safety]")
{
    eq_and_ineq_problem problem;
    Eigen::VectorXd x0{{0.5, 0.5}};

    // Direct policy contract: the first step() reports invalid_problem.
    CHECK(first_step_status<mma_policy<>>(problem, x0)
          == solver_status::invalid_problem);
    // Full-solver plumbing: the terminal status propagates.
    CHECK(solved_status<mma_policy<>>(problem, x0)
          == solver_status::invalid_problem);
}

TEST_CASE("gcmma rejects an equality-constrained problem at runtime",
          "[gcmma][reject][release-safety]")
{
    eq_and_ineq_problem problem;
    Eigen::VectorXd x0{{0.5, 0.5}};

    CHECK(first_step_status<gcmma_policy<>>(problem, x0)
          == solver_status::invalid_problem);
    CHECK(solved_status<gcmma_policy<>>(problem, x0)
          == solver_status::invalid_problem);
}

TEST_CASE("ccsa_quadratic rejects an equality-constrained problem at runtime",
          "[ccsa][reject][release-safety]")
{
    eq_and_ineq_problem problem;
    Eigen::VectorXd x0{{0.5, 0.5}};

    CHECK(first_step_status<ccsa_quadratic_policy<>>(problem, x0)
          == solver_status::invalid_problem);
    CHECK(solved_status<ccsa_quadratic_policy<>>(problem, x0)
          == solver_status::invalid_problem);
}
