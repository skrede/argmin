#include "mock_policy.h"
#include "argmin/solver/stepper.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/convergence.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/result/status.h"
#include "argmin/formulation/concepts.h"

#include <cstdint>
#include <type_traits>

#include <catch2/catch_test_macros.hpp>

using namespace argmin;

// Convex quadratic with an analytic gradient: lbfgsb descends it
// monotonically, so a solve()'s best-seen iterate coincides bit-for-bit with
// the terminal iterate -- exactly the property the driver/stepper identity
// section leans on.
struct quadratic_with_gradient
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }
    double value(const Eigen::Vector<double, 2>& x) const { return 0.5 * x.squaredNorm(); }
    void gradient(const Eigen::Vector<double, 2>& x, Eigen::Vector<double, 2>& g) const { g = x; }
};

// Dummy problem for the mock-policy sections (mock_policy ignores it).
struct quadratic
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }
    double value(const Eigen::Vector<double, 2>& x) const { return 0.5 * x.squaredNorm(); }
};

// ---------------------------------------------------------------------------
// (1) IDENTITY: a hand-rolled caller loop over stepper reproduces
//     step_budget_solver::solve() iterate/objective trajectories bit-for-bit.
// ---------------------------------------------------------------------------

TEST_CASE("stepper caller-loop reproduces step_budget_solver::solve() bit-identically",
          "[stepper][identity]")
{
    quadratic_with_gradient prob;
    Eigen::VectorXd x0{{3.0, 4.0}};
    solver_options<> opts;
    opts.max_iterations = 1000;

    // Driver: single loop-owning solve().
    step_budget_solver driver{lbfgsb_policy<>{}, prob, x0, opts};
    auto rd = driver.solve();
    REQUIRE(rd.status == solver_status::converged);
    REQUIRE(rd.iterations > 0);

    // Stepper: hand-rolled loop over step(), stopping on status().
    stepper st{lbfgsb_policy<>{}, prob, x0, opts};
    std::uint32_t k = 0;
    while(st.status() == solver_status::running && k < opts.max_iterations)
    {
        st.step();
        ++k;
    }

    // Same terminal verdict, same iteration count.
    CHECK(st.status() == rd.status);
    CHECK(k == rd.iterations);
    CHECK(st.converged());

    // Bit-identical terminal iterate (monotone descent => best-seen == terminal).
    REQUIRE(st.state().x.size() == rd.x.size());
    CHECK((st.state().x.array() == rd.x.array()).all());

    // Bit-identical reported gradient norm and objective.
    CHECK(st.gradient_norm() == rd.gradient_norm);
    CHECK(prob.value(st.state().x) == rd.objective_value);
}

// ---------------------------------------------------------------------------
// (2) CONCEPT BOUNDARY: stepper satisfies steppable but NOT nlp_solver; the
//     loop-owning driver satisfies both.
// ---------------------------------------------------------------------------

static_assert(steppable<stepper<test::mock_policy>>,
              "stepper exposes the passive single-step surface");
static_assert(!nlp_solver<stepper<test::mock_policy>>,
              "stepper has no solve()/step_n() -- it must NOT satisfy nlp_solver");

static_assert(steppable<step_budget_solver<test::mock_policy>>,
              "the driver refines steppable");
static_assert(nlp_solver<step_budget_solver<test::mock_policy>>,
              "the driver owns a convergence loop -- it satisfies nlp_solver");

// Also holds for a real gradient policy instantiation.
static_assert(steppable<stepper<lbfgsb_policy<2>, 2, quadratic_with_gradient>>);
static_assert(!nlp_solver<stepper<lbfgsb_policy<2>, 2, quadratic_with_gradient>>);

TEST_CASE("stepper satisfies steppable but not nlp_solver", "[stepper][concept]")
{
    // The compile-time asserts above are the real gate; this runtime case
    // exists so the boundary is visible in the ctest listing too.
    STATIC_REQUIRE(steppable<stepper<test::mock_policy>>);
    STATIC_REQUIRE(!nlp_solver<stepper<test::mock_policy>>);
    STATIC_REQUIRE(nlp_solver<step_budget_solver<test::mock_policy>>);
}

// ---------------------------------------------------------------------------
// (3) RESET: a mid-run reset restarts identically to a fresh construction
//     (mock_policy is stateless, so reset() == fresh init).
// ---------------------------------------------------------------------------

TEST_CASE("stepper reset mid-run restarts identically to a fresh construction",
          "[stepper][reset]")
{
    quadratic prob;
    Eigen::VectorXd x0{{3.0, 4.0}};
    solver_options<> opts;
    opts.max_iterations = 1000;
    std::get<gradient_tolerance_criterion>(opts.convergence.criteria).threshold = 1e-4;

    // A: step a few times, then reset back to x0 and drive to termination.
    stepper<test::mock_policy> a{prob, x0, opts};
    a.step();
    a.step();
    a.step();
    a.reset(x0);

    std::uint32_t ka = 0;
    while(a.status() == solver_status::running && ka < opts.max_iterations)
    {
        a.step();
        ++ka;
    }

    // B: fresh construction from x0, driven to termination.
    stepper<test::mock_policy> b{prob, x0, opts};
    std::uint32_t kb = 0;
    while(b.status() == solver_status::running && kb < opts.max_iterations)
    {
        b.step();
        ++kb;
    }

    CHECK(a.status() == b.status());
    CHECK(ka == kb);
    REQUIRE(a.state().x.size() == b.state().x.size());
    CHECK((a.state().x.array() == b.state().x.array()).all());
    CHECK(a.gradient_norm() == b.gradient_norm());
}
