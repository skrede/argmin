// Exceptions-off instantiation probe.
//
// This translation unit is compiled and LINKED with -fno-exceptions
// -fno-rtti at the C++20 floor. Unlike a -fsyntax-only header check --
// which passes even when live throw sites remain in the instantiated
// templates -- this probe forces every real-time-claimed solver policy
// through the driver init(), step() and reset() template bodies on
// concrete small problems. A single surviving throw in any instantiated
// path fails the -fno-exceptions compile, so the "exceptions-off-clean"
// guarantee is exercised rather than merely asserted.
//
// Coverage is the whole real-time-claimed surface, and it is deliberately
// two-dimensional:
//
//   * every solver policy, driven through step_budget_solver;
//   * every driver -- step_budget_solver, stepper, time_budget_solver and
//     step_and_time_budget_solver.
//
// The drivers are separate template bodies, not thin aliases: each owns its
// own stopping logic (stepper has no internal loop at all; the two time
// drivers add a deadline-polling run() loop). Instantiating a policy through
// step_budget_solver therefore says nothing about the other three, which is
// why they are exercised here rather than argued from the step-budget case.
//
// Catch2 is deliberately not used: it requires exceptions. This is a
// plain main() returning 0; convergence is not checked, only that the
// full step/init/reset instantiation compiles and links exceptions-off.

#include "argmin/types.h"

#include "argmin/solver/options.h"
#include "argmin/solver/stepper.h"
#include "argmin/solver/lm_policy.h"
#include "argmin/solver/mma_policy.h"
#include "argmin/solver/cmaes_policy.h"
#include "argmin/solver/gcmma_policy.h"
#include "argmin/solver/isres_policy.h"
#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/byrd_lbfgsb_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/time_budget_solver.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/filter_trsqp_policy.h"
#include "argmin/solver/projected_gn_policy.h"
#include "argmin/solver/time_budget_options.h"
#include "argmin/solver/filter_nw_sqp_policy.h"
#include "argmin/solver/ccsa_quadratic_policy.h"
#include "argmin/solver/augmented_lagrangian_policy.h"
#include "argmin/solver/step_and_time_budget_solver.h"
#include "argmin/solver/projected_gradient_gn_policy.h"

#include "argmin/test_functions/rosenbrock.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <cmath>
#include <chrono>
#include <utility>

namespace
{

// Accumulator that keeps the instantiated step() results observable so the
// calls are not eliminated and no [-Wunused] fires under -Werror.
volatile double g_sink = 0.0;

// Drives one solver through construction (init), a single step() and a
// reset() so all three template surfaces instantiate exceptions-off.
template <typename Policy, typename Problem, typename Vec>
void exercise(Policy policy, const Problem& problem, const Vec& x0)
{
    argmin::solver_options opts;
    opts.max_iterations = 2;

    argmin::step_budget_solver solver{std::move(policy), problem, x0, opts};
    const auto sr = solver.step();
    // Simple assignment (not compound) so the volatile sink stays observable
    // without tripping C++20's deprecated-volatile rule (P1152) on older
    // compilers; the read and write are both visible side effects.
    g_sink = g_sink + sr.objective_value;
    solver.reset(x0);
}

// Drives one solver through the stepper driver. stepper exposes no internal
// loop, so step() and reset() are its entire advancement surface.
template <typename Policy, typename Problem, typename Vec>
void exercise_stepper(Policy policy, const Problem& problem, const Vec& x0)
{
    argmin::solver_options opts;
    opts.max_iterations = 2;

    argmin::stepper st{std::move(policy), problem, x0, opts};
    const auto sr = st.step();
    g_sink = g_sink + sr.objective_value;
    st.reset(x0);
}

// Builds the deadline-bearing options both time drivers take. The budget is
// generous: this probe is a compile-and-link gate, and nothing here asserts
// which stopping condition fired.
inline argmin::time_budget_options<> time_opts()
{
    argmin::time_budget_options<> opts;
    opts.core.max_iterations = 2;
    opts.max_time = std::chrono::milliseconds{50};
    return opts;
}

// Drives one solver through time_budget_solver. solve() is instantiated as
// well as step(): the deadline-polling run() loop is this driver's own
// template body and is not reached through step() alone.
template <typename Policy, typename Problem, typename Vec>
void exercise_time_budget(Policy policy, const Problem& problem, const Vec& x0)
{
    argmin::time_budget_solver solver{std::move(policy), problem, x0, time_opts()};
    const auto sr = solver.step();
    g_sink = g_sink + sr.objective_value;
    const auto r = solver.solve();
    g_sink = g_sink + r.objective_value;
    solver.reset(x0);
}

// Drives one solver through step_and_time_budget_solver, whose run() loop
// honors an iteration cap and a deadline together.
template <typename Policy, typename Problem, typename Vec>
void exercise_step_and_time_budget(Policy policy, const Problem& problem, const Vec& x0)
{
    argmin::step_and_time_budget_solver solver{std::move(policy), problem, x0,
                                               time_opts()};
    const auto sr = solver.step();
    g_sink = g_sink + sr.objective_value;
    const auto r = solver.solve();
    g_sink = g_sink + r.objective_value;
    solver.reset(x0);
}

// ---------------------------------------------------------------------------
// Fixed-N fixtures (small compile-time-dimensioned problems).
// ---------------------------------------------------------------------------

// Fixed-2 Rosenbrock (differentiable, unconstrained) for the L-BFGS-B family.
struct fixed_rosenbrock
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        const double a = 1.0 - x(0);
        const double b = x(1) - x(0) * x(0);
        return a * a + 100.0 * b * b;
    }

    void gradient(const Eigen::Vector<double, 2>& x, Eigen::Vector<double, 2>& g) const
    {
        g(0) = -2.0 * (1.0 - x(0)) - 400.0 * x(0) * (x(1) - x(0) * x(0));
        g(1) = 200.0 * (x(1) - x(0) * x(0));
    }
};

// Fixed-2 Rosenbrock in least-squares residual form for lm_policy.
struct fixed_rosenbrock_ls
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }
    int num_residuals() const { return 2; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        const double r0 = 1.0 - x(0);
        const double r1 = std::sqrt(5.0) * (x(1) - x(0) * x(0));
        return 0.5 * (r0 * r0 + r1 * r1);
    }

    void residuals(const Eigen::Vector<double, 2>& x, Eigen::VectorXd& r) const
    {
        r(0) = 1.0 - x(0);
        r(1) = std::sqrt(5.0) * (x(1) - x(0) * x(0));
    }

    void jacobian(const Eigen::Vector<double, 2>& x, Eigen::MatrixXd& J) const
    {
        J(0, 0) = -1.0;
        J(0, 1) = 0.0;
        J(1, 0) = -2.0 * std::sqrt(5.0) * x(0);
        J(1, 1) = std::sqrt(5.0);
    }
};

// Bounded fixed-2 least-squares Rosenbrock for projected_gn_policy.
struct bounded_fixed_rosenbrock_ls
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }
    int num_residuals() const { return 2; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        const double r0 = 1.0 - x(0);
        const double r1 = std::sqrt(5.0) * (x(1) - x(0) * x(0));
        return 0.5 * (r0 * r0 + r1 * r1);
    }

    void residuals(const Eigen::Vector<double, 2>& x, Eigen::VectorXd& r) const
    {
        r(0) = 1.0 - x(0);
        r(1) = std::sqrt(5.0) * (x(1) - x(0) * x(0));
    }

    void jacobian(const Eigen::Vector<double, 2>& x, Eigen::MatrixXd& J) const
    {
        J(0, 0) = -1.0;
        J(0, 1) = 0.0;
        J(1, 0) = -2.0 * std::sqrt(5.0) * x(0);
        J(1, 1) = std::sqrt(5.0);
    }

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd{{-2.0, -2.0}}; }
    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd{{2.0, 2.0}}; }
};

// ---------------------------------------------------------------------------
// Derivative-free fixtures. These expose no gradient(), which is what selects
// the derivative-free instantiation paths in bobyqa / cobyla / isres.
// ---------------------------------------------------------------------------

// Bound-constrained, derivative-free Rosenbrock for bobyqa_policy.
struct derivative_free_box_rosenbrock
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        const double a = 1.0 - x[0];
        const double b = x[1] - x[0] * x[0];
        return a * a + 5.0 * b * b;
    }

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd::Constant(2, -5.0); }
    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd::Constant(2, 5.0); }
};

// Bound + single-inequality constrained, derivative-free problem for
// cobyla_policy and isres_policy. Constraint values only (no Jacobian).
struct derivative_free_constrained
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        return x[0] * x[0] + x[1] * x[1];
    }

    void constraints(const Eigen::Vector<double, 2>& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = x[0] + x[1] - 1.0;
    }

    int num_equality() const { return 0; }
    int num_inequality() const { return 1; }

    Eigen::Vector<double, 2> lower_bounds() const
    {
        return Eigen::Vector<double, 2>{-10.0, -10.0};
    }

    Eigen::Vector<double, 2> upper_bounds() const
    {
        return Eigen::Vector<double, 2>{10.0, 10.0};
    }
};

// ---------------------------------------------------------------------------
// Dynamic-N fixture (runtime-dimensioned constrained problem).
// ---------------------------------------------------------------------------

// Equality-and-bound constrained quadratic with a runtime dimension, used to
// instantiate an SQP policy on the dynamic_dimension axis.
struct dynamic_constrained_qp
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int n{2};

    int dimension() const { return n; }

    double value(const Eigen::VectorXd& x) const { return 0.5 * x.squaredNorm(); }
    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const { g = x; }

    int num_equality() const { return 1; }
    int num_inequality() const { return 0; }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c(0) = x.sum() - 1.0;
    }

    void constraint_jacobian(const Eigen::VectorXd&, Eigen::MatrixXd& J) const
    {
        J.setOnes();
    }

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd::Constant(n, -10.0); }
    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd::Constant(n, 10.0); }
};

}

int main()
{
    using namespace argmin;

    // --- Unconstrained differentiable: L-BFGS-B family + CMA-ES ------------

    // Dynamic-N Rosenbrock.
    {
        rosenbrock<> problem{.n = 2};
        Eigen::VectorXd x0{{-1.2, 1.0}};
        exercise(lbfgsb_policy<>{}, problem, x0);
        exercise(byrd_lbfgsb_policy<>{}, problem, x0);
        exercise(cmaes_policy<>{}, problem, x0);
    }

    // Fixed-N Rosenbrock (compile-time dimension axis).
    {
        fixed_rosenbrock problem;
        Eigen::Vector<double, 2> x0(-1.2, 1.0);
        exercise(lbfgsb_policy<fixed_rosenbrock::problem_dimension>{}, problem, x0);
        exercise(byrd_lbfgsb_policy<fixed_rosenbrock::problem_dimension>{}, problem, x0);
    }

    // --- Constrained NLP: the six SQP policies on fixed-N HS071 ------------
    {
        hs071<> problem;
        auto x0 = problem.initial_point();
        exercise(kraft_slsqp_policy<hs071<>::problem_dimension>{}, problem, x0);
        exercise(filter_slsqp_policy<hs071<>::problem_dimension>{}, problem, x0);
        exercise(nw_sqp_policy<hs071<>::problem_dimension>{}, problem, x0);
        exercise(filter_nw_sqp_policy<hs071<>::problem_dimension>{}, problem, x0);
        exercise(tr_sqp_policy<hs071<>::problem_dimension>{}, problem, x0);
        exercise(filter_trsqp_policy<hs071<>::problem_dimension>{}, problem, x0);
    }

    // --- Constrained NLP on the dynamic-N axis -----------------------------
    {
        dynamic_constrained_qp problem;
        Eigen::VectorXd x0{{0.5, 0.5}};
        exercise(kraft_slsqp_policy<>{}, problem, x0);
    }

    // --- Augmented Lagrangian composition over an inner L-BFGS-B ----------
    {
        hs035<> problem;
        auto x0 = problem.initial_point();
        exercise(augmented_lagrangian_policy<lbfgsb_policy<hs035<>::problem_dimension>>{},
                 problem, x0);
    }

    // --- Least squares: Levenberg-Marquardt + projected Gauss-Newton ------
    {
        fixed_rosenbrock_ls problem;
        Eigen::Vector<double, 2> x0(-1.0, 1.0);
        exercise(lm_policy{}, problem, x0);
    }
    {
        bounded_fixed_rosenbrock_ls problem;
        Eigen::Vector<double, 2> x0(-1.0, 1.0);
        exercise(projected_gn_policy<>{}, problem, x0);
        exercise(projected_gradient_gn_policy<>{}, problem, x0);
    }

    // --- Derivative-free: BOBYQA, COBYLA, ISRES ---------------------------
    {
        derivative_free_box_rosenbrock problem;
        Eigen::VectorXd x0{{-1.0, -1.0}};
        exercise(bobyqa_policy<>{}, problem, x0);
    }
    {
        derivative_free_constrained problem;
        Eigen::VectorXd x0{{2.0, 2.0}};
        // cobyla_policy is dimension-agnostic: it is a non-template whose
        // rebind is the identity, so it takes no dimension argument.
        exercise(cobyla_policy{}, problem, x0);
        exercise(isres_policy<derivative_free_constrained::problem_dimension>{},
                 problem, x0);
    }

    // --- CCSA family: MMA, GCMMA, CCSA-quadratic --------------------------
    {
        hs076<> problem;
        auto x0 = problem.initial_point();
        exercise(mma_policy<hs076<>::problem_dimension>{}, problem, x0);
        exercise(ccsa_quadratic_policy<hs076<>::problem_dimension>{}, problem, x0);
    }
    {
        hs024<> problem;
        auto x0 = problem.initial_point();
        exercise(gcmma_policy<hs024<>::problem_dimension>{}, problem, x0);
    }

    // --- The other three drivers -------------------------------------------
    //
    // step_budget_solver is instantiated by every exercise() call above. The
    // remaining drivers carry their own stopping logic, so each is driven
    // here in its own right. Both a constrained SQP policy and an
    // unconstrained one are pushed through each, so neither the constrained
    // nor the bound-projection path in a driver is left uninstantiated.
    {
        hs071<> problem;
        auto x0 = problem.initial_point();
        exercise_stepper(kraft_slsqp_policy<hs071<>::problem_dimension>{}, problem, x0);
        exercise_time_budget(kraft_slsqp_policy<hs071<>::problem_dimension>{},
                             problem, x0);
        exercise_step_and_time_budget(
            kraft_slsqp_policy<hs071<>::problem_dimension>{}, problem, x0);
    }
    {
        rosenbrock<> problem{.n = 2};
        Eigen::VectorXd x0{{-1.2, 1.0}};
        exercise_stepper(lbfgsb_policy<>{}, problem, x0);
        exercise_time_budget(lbfgsb_policy<>{}, problem, x0);
        exercise_step_and_time_budget(lbfgsb_policy<>{}, problem, x0);
    }

    return 0;
}
