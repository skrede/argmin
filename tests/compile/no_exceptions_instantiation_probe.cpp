// Exceptions-off instantiation probe.
//
// This translation unit is compiled and LINKED with -fno-exceptions
// -fno-rtti at the C++20 floor. Unlike a -fsyntax-only header check --
// which passes even when live throw sites remain in the instantiated
// templates -- this probe forces every real-time-claimed solver policy
// through step_budget_solver's init(), step() and reset() template bodies on
// concrete small problems. A single surviving throw in any instantiated
// path fails the -fno-exceptions compile, so the "exceptions-off-clean"
// guarantee is exercised rather than merely asserted.
//
// Catch2 is deliberately not used: it requires exceptions. This is a
// plain main() returning 0; convergence is not checked, only that the
// full step/init/reset instantiation compiles and links exceptions-off.

#include "argmin/types.h"

#include "argmin/solver/options.h"
#include "argmin/solver/lm_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/cmaes_policy.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/byrd_lbfgsb_policy.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/filter_trsqp_policy.h"
#include "argmin/solver/projected_gn_policy.h"
#include "argmin/solver/filter_nw_sqp_policy.h"
#include "argmin/solver/augmented_lagrangian_policy.h"

#include "argmin/test_functions/rosenbrock.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <cmath>
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
    g_sink += sr.objective_value;
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
    }

    return 0;
}
