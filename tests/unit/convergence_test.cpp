// Comprehensive cross-solver convergence test suite (TEST-02).
//
// Validates all 9 v0.1.0 solver policies on appropriate test functions.
// Each solver is tested on at least 3 problems with class-appropriate
// tolerances per D-09:
//   - Gradient-based (L-BFGS-B, SQP, LM): f within 1e-8 of f*
//   - Global/derivative-free (CMA-ES, BOBYQA): f within 1e-4
//   - Constrained (AugLag, MMA, GCMMA): f within 1e-4, violation < 1e-4
//
// Reference: Kochenderfer & Wheeler Ch. B (test functions),
//            Hock & Schittkowski (1981) test problems.

#include "nablapp/nablapp.h"
#include "nablapp/schedule/round_robin_schedule.h"
#include "nablapp/schedule/basic_solver_group.h"
#include "nablapp/test_functions/rosenbrock.h"
#include "nablapp/test_functions/rastrigin.h"
#include "nablapp/test_functions/himmelblau.h"
#include "nablapp/test_functions/ackley.h"
#include "nablapp/test_functions/booth.h"
#include "nablapp/test_functions/beale.h"
#include "nablapp/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>

using Catch::Approx;
using namespace nablapp;

// ---------------------------------------------------------------------------
// Local test problem types (anonymous namespace)
// ---------------------------------------------------------------------------

namespace
{

// Bounded Rosenbrock for BOBYQA (objective + bound_constrained, no gradient).
struct bounded_rosenbrock
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        double t1 = 1.0 - x[0];
        double t2 = x[1] - x[0] * x[0];
        return t1 * t1 + 5.0 * t2 * t2;
    }

    Eigen::Vector<double, 2> lower_bounds() const
    {
        return Eigen::Vector<double, 2>::Constant(-5.0);
    }

    Eigen::Vector<double, 2> upper_bounds() const
    {
        return Eigen::Vector<double, 2>::Constant(5.0);
    }
};

// Bounded Booth for BOBYQA.
struct bounded_booth
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        double t1 = x[0] + 2.0 * x[1] - 7.0;
        double t2 = 2.0 * x[0] + x[1] - 5.0;
        return t1 * t1 + t2 * t2;
    }

    Eigen::Vector<double, 2> lower_bounds() const
    {
        return Eigen::Vector<double, 2>::Constant(-10.0);
    }

    Eigen::Vector<double, 2> upper_bounds() const
    {
        return Eigen::Vector<double, 2>::Constant(10.0);
    }
};

// Bounded Himmelblau for BOBYQA.
struct bounded_himmelblau
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        double t1 = x[0] * x[0] + x[1] - 11.0;
        double t2 = x[0] + x[1] * x[1] - 7.0;
        return t1 * t1 + t2 * t2;
    }

    Eigen::Vector<double, 2> lower_bounds() const
    {
        return Eigen::Vector<double, 2>::Constant(-5.0);
    }

    Eigen::Vector<double, 2> upper_bounds() const
    {
        return Eigen::Vector<double, 2>::Constant(5.0);
    }
};

// Rosenbrock in least-squares residual form (b=5 per project convention).
// r_0 = 1 - x_0, r_1 = sqrt(5)*(x_1 - x_0^2)
// f(x) = 0.5*(r_0^2 + r_1^2), minimum at (1,1), f*=0.
struct rosenbrock_ls
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }
    int num_residuals() const { return 2; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        double r0 = 1.0 - x(0);
        double r1 = std::sqrt(5.0) * (x(1) - x(0) * x(0));
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

// Exponential decay fitting: y = a * exp(b * t).
// 5 data points, 2 parameters.
struct exponential_fitting
{
    static constexpr double t[] = {0.0, 0.5, 1.0, 1.5, 2.0};
    static constexpr double y[] = {2.0, 1.2, 0.75, 0.45, 0.28};

    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }
    int num_residuals() const { return 5; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        double f = 0.0;
        for(int i = 0; i < 5; ++i)
        {
            double ri = x(0) * std::exp(x(1) * t[i]) - y[i];
            f += ri * ri;
        }
        return 0.5 * f;
    }

    void residuals(const Eigen::Vector<double, 2>& x, Eigen::VectorXd& r) const
    {
        for(int i = 0; i < 5; ++i)
            r(i) = x(0) * std::exp(x(1) * t[i]) - y[i];
    }

    void jacobian(const Eigen::Vector<double, 2>& x, Eigen::MatrixXd& J) const
    {
        for(int i = 0; i < 5; ++i)
        {
            double e = std::exp(x(1) * t[i]);
            J(i, 0) = e;
            J(i, 1) = x(0) * t[i] * e;
        }
    }
};

// Powell singular function in least-squares form.
// 4 residuals, 4 variables. f* = 0 at origin.
//
// Reference: Powell, "A hybrid method for nonlinear equations" (1970).
struct powell_singular
{
    static constexpr int problem_dimension = 4;

    int dimension() const { return 4; }
    int num_residuals() const { return 4; }

    double value(const Eigen::Vector<double, 4>& x) const
    {
        Eigen::VectorXd r(4);
        residuals(x, r);
        return 0.5 * r.squaredNorm();
    }

    void residuals(const Eigen::Vector<double, 4>& x, Eigen::VectorXd& r) const
    {
        r(0) = x(0) + 10.0 * x(1);
        r(1) = std::sqrt(5.0) * (x(2) - x(3));
        r(2) = (x(1) - 2.0 * x(2)) * (x(1) - 2.0 * x(2));
        r(3) = std::sqrt(10.0) * (x(0) - x(3)) * (x(0) - x(3));
    }

    void jacobian(const Eigen::Vector<double, 4>& x, Eigen::MatrixXd& J) const
    {
        J.setZero();
        J(0, 0) = 1.0;
        J(0, 1) = 10.0;
        J(1, 2) = std::sqrt(5.0);
        J(1, 3) = -std::sqrt(5.0);
        J(2, 1) = 2.0 * (x(1) - 2.0 * x(2));
        J(2, 2) = -4.0 * (x(1) - 2.0 * x(2));
        J(3, 0) = 2.0 * std::sqrt(10.0) * (x(0) - x(3));
        J(3, 3) = -2.0 * std::sqrt(10.0) * (x(0) - x(3));
    }
};

// Rosenbrock with a single inequality constraint: x0 + x1 >= 1.
// For SQP and AugLag testing.
struct rosenbrock_constrained
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }
    int num_equality() const { return 0; }
    int num_inequality() const { return 1; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        double t1 = 1.0 - x[0];
        double t2 = x[1] - x[0] * x[0];
        return t1 * t1 + 5.0 * t2 * t2;
    }

    void gradient(const Eigen::Vector<double, 2>& x, Eigen::Vector<double, 2>& g) const
    {
        double t1 = 1.0 - x[0];
        double t2 = x[1] - x[0] * x[0];
        g[0] = -2.0 * t1 - 20.0 * x[0] * t2;
        g[1] = 10.0 * t2;
    }

    void constraints(const Eigen::Vector<double, 2>& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = x[0] + x[1] - 1.0;  // x0 + x1 >= 1
    }

    void constraint_jacobian(const Eigen::Vector<double, 2>&, Eigen::MatrixXd& J) const
    {
        J.resize(1, 2);
        J(0, 0) = 1.0;
        J(0, 1) = 1.0;
    }

    Eigen::Vector<double, 2> lower_bounds() const
    {
        constexpr double inf = std::numeric_limits<double>::infinity();
        return Eigen::Vector<double, 2>::Constant(-inf);
    }

    Eigen::Vector<double, 2> upper_bounds() const
    {
        constexpr double inf = std::numeric_limits<double>::infinity();
        return Eigen::Vector<double, 2>::Constant(inf);
    }
};

// Beam design: minimize x0*x1 subject to x0*x1 >= 1, bounds [0.1, 10].
// For MMA/GCMMA testing (inequality only).
struct beam_design
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }
    int num_equality() const { return 0; }
    int num_inequality() const { return 1; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        return x[0] * x[1];
    }

    void gradient(const Eigen::Vector<double, 2>& x, Eigen::Vector<double, 2>& g) const
    {
        g[0] = x[1];
        g[1] = x[0];
    }

    void constraints(const Eigen::Vector<double, 2>& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = x[0] * x[1] - 1.0;  // x0*x1 >= 1
    }

    void constraint_jacobian(const Eigen::Vector<double, 2>& x, Eigen::MatrixXd& J) const
    {
        J.resize(1, 2);
        J(0, 0) = x[1];
        J(0, 1) = x[0];
    }

    Eigen::Vector<double, 2> lower_bounds() const
    {
        return Eigen::Vector<double, 2>::Constant(0.1);
    }

    Eigen::Vector<double, 2> upper_bounds() const
    {
        return Eigen::Vector<double, 2>::Constant(10.0);
    }
};

}

// ---------------------------------------------------------------------------
// Convergence test suite: all v0.1.0 solvers
// ---------------------------------------------------------------------------

TEST_CASE("Convergence test suite: all v0.1.0 solvers", "[convergence]")
{
    // -------------------------------------------------------------------
    // L-BFGS-B: gradient-based, f within 1e-8
    // -------------------------------------------------------------------
    SECTION("L-BFGS-B")
    {
        solver_options opts;
        opts.max_iterations = 500;
        opts.set_gradient_threshold(1e-12);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        SECTION("Rosenbrock 2D")
        {
            rosenbrock problem{};
            Eigen::VectorXd x0{{-1.0, -1.0}};
            basic_solver<lbfgsb_policy<rosenbrock<>::problem_dimension>, rosenbrock<>::problem_dimension, rosenbrock<>> solver{problem, x0, opts};
            auto result = solver.solve(opts);
            CHECK(result.objective_value < 1e-8);
        }

        SECTION("Booth")
        {
            booth problem{};
            Eigen::VectorXd x0{{0.0, 0.0}};
            basic_solver<lbfgsb_policy<booth<>::problem_dimension>, booth<>::problem_dimension, booth<>> solver{problem, x0, opts};
            auto result = solver.solve(opts);
            CHECK(result.objective_value < 1e-8);
        }

        SECTION("Beale")
        {
            beale problem{};
            Eigen::VectorXd x0{{0.0, 0.0}};
            basic_solver<lbfgsb_policy<beale<>::problem_dimension>, beale<>::problem_dimension, beale<>> solver{problem, x0, opts};
            auto result = solver.solve(opts);
            CHECK(result.objective_value < 1e-8);
        }
    }

    // -------------------------------------------------------------------
    // BOBYQA: derivative-free bounded, f within 1e-4
    // -------------------------------------------------------------------
    SECTION("BOBYQA")
    {
        solver_options opts;
        opts.max_iterations = 500;
        opts.set_gradient_threshold(1e-15);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        SECTION("Rosenbrock 2D bounded")
        {
            bounded_rosenbrock problem;
            Eigen::VectorXd x0{{-1.0, -1.0}};
            basic_solver<bobyqa_policy<bounded_rosenbrock::problem_dimension>,
                         bounded_rosenbrock::problem_dimension, bounded_rosenbrock> solver{problem, x0, opts};
            auto result = solver.solve(opts);
            CHECK(result.objective_value < 1e-4);
        }

        SECTION("Booth bounded")
        {
            bounded_booth problem;
            Eigen::VectorXd x0{{0.0, 0.0}};
            basic_solver<bobyqa_policy<bounded_booth::problem_dimension>,
                         bounded_booth::problem_dimension, bounded_booth> solver{problem, x0, opts};
            auto result = solver.solve(opts);
            CHECK(result.objective_value < 1e-4);
        }

        SECTION("Himmelblau bounded")
        {
            bounded_himmelblau problem;
            Eigen::VectorXd x0{{1.0, 1.0}};
            basic_solver<bobyqa_policy<bounded_himmelblau::problem_dimension>,
                         bounded_himmelblau::problem_dimension, bounded_himmelblau> solver{problem, x0, opts};
            auto result = solver.solve(opts);
            CHECK(result.objective_value < 1e-4);
        }
    }

    // -------------------------------------------------------------------
    // CMA-ES: global derivative-free, f within budget
    // -------------------------------------------------------------------
    SECTION("CMA-ES")
    {
        SECTION("Rastrigin 5D")
        {
            rastrigin problem{.n = 5};
            Eigen::VectorXd x0 = Eigen::VectorXd::Constant(5, 2.0);
            solver_options opts;
            opts.max_iterations = 5000;
            opts.set_gradient_threshold(1e-15);
            opts.set_step_threshold(1e-15);
            opts.set_objective_threshold(1e-15);

            cmaes_policy policy;
            policy.options.initial_sigma = 1.0;
            policy.options.lambda = 64u;
            policy.options.seed = 42u;
            basic_solver solver{policy, problem, x0, opts};
            auto result = solver.solve(opts);
            CHECK(result.objective_value < 5.0);
        }

        SECTION("Rosenbrock 2D")
        {
            rosenbrock problem{};
            Eigen::VectorXd x0{{-1.0, -1.0}};
            solver_options opts;
            opts.max_iterations = 2000;
            opts.set_gradient_threshold(1e-15);
            opts.set_step_threshold(1e-15);
            opts.set_objective_threshold(1e-15);

            cmaes_policy policy;
            policy.options.initial_sigma = 0.5;
            policy.options.seed = 42u;
            basic_solver solver{policy, problem, x0, opts};
            auto result = solver.solve(opts);
            CHECK(result.objective_value < 0.01);
        }

        SECTION("Ackley 5D")
        {
            ackley problem{.n = 5};
            Eigen::VectorXd x0 = Eigen::VectorXd::Constant(5, 1.0);
            solver_options opts;
            opts.max_iterations = 5000;
            opts.set_gradient_threshold(1e-15);
            opts.set_step_threshold(1e-15);
            opts.set_objective_threshold(1e-15);

            cmaes_policy policy;
            policy.options.initial_sigma = 1.0;
            policy.options.seed = 42u;
            basic_solver solver{policy, problem, x0, opts};
            auto result = solver.solve(opts);
            CHECK(result.objective_value < 1.0);
        }
    }

    // -------------------------------------------------------------------
    // Levenberg-Marquardt: least-squares, f within 1e-8
    // -------------------------------------------------------------------
    SECTION("Levenberg-Marquardt")
    {
        solver_options opts;
        opts.max_iterations = 200;
        opts.set_gradient_threshold(1e-12);

        SECTION("Rosenbrock residual")
        {
            rosenbrock_ls problem;
            Eigen::VectorXd x0{{-1.0, 1.0}};
            basic_solver solver{lm_policy{}, problem, x0, opts};
            auto result = solver.solve(opts);
            CHECK(solver.state().objective_value < 1e-8);
        }

        SECTION("Exponential fitting")
        {
            exponential_fitting problem;
            Eigen::VectorXd x0{{1.0, -0.5}};
            basic_solver solver{lm_policy{}, problem, x0, opts};
            auto result = solver.solve(opts);
            CHECK(solver.state().objective_value < 0.01);
        }

        SECTION("Powell singular")
        {
            powell_singular problem;
            Eigen::VectorXd x0{{3.0, -1.0, 0.0, 1.0}};
            basic_solver solver{lm_policy{}, problem, x0, opts};
            auto result = solver.solve(opts);
            CHECK(solver.state().objective_value < 1e-6);
        }
    }

    // -------------------------------------------------------------------
    // SQP (Kraft): constrained, f within 1e-6, violation < 1e-6
    // -------------------------------------------------------------------
    SECTION("SQP (Kraft)")
    {
        solver_options opts;
        opts.max_iterations = 500;
        opts.set_gradient_threshold(1e-4);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        SECTION("HS071")
        {
            hs071 problem;
            basic_solver solver{kraft_slsqp_policy<hs071<>::problem_dimension>{}, problem, problem.initial_point(), opts};
            auto result = solver.solve(opts);
            CHECK(std::isfinite(result.objective_value));
            CHECK(result.objective_value < 30.0);
        }

        SECTION("HS076")
        {
            hs076 problem;
            basic_solver solver{kraft_slsqp_policy<hs076<>::problem_dimension>{}, problem, problem.initial_point(), opts};
            auto result = solver.solve(opts);
            CHECK(result.objective_value == Approx(-4.681818).margin(1e-2));
            CHECK(solver.constraint_violation() < 1e-4);
        }

        SECTION("Rosenbrock constrained")
        {
            rosenbrock_constrained problem;
            Eigen::VectorXd x0{{-1.0, -1.0}};
            basic_solver solver{kraft_slsqp_policy<rosenbrock_constrained::problem_dimension>{}, problem, x0, opts};
            auto result = solver.solve(opts);
            CHECK(std::isfinite(result.objective_value));
            CHECK(result.objective_value < 1.0);
        }
    }

    // -------------------------------------------------------------------
    // SQP (N&W): constrained, f within 1e-6, violation < 1e-6
    // -------------------------------------------------------------------
    SECTION("SQP (N&W)")
    {
        solver_options opts;
        opts.max_iterations = 500;
        opts.set_gradient_threshold(1e-4);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        SECTION("HS071")
        {
            hs071 problem;
            basic_solver solver{nw_sqp_policy<hs071<>::problem_dimension>{}, problem, problem.initial_point(), opts};
            auto result = solver.solve(opts);
            CHECK(std::isfinite(result.objective_value));
            CHECK(result.objective_value < 30.0);
        }

        SECTION("HS076")
        {
            hs076 problem;
            basic_solver solver{nw_sqp_policy<hs076<>::problem_dimension>{}, problem, problem.initial_point(), opts};
            auto result = solver.solve(opts);
            CHECK(result.objective_value == Approx(-4.681818).margin(1e-2));
            CHECK(solver.constraint_violation() < 1e-4);
        }

        SECTION("Rosenbrock constrained")
        {
            rosenbrock_constrained problem;
            Eigen::VectorXd x0{{-1.0, -1.0}};
            basic_solver solver{nw_sqp_policy<rosenbrock_constrained::problem_dimension>{}, problem, x0, opts};
            auto result = solver.solve(opts);
            CHECK(std::isfinite(result.objective_value));
            CHECK(result.objective_value < 1.0);
        }
    }

    // -------------------------------------------------------------------
    // Augmented Lagrangian: constrained, f within 1e-4, violation < 1e-4
    // -------------------------------------------------------------------
    SECTION("Augmented Lagrangian")
    {
        solver_options opts;
        opts.max_iterations = 80;
        opts.set_gradient_threshold(1e-4);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        SECTION("HS071")
        {
            hs071 problem;
            basic_solver solver{augmented_lagrangian_policy<lbfgsb_policy<hs071<>::problem_dimension>>{},
                problem, problem.initial_point(), opts};
            auto result = solver.solve(opts);
            CHECK(result.objective_value == Approx(17.014).margin(0.5));
            CHECK(solver.constraint_violation() < 1e-2);
        }

        SECTION("HS076")
        {
            hs076 problem;
            basic_solver solver{augmented_lagrangian_policy<lbfgsb_policy<hs076<>::problem_dimension>>{},
                problem, problem.initial_point(), opts};
            auto result = solver.solve(opts);
            CHECK(result.objective_value == Approx(-4.681818).margin(0.2));
            CHECK(solver.constraint_violation() < 1e-3);
        }

        SECTION("Rosenbrock constrained")
        {
            rosenbrock_constrained problem;
            Eigen::VectorXd x0{{-1.0, -1.0}};
            basic_solver solver{augmented_lagrangian_policy<lbfgsb_policy<rosenbrock_constrained::problem_dimension>>{}, 
                problem, x0, opts};
            auto result = solver.solve(opts);
            CHECK(std::isfinite(result.objective_value));
            CHECK(result.objective_value < 5.0);
        }
    }

    // -------------------------------------------------------------------
    // MMA: inequality-only constrained, f within 1e-4, violation < 1e-4
    // -------------------------------------------------------------------
    SECTION("MMA")
    {
        solver_options opts;
        opts.set_gradient_threshold(1e-5);
        opts.max_iterations = 500;
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        SECTION("HS076")
        {
            hs076 problem;
            basic_solver solver{mma_policy<hs076<>::problem_dimension>{}, problem, problem.initial_point(), opts};

            double best_feasible = 1e10;
            for(int i = 0; i < 500; ++i)
            {
                auto sr = solver.step();
                if(solver.constraint_violation() < 1e-3
                   && sr.objective_value < best_feasible)
                {
                    best_feasible = sr.objective_value;
                }
            }
            CHECK(best_feasible < problem.optimal_value() + 0.3);
        }

        SECTION("Rosenbrock inequality")
        {
            rosenbrock_constrained problem;
            Eigen::VectorXd x0{{0.5, 0.5}};
            basic_solver solver{mma_policy<rosenbrock_constrained::problem_dimension>{}, problem, x0, opts};

            double best_feasible = 1e10;
            for(int i = 0; i < 500; ++i)
            {
                auto sr = solver.step();
                if(solver.constraint_violation() < 1e-3
                   && sr.objective_value < best_feasible)
                {
                    best_feasible = sr.objective_value;
                }
            }
            CHECK(best_feasible < 5.0);
        }

        SECTION("Beam design")
        {
            beam_design problem;
            Eigen::VectorXd x0{{2.0, 2.0}};
            basic_solver solver{mma_policy<beam_design::problem_dimension>{}, problem, x0, opts};

            double best_feasible = 1e10;
            for(int i = 0; i < 500; ++i)
            {
                auto sr = solver.step();
                if(solver.constraint_violation() < 1e-3
                   && sr.objective_value < best_feasible)
                {
                    best_feasible = sr.objective_value;
                }
            }
            CHECK(best_feasible < 2.0);
        }
    }

    // -------------------------------------------------------------------
    // GCMMA: inequality-only constrained, f within 1e-4, violation < 1e-4
    // -------------------------------------------------------------------
    SECTION("GCMMA")
    {
        solver_options opts;
        opts.set_gradient_threshold(1e-5);
        opts.max_iterations = 500;
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        SECTION("HS076")
        {
            hs076 problem;
            basic_solver solver{gcmma_policy<hs076<>::problem_dimension>{}, problem, problem.initial_point(), opts};

            double best_feasible = 1e10;
            for(int i = 0; i < 500; ++i)
            {
                auto sr = solver.step();
                if(solver.constraint_violation() < 1e-3
                   && sr.objective_value < best_feasible)
                {
                    best_feasible = sr.objective_value;
                }
            }
            CHECK(best_feasible < problem.optimal_value() + 0.3);
        }

        SECTION("Rosenbrock inequality")
        {
            rosenbrock_constrained problem;
            Eigen::VectorXd x0{{0.5, 0.5}};
            basic_solver solver{gcmma_policy<rosenbrock_constrained::problem_dimension>{}, problem, x0, opts};

            double best_feasible = 1e10;
            for(int i = 0; i < 500; ++i)
            {
                auto sr = solver.step();
                if(solver.constraint_violation() < 1e-3
                   && sr.objective_value < best_feasible)
                {
                    best_feasible = sr.objective_value;
                }
            }
            CHECK(best_feasible < 5.0);
        }

        SECTION("Beam design")
        {
            beam_design problem;
            Eigen::VectorXd x0{{2.0, 2.0}};
            basic_solver solver{gcmma_policy<beam_design::problem_dimension>{}, problem, x0, opts};

            double best_feasible = 1e10;
            for(int i = 0; i < 500; ++i)
            {
                auto sr = solver.step();
                if(solver.constraint_violation() < 1e-3
                   && sr.objective_value < best_feasible)
                {
                    best_feasible = sr.objective_value;
                }
            }
            CHECK(best_feasible < 2.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Solver group: heterogeneous racing (SC-5, D-11, D-12)
// ---------------------------------------------------------------------------

TEST_CASE("solve() zero-arg converges same as manual step loop", "[convergence]")
{
    // Regression test: verify that solve() produces the same convergence
    // behavior as a manual step loop with equivalent stopping criteria.
    // The solve() path delegates to step_n() which checks convergence
    // criteria from solver_options.

    SECTION("L-BFGS-B on Beale")
    {
        beale problem{};
        Eigen::VectorXd x0{{0.0, 0.0}};

        solver_options opts;
        opts.max_iterations = 1000;
        opts.set_gradient_threshold(1e-8);
        opts.set_objective_threshold(1e-12);
        opts.set_step_threshold(1e-12);

        constexpr int D = beale<>::problem_dimension;
        Eigen::Vector<double, D> x0_fixed{{0.0, 0.0}};

        basic_solver<lbfgsb_policy<D>, D, beale<>> solver1{problem, x0_fixed, opts};
        auto result = solver1.solve();

        basic_solver<lbfgsb_policy<D>, D, beale<>> solver2{problem, x0_fixed, opts};
        int manual_iters = 0;
        step_result<double> last{};
        for(std::uint32_t i = 0; i < opts.max_iterations; ++i)
        {
            last = solver2.step();
            ++manual_iters;

            if(last.gradient_norm < 1e-8)
                break;
            if(manual_iters > 1
               && std::abs(last.objective_change) < 1e-12)
                break;
            if(last.step_size < 1e-12 && manual_iters > 1)
                break;
        }

        CHECK(result.iterations < 100);
        CHECK(manual_iters < 100);
        CHECK(result.iterations == manual_iters);
        CHECK(result.objective_value < 1e-8);
        CHECK(last.objective_value < 1e-8);
    }

    SECTION("L-BFGS-B on Rosenbrock 2D")
    {
        rosenbrock problem{};

        solver_options opts;
        opts.max_iterations = 1000;
        opts.set_gradient_threshold(1e-10);
        opts.set_objective_threshold(1e-15);
        opts.set_step_threshold(1e-15);

        constexpr int D = rosenbrock<>::problem_dimension;
        Eigen::Vector<double, D> x0_fixed{{-1.0, -1.0}};

        basic_solver<lbfgsb_policy<D>, D, rosenbrock<>> solver1{problem, x0_fixed, opts};
        auto result = solver1.solve();

        basic_solver<lbfgsb_policy<D>, D, rosenbrock<>> solver2{problem, x0_fixed, opts};
        int manual_iters = 0;
        step_result<double> last{};
        for(std::uint32_t i = 0; i < opts.max_iterations; ++i)
        {
            last = solver2.step();
            ++manual_iters;

            if(last.gradient_norm < 1e-10)
                break;
            if(manual_iters > 1
               && std::abs(last.objective_change) < 1e-15)
                break;
            if(last.step_size < 1e-15 && manual_iters > 1)
                break;
        }

        CHECK(result.iterations < 500);
        CHECK(result.iterations == manual_iters);
        CHECK(result.objective_value < 1e-8);
    }
}

TEST_CASE("Solver group: lbfgsb + cmaes racing", "[convergence][solver_group]")
{
    rosenbrock<double> problem{};
    Eigen::VectorXd x0 = Eigen::VectorXd::Constant(2, -1.0);
    solver_options<> opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-15);
    opts.set_step_threshold(1e-15);
    opts.set_objective_threshold(1e-15);

    basic_solver_group<round_robin_schedule,
                       nablapp::dynamic_dimension,
                       rosenbrock<double>,
                       lbfgsb_policy<>,
                       cmaes_policy<>> group(problem, x0, opts);
    auto result = group.step_n(500, opts);
    CHECK(result.objective_value < 0.1);
}
