#ifndef HPP_GUARD_ARGMIN_TESTS_UNIT_ALLOC_GATE_FIXTURES_H
#define HPP_GUARD_ARGMIN_TESTS_UNIT_ALLOC_GATE_FIXTURES_H

// Fixed-dimension problem fixtures for the zero-allocation correctness gates.
// Relocated from the retired benchmarks/micro_*.cpp alloc-trace drivers. Each
// fixture pins a compile-time dimension so step_budget_solver's reset() copy
// stays on the stack and the gate measures only genuine policy heap traffic.

#include <Eigen/Core>

#include <cmath>

namespace argmin::alloc_gate
{

// 12-D chained Rosenbrock with 2 equality + 4 inequality rows (6 runtime
// constraints) and box bounds, but a declared constraint_count of 12 -- the
// runtime constraint count sits strictly below the compile-time cap, so the
// qp_result multiplier storage is inline-max-bounded while carrying fewer
// active rows than its capacity. The constrained optimum is the all-ones point
// (feasible, objective 0). Math mirrors sd012; only the declared cap differs,
// so this exercises the runtime-m < cap path the sd012/sd024 gates
// (runtime-m == cap) never reach.
struct undercap_rosenbrock
{
    static constexpr int problem_dimension = 12;
    static constexpr int constraint_count = 12;

    [[nodiscard]] int dimension() const { return problem_dimension; }
    [[nodiscard]] int num_equality() const { return 2; }
    [[nodiscard]] int num_inequality() const { return 4; }

    [[nodiscard]] double value(const Eigen::Vector<double, problem_dimension>& x) const
    {
        double f = 0.0;
        for(int i = 0; i + 1 < problem_dimension; ++i)
        {
            const double a = x[i + 1] - x[i] * x[i];
            const double b = 1.0 - x[i];
            f += 100.0 * a * a + b * b;
        }
        return f;
    }

    void gradient(const Eigen::Vector<double, problem_dimension>& x,
                  Eigen::Vector<double, problem_dimension>& g) const
    {
        g.setZero();
        for(int i = 0; i + 1 < problem_dimension; ++i)
        {
            const double a = x[i + 1] - x[i] * x[i];
            g[i] += -400.0 * a * x[i] - 2.0 * (1.0 - x[i]);
            g[i + 1] += 200.0 * a;
        }
    }

    void constraints(const Eigen::Vector<double, problem_dimension>& x, auto& c) const
    {
        c[0] = x[0] + x[1] + x[2] + x[3] + x[4] + x[5] - 6.0;
        c[1] = x[6] + x[7] + x[8] + x[9] + x[10] + x[11] - 6.0;
        c[2] = 4.0 - x[0];
        c[3] = 4.0 - x[6];
        c[4] = x[3] + 3.0;
        c[5] = x[9] + 3.0;
    }

    void constraint_jacobian(const Eigen::Vector<double, problem_dimension>&, auto& J) const
    {
        J.setZero();
        for(int i = 0; i < 6; ++i)
            J(0, i) = 1.0;
        for(int i = 6; i < 12; ++i)
            J(1, i) = 1.0;
        J(2, 0) = -1.0;
        J(3, 6) = -1.0;
        J(4, 3) = 1.0;
        J(5, 9) = 1.0;
    }

    [[nodiscard]] Eigen::Vector<double, problem_dimension> lower_bounds() const
    {
        return Eigen::Vector<double, problem_dimension>::Constant(problem_dimension, -5.0);
    }

    [[nodiscard]] Eigen::Vector<double, problem_dimension> upper_bounds() const
    {
        return Eigen::Vector<double, problem_dimension>::Constant(problem_dimension, 5.0);
    }

    [[nodiscard]] Eigen::Vector<double, problem_dimension> initial_point() const
    {
        Eigen::Vector<double, problem_dimension> x0;
        for(int i = 0; i < problem_dimension; ++i)
            x0[i] = (i % 2 == 0) ? -1.2 : 1.0;
        return x0;
    }

    [[nodiscard]] double optimal_value() const { return 0.0; }
};

// Compile-time N=4 HS071 fixture for the kraft_slsqp gate.
struct hs071_fixed
{
    static constexpr int problem_dimension = 4;

    int dimension() const { return 4; }

    double value(const Eigen::Vector<double, 4>& x) const
    {
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
    }

    void gradient(const Eigen::Vector<double, 4>& x,
                  Eigen::Vector<double, 4>& g) const
    {
        g[0] = x[3] * (x[0] + x[1] + x[2]) + x[0] * x[3];
        g[1] = x[0] * x[3];
        g[2] = x[0] * x[3] + 1.0;
        g[3] = x[0] * (x[0] + x[1] + x[2]);
    }

    Eigen::Vector<double, 4> lower_bounds() const
    {
        return Eigen::Vector<double, 4>{1.0, 1.0, 1.0, 1.0};
    }

    Eigen::Vector<double, 4> upper_bounds() const
    {
        return Eigen::Vector<double, 4>{5.0, 5.0, 5.0, 5.0};
    }

    int num_equality() const { return 1; }
    int num_inequality() const { return 1; }

    void constraints(const Eigen::Vector<double, 4>& x,
                     Eigen::VectorXd& c) const
    {
        c.resize(2);
        c[0] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3] - 40.0;
        c[1] = x[0] * x[1] * x[2] * x[3] - 25.0;
    }

    void constraint_jacobian(const Eigen::Vector<double, 4>& x,
                             Eigen::MatrixXd& J) const
    {
        J.resize(2, 4);
        J(0, 0) = 2.0 * x[0];
        J(0, 1) = 2.0 * x[1];
        J(0, 2) = 2.0 * x[2];
        J(0, 3) = 2.0 * x[3];
        J(1, 0) = x[1] * x[2] * x[3];
        J(1, 1) = x[0] * x[2] * x[3];
        J(1, 2) = x[0] * x[1] * x[3];
        J(1, 3) = x[0] * x[1] * x[2];
    }
};

// Compile-time N=4 HS071 fixture for the filter_slsqp gate.
struct hs071_fixed_gate
{
    static constexpr int problem_dimension = 4;

    [[nodiscard]] int dimension() const { return 4; }
    [[nodiscard]] int num_equality() const { return 1; }
    [[nodiscard]] int num_inequality() const { return 1; }

    [[nodiscard]] double value(const Eigen::Vector<double, 4>& x) const
    {
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
    }

    void gradient(const Eigen::Vector<double, 4>& x,
                  Eigen::Vector<double, 4>& g) const
    {
        g[0] = x[3] * (x[0] + x[1] + x[2]) + x[0] * x[3];
        g[1] = x[0] * x[3];
        g[2] = x[0] * x[3] + 1.0;
        g[3] = x[0] * (x[0] + x[1] + x[2]);
    }

    void constraints(const Eigen::Vector<double, 4>& x, Eigen::VectorXd& c) const
    {
        c.resize(2);
        c[0] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3] - 40.0;
        c[1] = x[0] * x[1] * x[2] * x[3] - 25.0;
    }

    void constraint_jacobian(const Eigen::Vector<double, 4>& x,
                             Eigen::MatrixXd& J) const
    {
        J.resize(2, 4);
        J(0, 0) = 2.0 * x[0]; J(0, 1) = 2.0 * x[1];
        J(0, 2) = 2.0 * x[2]; J(0, 3) = 2.0 * x[3];
        J(1, 0) = x[1] * x[2] * x[3]; J(1, 1) = x[0] * x[2] * x[3];
        J(1, 2) = x[0] * x[1] * x[3]; J(1, 3) = x[0] * x[1] * x[2];
    }

    [[nodiscard]] Eigen::Vector<double, 4> lower_bounds() const
    {
        return Eigen::Vector<double, 4>::Constant(1.0);
    }

    [[nodiscard]] Eigen::Vector<double, 4> upper_bounds() const
    {
        return Eigen::Vector<double, 4>::Constant(5.0);
    }
};

// Compile-time N=3 HS026 fixture for the fixed-N trust-region gates. HS026 is a
// NONLINEAR-equality problem (no bounds, no inequality), so it drives the
// composite-step machinery -- dogleg normal step, Steihaug-CG tangential leg,
// equality-multiplier re-estimation, second-order-correction retry -- WITHOUT
// engaging the box-face tangential free-set restart.
struct hs026_fixed_gate
{
    static constexpr int problem_dimension = 3;

    [[nodiscard]] int dimension() const { return 3; }
    [[nodiscard]] int num_equality() const { return 1; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] double value(const Eigen::Vector<double, 3>& x) const
    {
        const double d01 = x[0] - x[1];
        const double d12 = x[1] - x[2];
        return d01 * d01 + d12 * d12 * d12 * d12;
    }

    void gradient(const Eigen::Vector<double, 3>& x,
                  Eigen::Vector<double, 3>& g) const
    {
        const double d01 = x[0] - x[1];
        const double d12 = x[1] - x[2];
        g[0] = 2.0 * d01;
        g[1] = -2.0 * d01 + 4.0 * d12 * d12 * d12;
        g[2] = -4.0 * d12 * d12 * d12;
    }

    void constraints(const Eigen::Vector<double, 3>& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = (1.0 + x[1] * x[1]) * x[0] + x[2] * x[2] * x[2] * x[2] - 3.0;
    }

    void constraint_jacobian(const Eigen::Vector<double, 3>& x,
                             Eigen::MatrixXd& J) const
    {
        J.resize(1, 3);
        J(0, 0) = 1.0 + x[1] * x[1];
        J(0, 1) = 2.0 * x[1] * x[0];
        J(0, 2) = 4.0 * x[2] * x[2] * x[2];
    }
};

// Fixed-dimension wide-bounds Rosenbrock for the L-BFGS-B gate family. Every
// variable stays interior, so the solver holds the all-free two-loop-recursion
// path and never touches the bound-active branch.
struct rosenbrock_fixed
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        const double t1 = 1.0 - x[0];
        const double t2 = x[1] - x[0] * x[0];
        return t1 * t1 + 100.0 * t2 * t2;
    }

    void gradient(const Eigen::Vector<double, 2>& x,
                  Eigen::Vector<double, 2>& g) const
    {
        g[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
        g[1] = 200.0 * (x[1] - x[0] * x[0]);
    }

    Eigen::Vector<double, 2> lower_bounds() const
    {
        return Eigen::Vector<double, 2>{-5.0, -5.0};
    }

    Eigen::Vector<double, 2> upper_bounds() const
    {
        return Eigen::Vector<double, 2>{5.0, 5.0};
    }
};

// Bound-active fixture for the L-BFGS-B gate family. The unconstrained
// minimizer lies OUTSIDE the box on the first axis (x0* = 2, upper bound 1) and
// interior on the second (x1* = 0.5): the constrained solution pins x0 to its
// upper bound while x1 stays free, so every steady-state step enters the
// bound-active branch (GCP -> free-variable subspace -> reduced_hessian) with a
// non-empty free set.
struct bounded_quadratic_fixed
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        const double a = x[0] - 2.0;
        const double b = x[1] - 0.5;
        return a * a + b * b;
    }

    void gradient(const Eigen::Vector<double, 2>& x,
                  Eigen::Vector<double, 2>& g) const
    {
        g[0] = 2.0 * (x[0] - 2.0);
        g[1] = 2.0 * (x[1] - 0.5);
    }

    Eigen::Vector<double, 2> lower_bounds() const
    {
        return Eigen::Vector<double, 2>{0.0, 0.0};
    }

    Eigen::Vector<double, 2> upper_bounds() const
    {
        return Eigen::Vector<double, 2>{1.0, 1.0};
    }
};

// Fixed-2 bounded linear least-squares. r(x) = x - b with b = (2, 0.25): the
// first coordinate's unconstrained minimizer (2) lies outside the box
// [-2, 0.5], so x0 pins at its upper bound 0.5 and stays active on every step,
// exercising the active-set / free-set path with a stable free set {1}.
struct bounded_linear_ls_fixed2
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }
    int num_residuals() const { return 2; }

    static Eigen::Vector2d bvec() { return Eigen::Vector2d{2.0, 0.25}; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        return 0.5 * (x - bvec()).squaredNorm();
    }
    void residuals(const Eigen::Vector<double, 2>& x, Eigen::VectorXd& r) const
    {
        r = x - bvec();
    }
    void jacobian(const Eigen::Vector<double, 2>& /*x*/, Eigen::MatrixXd& J) const
    {
        J = Eigen::Matrix2d::Identity();
    }
    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd{{-2.0, -2.0}}; }
    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd{{0.5, 2.0}}; }
};

// Fixed-2 Rosenbrock least-squares for the Levenberg-Marquardt gate.
struct rosenbrock_ls_fixed
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

    void residuals(const Eigen::Vector<double, 2>& x,
                   Eigen::VectorXd& r) const
    {
        r(0) = 1.0 - x(0);
        r(1) = std::sqrt(5.0) * (x(1) - x(0) * x(0));
    }

    void jacobian(const Eigen::Vector<double, 2>& x,
                  Eigen::MatrixXd& J) const
    {
        J(0, 0) = -1.0;
        J(0, 1) = 0.0;
        J(1, 0) = -2.0 * std::sqrt(5.0) * x(0);
        J(1, 1) = std::sqrt(5.0);
    }

    Eigen::Vector<double, 2> initial_point() const
    {
        return Eigen::Vector<double, 2>{-1.0, 1.0};
    }
};

}

#endif
