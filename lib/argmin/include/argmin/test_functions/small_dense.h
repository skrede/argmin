#ifndef HPP_GUARD_ARGMIN_TEST_FUNCTIONS_SMALL_DENSE_H
#define HPP_GUARD_ARGMIN_TEST_FUNCTIONS_SMALL_DENSE_H

// Small-dense, real-time-representative fixed-N fixtures.
//
// These problems span the unconstrained -> constrained range at dimensions
// characteristic of small condensed model-predictive control (a handful of
// controls over a short-to-moderate horizon), while staying well below the
// dense unblocked-Householder ceiling where the solver and Eigen's
// allocation-free decomposition path are both the right tool. Every accessor
// returns or writes fixed-size Eigen storage keyed on problem_dimension /
// constraint_count -- no dynamic-extent Eigen type appears here, since a
// dynamic accessor silently reintroduces the heap traffic these fixtures
// exist to keep out of the hot loop.
//
// Objective family: chained Rosenbrock (a narrow curved valley) so each
// consuming solver sustains a long pre-convergence descent rather than
// terminating in a couple of steps.
// Inequality convention: c_ineq >= 0 (argmin lower-bound form).
// Constraint layout: equality first (rows 0..n_eq-1), then inequality. The
// equality group sums pass through the all-ones point and the inequalities
// carry slack there, so the feasible set is well posed with LICQ holding
// along the constant-Jacobian rows.

#include "argmin/test_functions/problem_class.h"

#include <Eigen/Core>

namespace argmin
{

// sd006: 6D unconstrained chained Rosenbrock. The smallest dense anchor
// (one control, short horizon). Constraint hooks are present but empty so the
// fixture satisfies the constrained solver contract with zero constraint rows.
template <typename Scalar = double>
struct sd006
{
    static constexpr int problem_dimension = 6;
    static constexpr problem_class pclass = problem_class::unconstrained;
    static constexpr int constraint_count = 0;

    [[nodiscard]] int dimension() const { return problem_dimension; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        Scalar f = Scalar(0);
        for(int i = 0; i + 1 < problem_dimension; ++i)
        {
            const Scalar a = x[i + 1] - x[i] * x[i];
            const Scalar b = Scalar(1) - x[i];
            f += Scalar(100) * a * a + b * b;
        }
        return f;
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        g.setZero();
        for(int i = 0; i + 1 < problem_dimension; ++i)
        {
            const Scalar a = x[i + 1] - x[i] * x[i];
            g[i] += Scalar(-400) * a * x[i] - Scalar(2) * (Scalar(1) - x[i]);
            g[i + 1] += Scalar(200) * a;
        }
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>&, auto&) const {}
    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>&, auto&) const {}

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        Eigen::Vector<Scalar, problem_dimension> x0;
        for(int i = 0; i < problem_dimension; ++i)
            x0[i] = (i % 2 == 0) ? Scalar(-1.2) : Scalar(1);
        return x0;
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }
};

// sd012: 12D chained Rosenbrock with 2 equality + 4 inequality rows and box
// bounds. The mid anchor (two controls, moderate horizon). The two equality
// rows pin the sums of the first and second halves; the four inequalities cap
// or floor individual coordinates with slack at the all-ones point.
template <typename Scalar = double>
struct sd012
{
    static constexpr int problem_dimension = 12;
    static constexpr problem_class pclass = problem_class::mixed | problem_class::bound_constrained;
    static constexpr int constraint_count = 6;

    [[nodiscard]] int dimension() const { return problem_dimension; }
    [[nodiscard]] int num_equality() const { return 2; }
    [[nodiscard]] int num_inequality() const { return 4; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        Scalar f = Scalar(0);
        for(int i = 0; i + 1 < problem_dimension; ++i)
        {
            const Scalar a = x[i + 1] - x[i] * x[i];
            const Scalar b = Scalar(1) - x[i];
            f += Scalar(100) * a * a + b * b;
        }
        return f;
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        g.setZero();
        for(int i = 0; i + 1 < problem_dimension; ++i)
        {
            const Scalar a = x[i + 1] - x[i] * x[i];
            g[i] += Scalar(-400) * a * x[i] - Scalar(2) * (Scalar(1) - x[i]);
            g[i + 1] += Scalar(200) * a;
        }
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>& x, auto& c) const
    {
        c[0] = x[0] + x[1] + x[2] + x[3] + x[4] + x[5] - Scalar(6);
        c[1] = x[6] + x[7] + x[8] + x[9] + x[10] + x[11] - Scalar(6);
        c[2] = Scalar(4) - x[0];
        c[3] = Scalar(4) - x[6];
        c[4] = x[3] + Scalar(3);
        c[5] = x[9] + Scalar(3);
    }

    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>&, auto& J) const
    {
        J.setZero();
        for(int i = 0; i < 6; ++i)
            J(0, i) = Scalar(1);
        for(int i = 6; i < 12; ++i)
            J(1, i) = Scalar(1);
        J(2, 0) = Scalar(-1);
        J(3, 6) = Scalar(-1);
        J(4, 3) = Scalar(1);
        J(5, 9) = Scalar(1);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Constant(problem_dimension, Scalar(-5));
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Constant(problem_dimension, Scalar(5));
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        Eigen::Vector<Scalar, problem_dimension> x0;
        for(int i = 0; i < problem_dimension; ++i)
            x0[i] = (i % 2 == 0) ? Scalar(-1.2) : Scalar(1);
        return x0;
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }
};

// sd024: 24D chained Rosenbrock with 4 equality + 8 inequality rows and tight
// box bounds. The near-ceiling stress anchor (two controls, long horizon): the
// tighter box exercises the active set through the descent while staying well
// under the dense unblocked-Householder ceiling.
template <typename Scalar = double>
struct sd024
{
    static constexpr int problem_dimension = 24;
    static constexpr problem_class pclass = problem_class::mixed | problem_class::bound_constrained;
    static constexpr int constraint_count = 12;

    [[nodiscard]] int dimension() const { return problem_dimension; }
    [[nodiscard]] int num_equality() const { return 4; }
    [[nodiscard]] int num_inequality() const { return 8; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        Scalar f = Scalar(0);
        for(int i = 0; i + 1 < problem_dimension; ++i)
        {
            const Scalar a = x[i + 1] - x[i] * x[i];
            const Scalar b = Scalar(1) - x[i];
            f += Scalar(100) * a * a + b * b;
        }
        return f;
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        g.setZero();
        for(int i = 0; i + 1 < problem_dimension; ++i)
        {
            const Scalar a = x[i + 1] - x[i] * x[i];
            g[i] += Scalar(-400) * a * x[i] - Scalar(2) * (Scalar(1) - x[i]);
            g[i + 1] += Scalar(200) * a;
        }
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>& x, auto& c) const
    {
        for(int grp = 0; grp < 4; ++grp)
        {
            Scalar s = Scalar(0);
            for(int i = 0; i < 6; ++i)
                s += x[6 * grp + i];
            c[grp] = s - Scalar(6);
        }
        c[4] = Scalar(4) - x[0];
        c[5] = Scalar(4) - x[6];
        c[6] = Scalar(4) - x[12];
        c[7] = Scalar(4) - x[18];
        c[8] = x[3] + Scalar(3);
        c[9] = x[9] + Scalar(3);
        c[10] = x[15] + Scalar(3);
        c[11] = x[21] + Scalar(3);
    }

    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>&, auto& J) const
    {
        J.setZero();
        for(int grp = 0; grp < 4; ++grp)
            for(int i = 0; i < 6; ++i)
                J(grp, 6 * grp + i) = Scalar(1);
        J(4, 0) = Scalar(-1);
        J(5, 6) = Scalar(-1);
        J(6, 12) = Scalar(-1);
        J(7, 18) = Scalar(-1);
        J(8, 3) = Scalar(1);
        J(9, 9) = Scalar(1);
        J(10, 15) = Scalar(1);
        J(11, 21) = Scalar(1);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Constant(problem_dimension, Scalar(-2));
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Constant(problem_dimension, Scalar(2));
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        Eigen::Vector<Scalar, problem_dimension> x0;
        for(int i = 0; i < problem_dimension; ++i)
            x0[i] = (i % 2 == 0) ? Scalar(-1.2) : Scalar(1);
        return x0;
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }
};

// sd_ls012: 12D extended-Rosenbrock nonlinear least-squares (twelve residuals).
// The mid-size least-squares anchor for the damped Gauss-Newton hot loop,
// mirroring the residual/jacobian interface of the 2D Rosenbrock least-squares
// fixture the micro least-squares probe already drives.
template <typename Scalar = double>
struct sd_ls012
{
    static constexpr int problem_dimension = 12;

    [[nodiscard]] int dimension() const { return problem_dimension; }
    [[nodiscard]] int num_residuals() const { return problem_dimension; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        Scalar f = Scalar(0);
        for(int k = 0; k < problem_dimension / 2; ++k)
        {
            const Scalar r0 = Scalar(10) * (x[2 * k + 1] - x[2 * k] * x[2 * k]);
            const Scalar r1 = Scalar(1) - x[2 * k];
            f += r0 * r0 + r1 * r1;
        }
        return Scalar(0.5) * f;
    }

    void residuals(const Eigen::Vector<Scalar, problem_dimension>& x, auto& r) const
    {
        for(int k = 0; k < problem_dimension / 2; ++k)
        {
            r[2 * k] = Scalar(10) * (x[2 * k + 1] - x[2 * k] * x[2 * k]);
            r[2 * k + 1] = Scalar(1) - x[2 * k];
        }
    }

    void jacobian(const Eigen::Vector<Scalar, problem_dimension>& x, auto& J) const
    {
        J.setZero();
        for(int k = 0; k < problem_dimension / 2; ++k)
        {
            J(2 * k, 2 * k) = Scalar(-20) * x[2 * k];
            J(2 * k, 2 * k + 1) = Scalar(10);
            J(2 * k + 1, 2 * k) = Scalar(-1);
        }
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        Eigen::Vector<Scalar, problem_dimension> x0;
        for(int i = 0; i < problem_dimension; ++i)
            x0[i] = (i % 2 == 0) ? Scalar(-1.2) : Scalar(1);
        return x0;
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }
};

}

#endif
