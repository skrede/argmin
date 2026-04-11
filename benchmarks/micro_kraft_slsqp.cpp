// Micro-benchmark: nablapp kraft_slsqp vs NLopt LD_SLSQP on HS071.
//
// HS071 is a mixed-constraint problem (n=4, m_eq=1, m_ineq=1) at
// IK-relevant scale, testing per-step wall time for constrained SQP.
//
// Reference: Hock, W. and Schittkowski, K. (1981) "Test Examples for
//            Nonlinear Programming Codes", Lecture Notes in Economics
//            and Mathematical Systems, Vol. 187, Springer.

#include "nablapp/solver/kraft_slsqp_policy.h"
#include "nablapp/solver/nw_sqp_policy.h"
#include "nablapp/solver/basic_solver.h"

#include <Eigen/Core>

#include <nlopt.hpp>

#include <array>
#include <print>
#include <chrono>
#include <cstdint>
#include <optional>

namespace
{

// HS071: n=4, 1 equality, 1 inequality, box bounds.
//
//   min   x1*x4*(x1+x2+x3) + x3
//   s.t.  x1*x2*x3*x4 >= 25          (inequality: c_ineq >= 0)
//         x1^2+x2^2+x3^2+x4^2 = 40   (equality:   c_eq  = 0)
//         1 <= xi <= 5
//   x0 = {1, 5, 5, 1}
//   f* ~ 17.014
//
// This struct uses `problem_dimension = dynamic_dimension` to exercise
// the runtime-dimension path through kraft_slsqp_policy<-1>,
// kraft_lsq_qp_solver<double, -1>, and all the dynamic Eigen
// specializations. See hs071_fixed below for the compile-time N=4
// variant; comparing the two measures the fraction of per-step cost
// attributable to N being runtime vs compile-time.
struct hs071
{
    static constexpr int problem_dimension = nablapp::dynamic_dimension;

    int dimension() const { return 4; }

    double value(const Eigen::VectorXd& x) const
    {
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        g[0] = x[3] * (x[0] + x[1] + x[2]) + x[0] * x[3];
        g[1] = x[0] * x[3];
        g[2] = x[0] * x[3] + 1.0;
        g[3] = x[0] * (x[0] + x[1] + x[2]);
    }

    Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd{{1.0, 1.0, 1.0, 1.0}};
    }

    Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd{{5.0, 5.0, 5.0, 5.0}};
    }

    int num_equality() const { return 1; }
    int num_inequality() const { return 1; }

    // Combined constraint vector: [equality, inequality].
    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(2);
        // Equality: x1^2+x2^2+x3^2+x4^2 - 40 = 0
        c[0] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3] - 40.0;
        // Inequality: x1*x2*x3*x4 - 25 >= 0
        c[1] = x[0] * x[1] * x[2] * x[3] - 25.0;
    }

    void constraint_jacobian(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const
    {
        J.resize(2, 4);
        // Equality Jacobian
        J(0, 0) = 2.0 * x[0];
        J(0, 1) = 2.0 * x[1];
        J(0, 2) = 2.0 * x[2];
        J(0, 3) = 2.0 * x[3];
        // Inequality Jacobian
        J(1, 0) = x[1] * x[2] * x[3];
        J(1, 1) = x[0] * x[2] * x[3];
        J(1, 2) = x[0] * x[1] * x[3];
        J(1, 3) = x[0] * x[1] * x[2];
    }
};

// Compile-time fixed-N=4 variant of HS071. Exercises the compile-time
// dimension path through kraft_slsqp_policy<4>,
// kraft_lsq_qp_solver<double, 4>, adaptive_bfgs<double, 4, 10>, and
// lsi<double, 4>. The workspace-dimension fixes in adaptive_bfgs.h
// and lsi.h only take effect on this path; the dynamic-N hs071 struct
// above is insensitive to them. Comparing per-step timings across the
// two structs on the same problem isolates the benefit of compile-time
// dimension propagation at n=4 and tells us what fraction of the per-
// step cost comes from dynamic-dispatch overhead in Eigen's inner
// loops.
//
// All method signatures mirror hs071 but use Eigen::Vector<double, 4>
// and Eigen::Matrix<double, 2, 4> for the state types so the solver's
// state_type picks up the fixed-size storage via CTAD.
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

// Synthetic 6-DoF ill-conditioned equality-constrained least squares.
//
// f(x) = 0.5 * sum_i M_ii^2 * (x_i - x*_i)^2 with a diagonal M whose
// singular-value spread is exactly 1e4 by construction. Two linear
// equality constraints place the feasible manifold off the principal
// axes so the solver must cooperate across all coordinates.
//
// The problem is linear-algebra-only. It exercises the same SLSQP hot
// path that a typical 6-DoF Jacobian-based workload exercises
// (kraft_slsqp_policy + kraft_lsq_qp_solver + adaptive_bfgs).
//
// Optimum lies on the intersection of the two hyperplanes
//   e^T x = 6  and  [1 -1 1 -1 1 -1]^T x = 0
// which both pass through x* = (1,1,1,1,1,1), so f_star = 0.

// Condition number kappa(M) = 1e4 by construction:
// M_ii = 10^(-0.8 * i) for i = 0..5, so M_00 = 1.0 and M_55 = 1e-4.
inline constexpr std::array<double, 6> synthetic_6dof_m_diag{
    1.0,
    1.584893192461114e-1,   // 10^-0.8
    2.511886431509580e-2,   // 10^-1.6
    3.981071705534972e-3,   // 10^-2.4
    6.309573444801934e-4,   // 10^-3.2
    1.0e-4
};
inline constexpr std::array<double, 6> synthetic_6dof_x_star{
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0
};

struct synthetic_6dof
{
    static constexpr int problem_dimension = nablapp::dynamic_dimension;

    int dimension() const { return 6; }

    double value(const Eigen::VectorXd& x) const
    {
        double f = 0.0;
        for(std::size_t i = 0; i < 6; ++i)
        {
            const double d = x[static_cast<Eigen::Index>(i)]
                             - synthetic_6dof_x_star[i];
            f += 0.5 * synthetic_6dof_m_diag[i]
                     * synthetic_6dof_m_diag[i] * d * d;
        }
        return f;
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        for(std::size_t i = 0; i < 6; ++i)
        {
            const double d = x[static_cast<Eigen::Index>(i)]
                             - synthetic_6dof_x_star[i];
            g[static_cast<Eigen::Index>(i)] = synthetic_6dof_m_diag[i]
                                            * synthetic_6dof_m_diag[i] * d;
        }
    }

    Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd::Constant(6, -1.0e6);
    }

    Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd::Constant(6, 1.0e6);
    }

    int num_equality() const { return 2; }
    int num_inequality() const { return 0; }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(2);
        // Row 0: sum of components = 6
        c[0] = x.sum() - 6.0;
        // Row 1: alternating sum = 0
        c[1] = x[0] - x[1] + x[2] - x[3] + x[4] - x[5];
    }

    void constraint_jacobian(const Eigen::VectorXd& /*x*/,
                             Eigen::MatrixXd& J) const
    {
        J.resize(2, 6);
        J.row(0) = Eigen::RowVectorXd::Constant(6, 1.0);
        J(1, 0) = 1.0;  J(1, 1) = -1.0;
        J(1, 2) = 1.0;  J(1, 3) = -1.0;
        J(1, 4) = 1.0;  J(1, 5) = -1.0;
    }
};

// Compile-time N=6 variant of synthetic_6dof. Uses
// Eigen::Vector<double, 6> for x/g state types so the solver's
// state_type picks up fixed-size storage via CTAD. Constraint vector
// and Jacobian still use the dynamic types because kraft_slsqp_policy
// stores them dynamically internally (matching the hs071_fixed
// precedent).
struct synthetic_6dof_fixed
{
    static constexpr int problem_dimension = 6;

    int dimension() const { return 6; }

    double value(const Eigen::Vector<double, 6>& x) const
    {
        double f = 0.0;
        for(std::size_t i = 0; i < 6; ++i)
        {
            const double d = x[static_cast<Eigen::Index>(i)]
                             - synthetic_6dof_x_star[i];
            f += 0.5 * synthetic_6dof_m_diag[i]
                     * synthetic_6dof_m_diag[i] * d * d;
        }
        return f;
    }

    void gradient(const Eigen::Vector<double, 6>& x,
                  Eigen::Vector<double, 6>& g) const
    {
        for(std::size_t i = 0; i < 6; ++i)
        {
            const double d = x[static_cast<Eigen::Index>(i)]
                             - synthetic_6dof_x_star[i];
            g[static_cast<Eigen::Index>(i)] = synthetic_6dof_m_diag[i]
                                            * synthetic_6dof_m_diag[i] * d;
        }
    }

    Eigen::Vector<double, 6> lower_bounds() const
    {
        return Eigen::Vector<double, 6>::Constant(-1.0e6);
    }

    Eigen::Vector<double, 6> upper_bounds() const
    {
        return Eigen::Vector<double, 6>::Constant(1.0e6);
    }

    int num_equality() const { return 2; }
    int num_inequality() const { return 0; }

    void constraints(const Eigen::Vector<double, 6>& x,
                     Eigen::VectorXd& c) const
    {
        c.resize(2);
        c[0] = x.sum() - 6.0;
        c[1] = x[0] - x[1] + x[2] - x[3] + x[4] - x[5];
    }

    void constraint_jacobian(const Eigen::Vector<double, 6>& /*x*/,
                             Eigen::MatrixXd& J) const
    {
        J.resize(2, 6);
        J.row(0) = Eigen::RowVectorXd::Constant(6, 1.0);
        J(1, 0) = 1.0;  J(1, 1) = -1.0;
        J(1, 2) = 1.0;  J(1, 3) = -1.0;
        J(1, 4) = 1.0;  J(1, 5) = -1.0;
    }
};

// NLopt objective callback.
double nlopt_hs071_objective(unsigned, const double* x, double* grad, void*)
{
    if(grad)
    {
        grad[0] = x[3] * (x[0] + x[1] + x[2]) + x[0] * x[3];
        grad[1] = x[0] * x[3];
        grad[2] = x[0] * x[3] + 1.0;
        grad[3] = x[0] * (x[0] + x[1] + x[2]);
    }
    return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
}

// NLopt equality constraint: x1^2+x2^2+x3^2+x4^2 - 40 = 0
double nlopt_hs071_equality(unsigned, const double* x, double* grad, void*)
{
    if(grad)
    {
        grad[0] = 2.0 * x[0];
        grad[1] = 2.0 * x[1];
        grad[2] = 2.0 * x[2];
        grad[3] = 2.0 * x[3];
    }
    return x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3] - 40.0;
}

// NLopt inequality constraint: 25 - x1*x2*x3*x4 <= 0
// NLopt convention: c(x) <= 0, so we negate the >= 0 form.
double nlopt_hs071_inequality(unsigned, const double* x, double* grad, void*)
{
    if(grad)
    {
        grad[0] = -x[1] * x[2] * x[3];
        grad[1] = -x[0] * x[2] * x[3];
        grad[2] = -x[0] * x[1] * x[3];
        grad[3] = -x[0] * x[1] * x[2];
    }
    return 25.0 - x[0] * x[1] * x[2] * x[3];
}

struct timing
{
    double wall_us;
    double objective;
    std::uint32_t evals;
    double per_step_us;
    double line_search_calls_per_step;  // kraft_slsqp only; 0 for others
};

timing bench_nablapp(std::uint32_t reps)
{
    hs071 problem;
    Eigen::VectorXd x0{{1.0, 5.0, 5.0, 1.0}};
    nablapp::solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-10);

    // Warmup + convergence check.
    std::uint32_t iters = 0;
    {
        nablapp::basic_solver solver{nablapp::kraft_slsqp_policy{}, problem, x0, opts};
        auto result = solver.solve();
        iters = static_cast<std::uint32_t>(result.iterations);
        if(std::abs(result.objective_value - 17.014) > 0.5)
            std::println("WARNING: kraft_slsqp did not converge correctly, f={:.6e}",
                         result.objective_value);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint64_t total_ls_calls = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nablapp::basic_solver solver{nablapp::kraft_slsqp_policy{}, problem, x0, opts};
        auto result = solver.solve();
        fval = result.objective_value;
        iters = static_cast<std::uint32_t>(result.iterations);
        total_ls_calls += solver.state().line_search_calls;
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double per_solve = total_us / reps;
    double per_step = total_us / (reps * iters);
    double ls_per_step = static_cast<double>(total_ls_calls) / (reps * iters);
    return {per_solve, fval, iters, per_step, ls_per_step};
}

timing bench_nablapp_fixed(std::uint32_t reps)
{
    hs071_fixed problem;
    Eigen::Vector<double, 4> x0{1.0, 5.0, 5.0, 1.0};
    nablapp::solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-10);

    // Warmup + convergence check.
    std::uint32_t iters = 0;
    {
        nablapp::basic_solver solver{nablapp::kraft_slsqp_policy<4>{}, problem, x0, opts};
        auto result = solver.solve();
        iters = static_cast<std::uint32_t>(result.iterations);
        if(std::abs(result.objective_value - 17.014) > 0.5)
            std::println("WARNING: kraft_slsqp<4> did not converge correctly, f={:.6e}",
                         result.objective_value);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint64_t total_ls_calls = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nablapp::basic_solver solver{nablapp::kraft_slsqp_policy<4>{}, problem, x0, opts};
        auto result = solver.solve();
        fval = result.objective_value;
        iters = static_cast<std::uint32_t>(result.iterations);
        total_ls_calls += solver.state().line_search_calls;
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double per_solve = total_us / reps;
    double per_step = total_us / (reps * iters);
    double ls_per_step = static_cast<double>(total_ls_calls) / (reps * iters);
    return {per_solve, fval, iters, per_step, ls_per_step};
}

timing bench_nablapp_nw_sqp(std::uint32_t reps)
{
    hs071 problem;
    Eigen::VectorXd x0{{1.0, 5.0, 5.0, 1.0}};
    nablapp::solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-10);

    // Warmup + convergence check.
    std::uint32_t iters = 0;
    {
        nablapp::basic_solver solver{nablapp::nw_sqp_policy{}, problem, x0, opts};
        auto result = solver.solve();
        iters = static_cast<std::uint32_t>(result.iterations);
        if(std::abs(result.objective_value - 17.014) > 0.5)
            std::println("WARNING: nw_sqp did not converge correctly, f={:.6e}",
                         result.objective_value);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nablapp::basic_solver solver{nablapp::nw_sqp_policy{}, problem, x0, opts};
        auto result = solver.solve();
        fval = result.objective_value;
        iters = static_cast<std::uint32_t>(result.iterations);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double per_solve = total_us / reps;
    double per_step = total_us / (reps * iters);
    return {per_solve, fval, iters, per_step, 0.0};
}

timing bench_nlopt(std::uint32_t reps)
{
    // Warmup.
    std::uint32_t evals = 0;
    {
        nlopt::opt opt(nlopt::LD_SLSQP, 4);
        opt.set_min_objective(nlopt_hs071_objective, nullptr);
        opt.set_lower_bounds({1.0, 1.0, 1.0, 1.0});
        opt.set_upper_bounds({5.0, 5.0, 5.0, 5.0});
        opt.add_equality_constraint(nlopt_hs071_equality, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs071_inequality, nullptr, 1e-10);
        opt.set_maxeval(200);
        opt.set_ftol_rel(1e-10);
        opt.set_xtol_rel(1e-10);
        std::vector<double> x = {1.0, 5.0, 5.0, 1.0};
        double fval;
        opt.optimize(x, fval);
        evals = static_cast<std::uint32_t>(opt.get_numevals());
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nlopt::opt opt(nlopt::LD_SLSQP, 4);
        opt.set_min_objective(nlopt_hs071_objective, nullptr);
        opt.set_lower_bounds({1.0, 1.0, 1.0, 1.0});
        opt.set_upper_bounds({5.0, 5.0, 5.0, 5.0});
        opt.add_equality_constraint(nlopt_hs071_equality, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs071_inequality, nullptr, 1e-10);
        opt.set_maxeval(200);
        opt.set_ftol_rel(1e-10);
        opt.set_xtol_rel(1e-10);
        std::vector<double> x = {1.0, 5.0, 5.0, 1.0};
        opt.optimize(x, fval);
        evals = static_cast<std::uint32_t>(opt.get_numevals());
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double per_solve = total_us / reps;
    double per_step = total_us / (reps * evals);
    return {per_solve, fval, evals, per_step, 0.0};
}

// Synthetic 6-DoF NLopt callbacks. Shared constants with the nablapp
// problem structs via the file-scope synthetic_6dof_m_diag / x_star.
double nlopt_synthetic_6dof_objective(unsigned, const double* x,
                                      double* grad, void*)
{
    double f = 0.0;
    for(std::size_t i = 0; i < 6; ++i)
    {
        const double d = x[i] - synthetic_6dof_x_star[i];
        const double m2 = synthetic_6dof_m_diag[i] * synthetic_6dof_m_diag[i];
        f += 0.5 * m2 * d * d;
        if(grad)
            grad[i] = m2 * d;
    }
    return f;
}

// Equality constraint 0: sum_i x_i - 6 = 0.
double nlopt_synthetic_6dof_eq0(unsigned, const double* x,
                                double* grad, void*)
{
    if(grad)
    {
        for(std::size_t i = 0; i < 6; ++i)
            grad[i] = 1.0;
    }
    return x[0] + x[1] + x[2] + x[3] + x[4] + x[5] - 6.0;
}

// Equality constraint 1: x0 - x1 + x2 - x3 + x4 - x5 = 0.
double nlopt_synthetic_6dof_eq1(unsigned, const double* x,
                                double* grad, void*)
{
    if(grad)
    {
        grad[0] = 1.0; grad[1] = -1.0;
        grad[2] = 1.0; grad[3] = -1.0;
        grad[4] = 1.0; grad[5] = -1.0;
    }
    return x[0] - x[1] + x[2] - x[3] + x[4] - x[5];
}

timing bench_nablapp_6dof(std::uint32_t reps)
{
    synthetic_6dof problem;
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(6);
    nablapp::solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-10);

    std::uint32_t iters = 0;
    {
        nablapp::basic_solver solver{nablapp::kraft_slsqp_policy{}, problem, x0, opts};
        auto result = solver.solve();
        iters = static_cast<std::uint32_t>(result.iterations);
        if(!std::isfinite(result.objective_value))
            std::println("WARNING: kraft_slsqp synthetic 6-DoF diverged, f={:.6e}",
                         result.objective_value);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint64_t total_ls_calls = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nablapp::basic_solver solver{nablapp::kraft_slsqp_policy{}, problem, x0, opts};
        auto result = solver.solve();
        fval = result.objective_value;
        iters = static_cast<std::uint32_t>(result.iterations);
        total_ls_calls += solver.state().line_search_calls;
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double per_solve = total_us / reps;
    double per_step = total_us / (reps * iters);
    double ls_per_step = static_cast<double>(total_ls_calls) / (reps * iters);
    return {per_solve, fval, iters, per_step, ls_per_step};
}

timing bench_nablapp_6dof_fixed(std::uint32_t reps)
{
    synthetic_6dof_fixed problem;
    Eigen::Vector<double, 6> x0 = Eigen::Vector<double, 6>::Zero();
    nablapp::solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-10);

    std::uint32_t iters = 0;
    {
        nablapp::basic_solver solver{nablapp::kraft_slsqp_policy<6>{}, problem, x0, opts};
        auto result = solver.solve();
        iters = static_cast<std::uint32_t>(result.iterations);
        if(!std::isfinite(result.objective_value))
            std::println("WARNING: kraft_slsqp<6> synthetic 6-DoF diverged, f={:.6e}",
                         result.objective_value);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint64_t total_ls_calls = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nablapp::basic_solver solver{nablapp::kraft_slsqp_policy<6>{}, problem, x0, opts};
        auto result = solver.solve();
        fval = result.objective_value;
        iters = static_cast<std::uint32_t>(result.iterations);
        total_ls_calls += solver.state().line_search_calls;
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double per_solve = total_us / reps;
    double per_step = total_us / (reps * iters);
    double ls_per_step = static_cast<double>(total_ls_calls) / (reps * iters);
    return {per_solve, fval, iters, per_step, ls_per_step};
}

timing bench_nlopt_6dof(std::uint32_t reps)
{
    std::uint32_t evals = 0;
    {
        nlopt::opt opt(nlopt::LD_SLSQP, 6);
        opt.set_min_objective(nlopt_synthetic_6dof_objective, nullptr);
        opt.set_lower_bounds(std::vector<double>(6, -1.0e6));
        opt.set_upper_bounds(std::vector<double>(6, 1.0e6));
        opt.add_equality_constraint(nlopt_synthetic_6dof_eq0, nullptr, 1e-10);
        opt.add_equality_constraint(nlopt_synthetic_6dof_eq1, nullptr, 1e-10);
        opt.set_maxeval(200);
        opt.set_ftol_rel(1e-10);
        opt.set_xtol_rel(1e-10);
        std::vector<double> x(6, 0.0);
        double fval;
        opt.optimize(x, fval);
        evals = static_cast<std::uint32_t>(opt.get_numevals());
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nlopt::opt opt(nlopt::LD_SLSQP, 6);
        opt.set_min_objective(nlopt_synthetic_6dof_objective, nullptr);
        opt.set_lower_bounds(std::vector<double>(6, -1.0e6));
        opt.set_upper_bounds(std::vector<double>(6, 1.0e6));
        opt.add_equality_constraint(nlopt_synthetic_6dof_eq0, nullptr, 1e-10);
        opt.add_equality_constraint(nlopt_synthetic_6dof_eq1, nullptr, 1e-10);
        opt.set_maxeval(200);
        opt.set_ftol_rel(1e-10);
        opt.set_xtol_rel(1e-10);
        std::vector<double> x(6, 0.0);
        opt.optimize(x, fval);
        evals = static_cast<std::uint32_t>(opt.get_numevals());
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double per_solve = total_us / reps;
    double per_step = total_us / (reps * evals);
    return {per_solve, fval, evals, per_step, 0.0};
}

// Pretty-print helper: the four default_convergence criteria in order.
// Size is hard-coded to 4 to match default_convergence; templating this
// helper would bloat the bench for no runtime gain.
void print_last_check_results(
    const char* label,
    const std::array<std::optional<nablapp::solver_status>, 4>& results)
{
    constexpr std::array<const char*, 4> criterion_names{
        "gradient_tolerance",
        "objective_tolerance",
        "step_tolerance",
        "stall_tolerance"
    };
    std::println("  {} last_check_results:", label);
    for(std::size_t i = 0; i < results.size(); ++i)
    {
        if(results[i])
            std::println("    [{}] {:<20s} fired -> status code {}",
                         i, criterion_names[i],
                         static_cast<int>(*results[i]));
        else
            std::println("    [{}] {:<20s} -", i, criterion_names[i]);
    }
}

}

int main()
{
    constexpr std::uint32_t reps = 10000;

    std::println("HS071 (n=4, m_eq=1, m_ineq=1), {} repetitions each\n", reps);

    auto kraft = bench_nablapp(reps);
    auto kraft_fixed = bench_nablapp_fixed(reps);
    auto nw = bench_nablapp_nw_sqp(reps);
    auto nl = bench_nlopt(reps);

    std::println("  {:>14s}  {:>12s}  {:>12s}  {:>10s}  {:>12s}",
                 "solver", "solve (us)", "step (us)", "iters", "objective");
    std::println("  {:>14s}  {:12.2f}  {:12.2f}  {:10d}  {:.6e}",
                 "kraft_slsqp<-1>", kraft.wall_us, kraft.per_step_us, kraft.evals, kraft.objective);
    std::println("  {:>14s}  {:12.2f}  {:12.2f}  {:10d}  {:.6e}",
                 "kraft_slsqp<4>", kraft_fixed.wall_us, kraft_fixed.per_step_us, kraft_fixed.evals, kraft_fixed.objective);
    std::println("  {:>14s}  {:12.2f}  {:12.2f}  {:10d}  {:.6e}",
                 "nw_sqp", nw.wall_us, nw.per_step_us, nw.evals, nw.objective);
    std::println("  {:>14s}  {:12.2f}  {:12.2f}  {:10d}  {:.6e}",
                 "nlopt", nl.wall_us, nl.per_step_us, nl.evals, nl.objective);
    std::println("\n  per-solve ratio kraft_slsqp<-1>/nlopt: {:.2f}x", kraft.wall_us / nl.wall_us);
    std::println("  per-step  ratio kraft_slsqp<-1>/nlopt: {:.2f}x", kraft.per_step_us / nl.per_step_us);
    std::println("  per-solve ratio kraft_slsqp<4>/nlopt:  {:.2f}x", kraft_fixed.wall_us / nl.wall_us);
    std::println("  per-step  ratio kraft_slsqp<4>/nlopt:  {:.2f}x", kraft_fixed.per_step_us / nl.per_step_us);
    std::println("  per-step  kraft_slsqp<-1>/<4>:         {:.2f}x  (dynamic overhead on n=4)",
                 kraft.per_step_us / kraft_fixed.per_step_us);
    std::println("  per-solve ratio nw_sqp/nlopt:          {:.2f}x", nw.wall_us / nl.wall_us);
    std::println("  per-step  ratio nw_sqp/nlopt:          {:.2f}x", nw.per_step_us / nl.per_step_us);

    std::println("\n  kraft_slsqp<-1> phi_ls calls per step: {:.3f}",
                 kraft.line_search_calls_per_step);
    std::println("  kraft_slsqp<4>  phi_ls calls per step: {:.3f}",
                 kraft_fixed.line_search_calls_per_step);
    std::println("  (average number of merit-function evaluations per kraft_slsqp_policy::step()");
    std::println("   invocation, averaged over {} reps x {} iters. Armijo success on first try = 1.0;",
                 reps, kraft.evals);
    std::println("   2.0 means one backtrack on average; 3.0 means two backtracks.)");

    // HS071 per-criterion convergence telemetry. One extra solve per
    // variant outside the timing loop so it never pollutes the per-step
    // measurement. Reach path: solver.convergence().last_check_results().
    std::println("");
    {
        hs071 problem;
        Eigen::VectorXd x0{{1.0, 5.0, 5.0, 1.0}};
        nablapp::solver_options opts;
        opts.max_iterations = 200;
        opts.set_gradient_threshold(1e-8);
        opts.set_objective_threshold(1e-10);
        opts.set_step_threshold(1e-10);
        nablapp::basic_solver solver{nablapp::kraft_slsqp_policy{}, problem, x0, opts};
        (void)solver.solve();
        print_last_check_results("kraft_slsqp<-1> HS071",
                                 solver.convergence().last_check_results());
    }
    {
        hs071_fixed problem;
        Eigen::Vector<double, 4> x0{1.0, 5.0, 5.0, 1.0};
        nablapp::solver_options opts;
        opts.max_iterations = 200;
        opts.set_gradient_threshold(1e-8);
        opts.set_objective_threshold(1e-10);
        opts.set_step_threshold(1e-10);
        nablapp::basic_solver solver{nablapp::kraft_slsqp_policy<4>{}, problem, x0, opts};
        (void)solver.solve();
        print_last_check_results("kraft_slsqp<4>  HS071",
                                 solver.convergence().last_check_results());
    }

    // Synthetic 6-DoF benchmarks.
    std::println("\nSynthetic 6-DoF (n=6, m_eq=2, m_ineq=0, kappa(M)~1e4), {} repetitions each\n",
                 reps);
    auto kraft6 = bench_nablapp_6dof(reps);
    auto kraft6f = bench_nablapp_6dof_fixed(reps);
    auto nl6 = bench_nlopt_6dof(reps);

    std::println("  {:>14s}  {:>12s}  {:>12s}  {:>10s}  {:>12s}",
                 "solver", "solve (us)", "step (us)", "iters", "objective");
    std::println("  {:>14s}  {:12.2f}  {:12.2f}  {:10d}  {:.6e}",
                 "kraft_slsqp<-1>", kraft6.wall_us, kraft6.per_step_us,
                 kraft6.evals, kraft6.objective);
    std::println("  {:>14s}  {:12.2f}  {:12.2f}  {:10d}  {:.6e}",
                 "kraft_slsqp<6>", kraft6f.wall_us, kraft6f.per_step_us,
                 kraft6f.evals, kraft6f.objective);
    std::println("  {:>14s}  {:12.2f}  {:12.2f}  {:10d}  {:.6e}",
                 "nlopt", nl6.wall_us, nl6.per_step_us, nl6.evals, nl6.objective);
    std::println("\n  per-solve ratio kraft_slsqp<-1>/nlopt: {:.2f}x",
                 kraft6.wall_us / nl6.wall_us);
    std::println("  per-step  ratio kraft_slsqp<-1>/nlopt: {:.2f}x",
                 kraft6.per_step_us / nl6.per_step_us);
    std::println("  per-solve ratio kraft_slsqp<6>/nlopt:  {:.2f}x",
                 kraft6f.wall_us / nl6.wall_us);
    std::println("  per-step  ratio kraft_slsqp<6>/nlopt:  {:.2f}x",
                 kraft6f.per_step_us / nl6.per_step_us);
    std::println("  per-step  kraft_slsqp<-1>/<6>:         {:.2f}x  (dynamic overhead on n=6)",
                 kraft6.per_step_us / kraft6f.per_step_us);
    std::println("\n  kraft_slsqp<-1> 6-DoF phi_ls calls per step: {:.3f}",
                 kraft6.line_search_calls_per_step);
    std::println("  kraft_slsqp<6>  6-DoF phi_ls calls per step: {:.3f}",
                 kraft6f.line_search_calls_per_step);

    std::println("");
    {
        synthetic_6dof problem;
        Eigen::VectorXd x0 = Eigen::VectorXd::Zero(6);
        nablapp::solver_options opts;
        opts.max_iterations = 200;
        opts.set_gradient_threshold(1e-8);
        opts.set_objective_threshold(1e-10);
        opts.set_step_threshold(1e-10);
        nablapp::basic_solver solver{nablapp::kraft_slsqp_policy{}, problem, x0, opts};
        (void)solver.solve();
        print_last_check_results("kraft_slsqp<-1> synthetic 6-DoF",
                                 solver.convergence().last_check_results());
    }
    {
        synthetic_6dof_fixed problem;
        Eigen::Vector<double, 6> x0 = Eigen::Vector<double, 6>::Zero();
        nablapp::solver_options opts;
        opts.max_iterations = 200;
        opts.set_gradient_threshold(1e-8);
        opts.set_objective_threshold(1e-10);
        opts.set_step_threshold(1e-10);
        nablapp::basic_solver solver{nablapp::kraft_slsqp_policy<6>{}, problem, x0, opts};
        (void)solver.solve();
        print_last_check_results("kraft_slsqp<6>  synthetic 6-DoF",
                                 solver.convergence().last_check_results());
    }

    std::println("\nNow profile with:");
    std::println("  perf record -F 99999 -g -- ./micro_kraft_slsqp");
    std::println("  perf report --stdio --percent-limit=1.0");
}
