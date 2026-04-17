// Micro-benchmark: nablapp augmented_lagrangian vs NLopt LD_AUGLAG on constrained HS problems.
//
// Augmented Lagrangian wraps an inner L-BFGS-B solver and converts it
// into a constrained optimizer via penalty/multiplier updates. Tested
// on problems with equality and mixed constraints.
//
// Reference: K&W Section 10.9, Algorithm 10.2;
//            N&W Section 17.4, Algorithm 17.4.

#include "nablapp/solver/augmented_lagrangian_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <nlopt.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <print>

namespace
{

struct timing
{
    double wall_us;
    double objective;
    std::uint32_t evals;
};

// Dynamic-dimension HS039 wrapper (equality, n=4).
struct hs039_dynamic
{
    static constexpr int problem_dimension = nablapp::dynamic_dimension;

    [[nodiscard]] int dimension() const { return 4; }
    [[nodiscard]] int num_equality() const { return 2; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] double value(const Eigen::VectorXd& x) const
    {
        return -x[0];
    }

    void gradient(const Eigen::VectorXd&, Eigen::VectorXd& g) const
    {
        g[0] = -1.0; g[1] = 0.0; g[2] = 0.0; g[3] = 0.0;
    }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(2);
        c[0] = x[1] - x[0] * x[0] * x[0] - x[2] * x[2];
        c[1] = x[0] * x[0] - x[1] - x[3] * x[3];
    }

    void constraint_jacobian(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const
    {
        J.resize(2, 4);
        J(0, 0) = -3.0 * x[0] * x[0]; J(0, 1) = 1.0;
        J(0, 2) = -2.0 * x[2];         J(0, 3) = 0.0;
        J(1, 0) = 2.0 * x[0];          J(1, 1) = -1.0;
        J(1, 2) = 0.0;                  J(1, 3) = -2.0 * x[3];
    }

    [[nodiscard]] Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd::Constant(4, -std::numeric_limits<double>::infinity());
    }

    [[nodiscard]] Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd::Constant(4, std::numeric_limits<double>::infinity());
    }

    [[nodiscard]] Eigen::VectorXd initial_point() const
    {
        return Eigen::VectorXd::Constant(4, 2.0);
    }
};

// Dynamic-dimension HS071 wrapper (mixed, n=4).
struct hs071_dynamic
{
    static constexpr int problem_dimension = nablapp::dynamic_dimension;

    [[nodiscard]] int dimension() const { return 4; }
    [[nodiscard]] int num_equality() const { return 1; }
    [[nodiscard]] int num_inequality() const { return 1; }

    [[nodiscard]] double value(const Eigen::VectorXd& x) const
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

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(2);
        c[0] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3] - 40.0;
        c[1] = x[0] * x[1] * x[2] * x[3] - 25.0;
    }

    void constraint_jacobian(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const
    {
        J.resize(2, 4);
        J(0, 0) = 2.0 * x[0]; J(0, 1) = 2.0 * x[1];
        J(0, 2) = 2.0 * x[2]; J(0, 3) = 2.0 * x[3];
        J(1, 0) = x[1] * x[2] * x[3]; J(1, 1) = x[0] * x[2] * x[3];
        J(1, 2) = x[0] * x[1] * x[3]; J(1, 3) = x[0] * x[1] * x[2];
    }

    [[nodiscard]] Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd::Constant(4, 1.0);
    }

    [[nodiscard]] Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd::Constant(4, 5.0);
    }

    [[nodiscard]] Eigen::VectorXd initial_point() const
    {
        return Eigen::VectorXd{{1.0, 5.0, 5.0, 1.0}};
    }
};

// Dynamic-dimension HS076 wrapper (inequality + box, n=4).
struct hs076_dynamic
{
    static constexpr int problem_dimension = nablapp::dynamic_dimension;
    static constexpr nablapp::problem_class pclass =
        nablapp::problem_class::inequality | nablapp::problem_class::bound_constrained;

    [[nodiscard]] int dimension() const { return 4; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 3; }

    [[nodiscard]] double value(const Eigen::VectorXd& x) const
    {
        return x[0] * x[0] + 0.5 * x[1] * x[1]
               + x[2] * x[2] + 0.5 * x[3] * x[3]
               - x[0] * x[2] + x[2] * x[3]
               - x[0] - 3.0 * x[1] + x[2] - x[3];
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        g[0] = 2.0 * x[0] - x[2] - 1.0;
        g[1] = x[1] - 3.0;
        g[2] = 2.0 * x[2] - x[0] + x[3] + 1.0;
        g[3] = x[3] + x[2] - 1.0;
    }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(3);
        c[0] = 5.0 - (x[0] + 2.0 * x[1] + x[2] + x[3]);
        c[1] = 4.0 - (3.0 * x[0] + x[1] + 2.0 * x[2] - x[3]);
        c[2] = x[1] + 4.0 * x[2] - 1.5;
    }

    void constraint_jacobian(const Eigen::VectorXd&, Eigen::MatrixXd& J) const
    {
        J.resize(3, 4);
        J << -1.0, -2.0, -1.0, -1.0,
             -3.0, -1.0, -2.0,  1.0,
              0.0,  1.0,  4.0,  0.0;
    }

    [[nodiscard]] Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd::Zero(4);
    }

    [[nodiscard]] Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd::Constant(4, std::numeric_limits<double>::infinity());
    }

    [[nodiscard]] Eigen::VectorXd initial_point() const
    {
        return Eigen::VectorXd::Constant(4, 0.5);
    }
};

// NLopt callbacks for HS039.
double nlopt_hs039_obj(unsigned, const double* x, double* grad, void*)
{
    if(grad) { grad[0] = -1.0; grad[1] = 0.0; grad[2] = 0.0; grad[3] = 0.0; }
    return -x[0];
}

double nlopt_hs039_eq0(unsigned, const double* x, double* grad, void*)
{
    if(grad)
    {
        grad[0] = -3.0 * x[0] * x[0]; grad[1] = 1.0;
        grad[2] = -2.0 * x[2];         grad[3] = 0.0;
    }
    return x[1] - x[0] * x[0] * x[0] - x[2] * x[2];
}

double nlopt_hs039_eq1(unsigned, const double* x, double* grad, void*)
{
    if(grad)
    {
        grad[0] = 2.0 * x[0]; grad[1] = -1.0;
        grad[2] = 0.0;         grad[3] = -2.0 * x[3];
    }
    return x[0] * x[0] - x[1] - x[3] * x[3];
}

// NLopt callbacks for HS071.
double nlopt_hs071_obj(unsigned, const double* x, double* grad, void*)
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

double nlopt_hs071_eq(unsigned, const double* x, double* grad, void*)
{
    if(grad)
    {
        grad[0] = 2.0 * x[0]; grad[1] = 2.0 * x[1];
        grad[2] = 2.0 * x[2]; grad[3] = 2.0 * x[3];
    }
    return x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3] - 40.0;
}

double nlopt_hs071_ineq(unsigned, const double* x, double* grad, void*)
{
    if(grad)
    {
        grad[0] = -x[1] * x[2] * x[3]; grad[1] = -x[0] * x[2] * x[3];
        grad[2] = -x[0] * x[1] * x[3]; grad[3] = -x[0] * x[1] * x[2];
    }
    return 25.0 - x[0] * x[1] * x[2] * x[3];
}

// NLopt callbacks for HS076.
double nlopt_hs076_obj(unsigned, const double* x, double* grad, void*)
{
    if(grad)
    {
        grad[0] = 2.0 * x[0] - x[2] - 1.0;
        grad[1] = x[1] - 3.0;
        grad[2] = 2.0 * x[2] - x[0] + x[3] + 1.0;
        grad[3] = x[3] + x[2] - 1.0;
    }
    return x[0] * x[0] + 0.5 * x[1] * x[1]
           + x[2] * x[2] + 0.5 * x[3] * x[3]
           - x[0] * x[2] + x[2] * x[3]
           - x[0] - 3.0 * x[1] + x[2] - x[3];
}

double nlopt_hs076_ineq0(unsigned, const double* x, double* grad, void*)
{
    if(grad) { grad[0] = 1.0; grad[1] = 2.0; grad[2] = 1.0; grad[3] = 1.0; }
    return (x[0] + 2.0 * x[1] + x[2] + x[3]) - 5.0;
}

double nlopt_hs076_ineq1(unsigned, const double* x, double* grad, void*)
{
    if(grad) { grad[0] = 3.0; grad[1] = 1.0; grad[2] = 2.0; grad[3] = -1.0; }
    return (3.0 * x[0] + x[1] + 2.0 * x[2] - x[3]) - 4.0;
}

double nlopt_hs076_ineq2(unsigned, const double* x, double* grad, void*)
{
    if(grad) { grad[0] = 0.0; grad[1] = -1.0; grad[2] = -4.0; grad[3] = 0.0; }
    return -(x[1] + 4.0 * x[2] - 1.5);
}

template <typename Problem>
timing bench_nablapp(const Problem& problem, std::uint32_t reps)
{
    auto x0 = problem.initial_point();
    nablapp::solver_options opts;
    opts.max_iterations = 10000;
    opts.set_objective_threshold(1e-8);
    opts.set_step_threshold(1e-12);

    // Warmup.
    {
        nablapp::basic_solver solver{nablapp::augmented_lagrangian_policy<>{}, problem, x0, opts};
        solver.solve();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t iters = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nablapp::basic_solver solver{nablapp::augmented_lagrangian_policy<>{}, problem, x0, opts};
        auto result = solver.solve();
        fval = result.objective_value;
        iters = result.iterations;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, iters};
}

std::optional<timing> bench_nlopt_hs039(std::uint32_t reps)
{
    try
    {
        {
            nlopt::opt local(nlopt::LD_LBFGS, 4);
            local.set_xtol_rel(1e-14);
            nlopt::opt opt(nlopt::LD_AUGLAG, 4);
            opt.set_local_optimizer(local);
            opt.set_min_objective(nlopt_hs039_obj, nullptr);
            opt.add_equality_constraint(nlopt_hs039_eq0, nullptr, 1e-10);
            opt.add_equality_constraint(nlopt_hs039_eq1, nullptr, 1e-10);
            opt.set_maxeval(10000);
            opt.set_ftol_rel(1e-12);
            opt.set_xtol_rel(1e-12);
            std::vector<double> x = {2.0, 2.0, 2.0, 2.0};
            double fval;
            opt.optimize(x, fval);
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        double fval = 0.0;
        std::uint32_t evals = 0;
        for(std::uint32_t r = 0; r < reps; ++r)
        {
            nlopt::opt local(nlopt::LD_LBFGS, 4);
            local.set_xtol_rel(1e-14);
            nlopt::opt opt(nlopt::LD_AUGLAG, 4);
            opt.set_local_optimizer(local);
            opt.set_min_objective(nlopt_hs039_obj, nullptr);
            opt.add_equality_constraint(nlopt_hs039_eq0, nullptr, 1e-10);
            opt.add_equality_constraint(nlopt_hs039_eq1, nullptr, 1e-10);
            opt.set_maxeval(10000);
            opt.set_ftol_rel(1e-12);
            opt.set_xtol_rel(1e-12);
            std::vector<double> x = {2.0, 2.0, 2.0, 2.0};
            opt.optimize(x, fval);
            evals = static_cast<std::uint32_t>(opt.get_numevals());
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
        return timing{us, fval, evals};
    }
    catch(const std::exception& e)
    {
        std::println("  nlopt AUGLAG HS039 failed: {}", e.what());
        return std::nullopt;
    }
}

std::optional<timing> bench_nlopt_hs071(std::uint32_t reps)
{
    try
    {
        {
            nlopt::opt local(nlopt::LD_LBFGS, 4);
            local.set_xtol_rel(1e-14);
            nlopt::opt opt(nlopt::LD_AUGLAG, 4);
            opt.set_local_optimizer(local);
            opt.set_min_objective(nlopt_hs071_obj, nullptr);
            opt.set_lower_bounds({1.0, 1.0, 1.0, 1.0});
            opt.set_upper_bounds({5.0, 5.0, 5.0, 5.0});
            opt.add_equality_constraint(nlopt_hs071_eq, nullptr, 1e-10);
            opt.add_inequality_constraint(nlopt_hs071_ineq, nullptr, 1e-10);
            opt.set_maxeval(10000);
            opt.set_ftol_rel(1e-12);
            opt.set_xtol_rel(1e-12);
            std::vector<double> x = {1.0, 5.0, 5.0, 1.0};
            double fval;
            opt.optimize(x, fval);
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        double fval = 0.0;
        std::uint32_t evals = 0;
        for(std::uint32_t r = 0; r < reps; ++r)
        {
            nlopt::opt local(nlopt::LD_LBFGS, 4);
            local.set_xtol_rel(1e-14);
            nlopt::opt opt(nlopt::LD_AUGLAG, 4);
            opt.set_local_optimizer(local);
            opt.set_min_objective(nlopt_hs071_obj, nullptr);
            opt.set_lower_bounds({1.0, 1.0, 1.0, 1.0});
            opt.set_upper_bounds({5.0, 5.0, 5.0, 5.0});
            opt.add_equality_constraint(nlopt_hs071_eq, nullptr, 1e-10);
            opt.add_inequality_constraint(nlopt_hs071_ineq, nullptr, 1e-10);
            opt.set_maxeval(10000);
            opt.set_ftol_rel(1e-12);
            opt.set_xtol_rel(1e-12);
            std::vector<double> x = {1.0, 5.0, 5.0, 1.0};
            opt.optimize(x, fval);
            evals = static_cast<std::uint32_t>(opt.get_numevals());
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
        return timing{us, fval, evals};
    }
    catch(const std::exception& e)
    {
        std::println("  nlopt AUGLAG HS071 failed: {}", e.what());
        return std::nullopt;
    }
}

std::optional<timing> bench_nlopt_hs076(std::uint32_t reps)
{
    try
    {
        {
            nlopt::opt local(nlopt::LD_LBFGS, 4);
            local.set_xtol_rel(1e-14);
            nlopt::opt opt(nlopt::LD_AUGLAG, 4);
            opt.set_local_optimizer(local);
            opt.set_min_objective(nlopt_hs076_obj, nullptr);
            opt.set_lower_bounds({0.0, 0.0, 0.0, 0.0});
            opt.set_upper_bounds({1e20, 1e20, 1e20, 1e20});
            opt.add_inequality_constraint(nlopt_hs076_ineq0, nullptr, 1e-10);
            opt.add_inequality_constraint(nlopt_hs076_ineq1, nullptr, 1e-10);
            opt.add_inequality_constraint(nlopt_hs076_ineq2, nullptr, 1e-10);
            opt.set_maxeval(10000);
            opt.set_ftol_rel(1e-12);
            opt.set_xtol_rel(1e-12);
            std::vector<double> x = {0.5, 0.5, 0.5, 0.5};
            double fval;
            opt.optimize(x, fval);
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        double fval = 0.0;
        std::uint32_t evals = 0;
        for(std::uint32_t r = 0; r < reps; ++r)
        {
            nlopt::opt local(nlopt::LD_LBFGS, 4);
            local.set_xtol_rel(1e-14);
            nlopt::opt opt(nlopt::LD_AUGLAG, 4);
            opt.set_local_optimizer(local);
            opt.set_min_objective(nlopt_hs076_obj, nullptr);
            opt.set_lower_bounds({0.0, 0.0, 0.0, 0.0});
            opt.set_upper_bounds({1e20, 1e20, 1e20, 1e20});
            opt.add_inequality_constraint(nlopt_hs076_ineq0, nullptr, 1e-10);
            opt.add_inequality_constraint(nlopt_hs076_ineq1, nullptr, 1e-10);
            opt.add_inequality_constraint(nlopt_hs076_ineq2, nullptr, 1e-10);
            opt.set_maxeval(10000);
            opt.set_ftol_rel(1e-12);
            opt.set_xtol_rel(1e-12);
            std::vector<double> x = {0.5, 0.5, 0.5, 0.5};
            opt.optimize(x, fval);
            evals = static_cast<std::uint32_t>(opt.get_numevals());
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
        return timing{us, fval, evals};
    }
    catch(const std::exception& e)
    {
        std::println("  nlopt AUGLAG HS076 failed: {}", e.what());
        return std::nullopt;
    }
}

void print_row(std::string_view solver, const timing& t)
{
    std::println("  {:>12s}  {:10.2f}  {:10d}  {:.6e}", solver, t.wall_us, t.evals, t.objective);
}

// kkt_residual regression probe.
//
// augmented_lagrangian uses outer-loop multiplier estimates
// (s.lambda_eq, s.lambda_ineq) in detail::kkt_residual. With a
// differentiable inner solver the field is populated on every outer
// step. This probe runs the policy on HS039 via step_n() with a
// small budget and confirms the terminal step_result carries a
// populated, non-negative kkt_residual.
//
// Reference: N&W 2e Section 12.3 / eq. 12.34.
bool probe_kkt_residual()
{
    hs039_dynamic problem;
    auto x0 = problem.initial_point();
    nablapp::solver_options opts;
    opts.max_iterations = 30;
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);

    nablapp::basic_solver solver{nablapp::augmented_lagrangian_policy<>{},
                                 problem, x0, opts};

    nablapp::step_result<double> last{};
    for(std::uint32_t i = 0; i < opts.max_iterations; ++i)
    {
        last = solver.step();
        if(last.policy_status)
            break;
    }

    if(!last.kkt_residual.has_value())
    {
        std::println("FAIL: kkt_residual not populated (augmented_lagrangian)");
        return false;
    }
    if(last.kkt_residual.value() < 0.0)
    {
        std::println("FAIL: kkt_residual is negative: {}",
                     last.kkt_residual.value());
        return false;
    }
    std::println("  augmented_lagrangian HS039 kkt_residual: {:.6e} (gradient_norm: {:.6e})",
                 last.kkt_residual.value(), last.gradient_norm);
    return true;
}

// Phase 31.1 regression probe: augmented_lagrangian on HS039 must
// reach f < 1e-5. The policy is not itself a regression driver in
// Phase 31.1 (the Full E-measure rewrite at its kkt call site is
// the D-C3 feasibility_gate removal, not a convergence path), so
// this probe asserts stability rather than a tightened iter bound.
//
// Reference: N&W 2e Definition 12.1.
bool probe_regression_hs039()
{
    nablapp::hs039<> p;
    Eigen::VectorXd x0 = p.initial_point();
    nablapp::solver_options opts;
    opts.max_iterations = 50;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);

    nablapp::basic_solver solver{
        nablapp::augmented_lagrangian_policy<
            nablapp::lbfgsb_policy<nablapp::hs039<>::problem_dimension>,
            nablapp::hs039<>::problem_dimension>{},
        p, x0, opts};
    nablapp::step_result<double> last{};
    for(std::uint32_t i = 0; i < opts.max_iterations; ++i)
    {
        last = solver.step();
        if(last.policy_status)
            break;
    }

    const double kkt = last.kkt_residual.value_or(-1.0);
    const bool ok = last.objective_value < -0.999;  // f* = -1
    if(!ok)
        std::println(stderr,
                     "FAIL: augmented_lagrangian HS039 f={:.6e} kkt={:.6e}",
                     last.objective_value, kkt);
    std::println("  augmented_lagrangian HS039: f={:.6e} kkt={:.6e}",
                 last.objective_value, kkt);
    return ok;
}

}

int main()
{
    constexpr std::uint32_t reps = 200;

    if(!probe_kkt_residual())
        return 1;
    if(!probe_regression_hs039())
        return 1;

    std::println("Augmented Lagrangian micro-benchmark, {} repetitions each\n", reps);
    std::println("  {:>12s}  {:>10s}  {:>10s}  {:>12s}", "solver", "wall (us)", "evals", "objective");

    // HS039
    {
        std::println("\n--- HS039 (equality, n=4, f*=-1) ---");
        auto nab  = bench_nablapp(hs039_dynamic{}, reps);
        auto nlop = bench_nlopt_hs039(reps);
        print_row("nablapp", nab);
        if(nlop)
        {
            print_row("nlopt", *nlop);
            std::println("  ratio nablapp/nlopt: {:.1f}x wall, {:.1f}x evals",
                nab.wall_us / nlop->wall_us, double(nab.evals) / nlop->evals);
        }
    }

    // HS071
    {
        std::println("\n--- HS071 (mixed, n=4, f*~17.014) ---");
        auto nab  = bench_nablapp(hs071_dynamic{}, reps);
        auto nlop = bench_nlopt_hs071(reps);
        print_row("nablapp", nab);
        if(nlop)
        {
            print_row("nlopt", *nlop);
            std::println("  ratio nablapp/nlopt: {:.1f}x wall, {:.1f}x evals",
                nab.wall_us / nlop->wall_us, double(nab.evals) / nlop->evals);
        }
    }

    // HS076
    {
        std::println("\n--- HS076 (inequality + box, n=4, f*=-4.68) ---");
        auto nab  = bench_nablapp(hs076_dynamic{}, reps);
        auto nlop = bench_nlopt_hs076(reps);
        print_row("nablapp", nab);
        if(nlop)
        {
            print_row("nlopt", *nlop);
            std::println("  ratio nablapp/nlopt: {:.1f}x wall, {:.1f}x evals",
                nab.wall_us / nlop->wall_us, double(nab.evals) / nlop->evals);
        }
    }
}
