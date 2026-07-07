// Micro-benchmark: argmin ISRES variants vs NLopt GN_ISRES.
//
// Multi-variant comparison harness covering all four ISRES policies that
// ship in the project:
//   - isres_policy<>                    (production alias; currently
//                                        nlopt_faithful_policy)
//   - alternative::isres::nlopt_faithful_policy<>     (NLopt isres.c form)
//   - alternative::isres::original_argmin_policy<>   (frozen baseline,
//                                                      pre-rewrite)
//   - alternative::isres::runarsson_yao_paper_policy<>(paper-form DE)
// plus the NLopt GN_ISRES control.
//
// Per-cell statistics: reps x seeds trials per (variant, problem). Each
// row reports mean wall-time, stddev, evals, objective, iterations, and
// a z-score against the NLopt control row's wall-time distribution. The
// z-score column resolves the noise floor on small differences between
// stochastic runs (per the micro_kraft_slsqp 10000-rep precedent).
//
// Test set:
//   HS006, HS024, HS035, HS076 (Hock-Schittkowski subset), Rastrigin 2D,
//   Rastrigin 5D, Schwefel 2D, Schwefel 5D, plus a bounds-degenerate
//   problem (one dimension with near-zero feasible interval) that
//   stresses the bounded resample retry budget.
//
// References:
//   Runarsson, T. P., and Yao, X. (2005), "Search Biases in Constrained
//     Evolutionary Optimization," IEEE Trans. Systems, Man, and
//     Cybernetics, Part C: Applications and Reviews, 35(2):233-243.
//   Kochenderfer, M. J., and Wheeler, T. A., "Algorithms for
//     Optimization", 2e, MIT Press 2019, Section 8.6 (Evolution
//     Strategies).
//   NLopt 2.10.0 isres.c (Steven G. Johnson 2009).

#include "argmin/solver/isres_policy.h"
#include "argmin/solver/alternative/isres/nlopt_faithful_policy.h"
#include "argmin/solver/alternative/isres/original_argmin_policy.h"
#include "argmin/solver/alternative/isres/runarsson_yao_paper_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/test_functions/rastrigin.h"
#include "argmin/test_functions/schwefel.h"

#include <Eigen/Core>

#include <nlopt.hpp>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numbers>
#include <print>
#include <string_view>
#include <vector>

namespace
{

struct timing
{
    double mean_wall_us;
    double stddev_wall_us;
    double objective;
    std::uint32_t evals;
    std::uint32_t iterations;
};

// ---------------------------------------------------------------------
// Test-problem fixtures (constrained_values + bound_constrained).
// ---------------------------------------------------------------------

// Rastrigin in n dimensions with trivial constraints + bounds, satisfying
// ISRES's constrained_values + bound_constrained preconditions.
struct rastrigin_box
{
    int n;

    static constexpr int problem_dimension = argmin::dynamic_dimension;
    static constexpr argmin::problem_class pclass =
        argmin::problem_class::global | argmin::problem_class::bound_constrained;

    [[nodiscard]] int dimension() const { return n; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] double value(const Eigen::VectorXd& x) const
    {
        constexpr double two_pi = 2.0 * std::numbers::pi;
        double f = 10.0 * static_cast<double>(n);
        for(int i = 0; i < n; ++i)
            f += x[i] * x[i] - 10.0 * std::cos(two_pi * x[i]);
        return f;
    }

    void constraints(const Eigen::VectorXd&, Eigen::VectorXd& c) const
    {
        c.resize(0);
    }

    [[nodiscard]] Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd::Constant(n, -5.12);
    }

    [[nodiscard]] Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd::Constant(n, 5.12);
    }

    [[nodiscard]] Eigen::VectorXd initial_point() const
    {
        return Eigen::VectorXd::Constant(n, 2.5);
    }
};

struct schwefel_box
{
    int n;

    static constexpr int problem_dimension = argmin::dynamic_dimension;
    static constexpr argmin::problem_class pclass =
        argmin::problem_class::global | argmin::problem_class::bound_constrained;

    [[nodiscard]] int dimension() const { return n; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] double value(const Eigen::VectorXd& x) const
    {
        double s = 0.0;
        for(int i = 0; i < n; ++i)
            s += x[i] * std::sin(std::sqrt(std::abs(x[i])));
        return 418.9829 * static_cast<double>(n) - s;
    }

    void constraints(const Eigen::VectorXd&, Eigen::VectorXd& c) const
    {
        c.resize(0);
    }

    [[nodiscard]] Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd::Constant(n, -500.0);
    }

    [[nodiscard]] Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd::Constant(n, 500.0);
    }

    [[nodiscard]] Eigen::VectorXd initial_point() const
    {
        return Eigen::VectorXd::Constant(n, -200.0);
    }
};

// Hock-Schittkowski wrappers: replace semi-bounded / unbounded boxes with
// finite ranges large enough that the canonical optima lie inside, so
// ISRES can sample uniformly. Rationale: the upstream HS006/HS024/HS035/
// HS076 fixtures use +/-inf upper bounds; ISRES requires finite boxes for
// uniform initial-population sampling.

// HS006: 2D, 1 equality constraint c[0] = 10*(x1 - x0^2) = 0. Optimum at
// (1, 1) with f* = 0. Box [-10, 10] is comfortable for ISRES's uniform
// init.
struct hs006_box
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;
    static constexpr argmin::problem_class pclass =
        argmin::problem_class::equality | argmin::problem_class::bound_constrained;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 1; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] double value(const Eigen::VectorXd& x) const
    {
        const double t = 1.0 - x[0];
        return t * t;
    }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = 10.0 * (x[1] - x[0] * x[0]);
    }

    [[nodiscard]] Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd::Constant(2, -10.0);
    }

    [[nodiscard]] Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd::Constant(2, 10.0);
    }

    [[nodiscard]] Eigen::VectorXd initial_point() const
    {
        return Eigen::VectorXd{{-1.2, 1.0}};
    }
};

// HS024: 2D, 3 inequality + box [0, inf]^2. Optimum at (3, sqrt(3)) with
// f* = -1. Box-cap upper bound at 10 to give ISRES a finite sampling
// region; the optimum sits well inside.
struct hs024_box
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;
    static constexpr argmin::problem_class pclass =
        argmin::problem_class::inequality | argmin::problem_class::bound_constrained;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 3; }

    [[nodiscard]] double value(const Eigen::VectorXd& x) const
    {
        const double k = 1.0 / (27.0 * std::sqrt(3.0));
        const double t = (x[0] - 3.0) * (x[0] - 3.0) - 9.0;
        return k * t * x[1] * x[1] * x[1];
    }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        const double s3 = std::sqrt(3.0);
        c.resize(3);
        c[0] = x[0] / s3 - x[1];
        c[1] = x[0] + s3 * x[1];
        c[2] = 6.0 - x[0] - s3 * x[1];
    }

    [[nodiscard]] Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd::Zero(2);
    }

    [[nodiscard]] Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd::Constant(2, 10.0);
    }

    [[nodiscard]] Eigen::VectorXd initial_point() const
    {
        return Eigen::VectorXd{{1.0, 0.5}};
    }
};

// HS035: 3D, 1 inequality, x >= 0. Optimum at (4/3, 7/9, 4/9) with
// f* = 1/9. Box [0, 10]^3.
struct hs035_box
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;
    static constexpr argmin::problem_class pclass =
        argmin::problem_class::inequality | argmin::problem_class::bound_constrained;

    [[nodiscard]] int dimension() const { return 3; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 1; }

    [[nodiscard]] double value(const Eigen::VectorXd& x) const
    {
        return 9.0 - 8.0 * x[0] - 6.0 * x[1] - 4.0 * x[2]
             + 2.0 * x[0] * x[0] + 2.0 * x[1] * x[1] + x[2] * x[2]
             + 2.0 * x[0] * x[1] + 2.0 * x[0] * x[2];
    }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = 3.0 - (x[0] + x[1] + 2.0 * x[2]);
    }

    [[nodiscard]] Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd::Zero(3);
    }

    [[nodiscard]] Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd::Constant(3, 10.0);
    }

    [[nodiscard]] Eigen::VectorXd initial_point() const
    {
        return Eigen::VectorXd::Constant(3, 0.5);
    }
};

// HS076: 4D, 3 inequality, x >= 0. Optimum near
// (0.2727, 2.0909, -0.2603, 0.5455) with f* = -4.6818... Box [0, 5]^4
// (HS076 has lb = 0; upper-cap at 5 is well outside the optimum's
// support).
struct hs076_box
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;
    static constexpr argmin::problem_class pclass =
        argmin::problem_class::inequality | argmin::problem_class::bound_constrained;

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

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(3);
        c[0] = 5.0 - (x[0] + 2.0 * x[1] + x[2] + x[3]);
        c[1] = 4.0 - (3.0 * x[0] + x[1] + 2.0 * x[2] - x[3]);
        c[2] = x[1] + 4.0 * x[2] - 1.5;
    }

    [[nodiscard]] Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd::Zero(4);
    }

    [[nodiscard]] Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd::Constant(4, 5.0);
    }

    [[nodiscard]] Eigen::VectorXd initial_point() const
    {
        return Eigen::VectorXd::Constant(4, 0.5);
    }
};

// Bounds-degenerate problem: 2D quadratic with the second dimension
// pinned to a near-zero-width interval. This stresses the I11 bounded
// resample retry budget; under a wide-enough mutation step the operator
// may exhaust the budget and fall back to clamp.
struct bounds_degenerate
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;
    static constexpr argmin::problem_class pclass =
        argmin::problem_class::global | argmin::problem_class::bound_constrained;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] double value(const Eigen::VectorXd& x) const
    {
        return (x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5);
    }

    void constraints(const Eigen::VectorXd&, Eigen::VectorXd& c) const
    {
        c.resize(0);
    }

    [[nodiscard]] Eigen::VectorXd lower_bounds() const
    {
        Eigen::VectorXd lb(2);
        lb << -1.0, 0.5;
        return lb;
    }

    [[nodiscard]] Eigen::VectorXd upper_bounds() const
    {
        Eigen::VectorXd ub(2);
        ub << 1.0, 0.5 + 1e-10;
        return ub;
    }

    [[nodiscard]] Eigen::VectorXd initial_point() const
    {
        return Eigen::VectorXd{{0.0, 0.5}};
    }
};

// ---------------------------------------------------------------------
// NLopt callback dispatch.
// ---------------------------------------------------------------------

template <typename Problem>
double nlopt_obj_callback(unsigned n, const double* x, double*, void* data)
{
    const Problem& p = *static_cast<const Problem*>(data);
    Eigen::VectorXd xv(static_cast<int>(n));
    for(unsigned i = 0; i < n; ++i)
        xv[static_cast<int>(i)] = x[i];
    return p.value(xv);
}

template <typename Problem>
void nlopt_eq_mcallback(unsigned m, double* result,
                        unsigned n, const double* x, double*, void* data)
{
    const Problem& p = *static_cast<const Problem*>(data);
    Eigen::VectorXd xv(static_cast<int>(n));
    for(unsigned i = 0; i < n; ++i)
        xv[static_cast<int>(i)] = x[i];
    Eigen::VectorXd c(static_cast<int>(p.num_equality() + p.num_inequality()));
    p.constraints(xv, c);
    // c[0..num_equality()) are equality terms (caller convention c == 0).
    for(unsigned i = 0; i < m; ++i)
        result[i] = c[static_cast<int>(i)];
}

template <typename Problem>
void nlopt_ineq_mcallback(unsigned m, double* result,
                          unsigned n, const double* x, double*, void* data)
{
    const Problem& p = *static_cast<const Problem*>(data);
    Eigen::VectorXd xv(static_cast<int>(n));
    for(unsigned i = 0; i < n; ++i)
        xv[static_cast<int>(i)] = x[i];
    Eigen::VectorXd c(static_cast<int>(p.num_equality() + p.num_inequality()));
    p.constraints(xv, c);
    // argmin convention: c_ineq[i] >= 0 feasible. NLopt convention:
    // ineq result[i] <= 0 feasible. Negate to translate.
    const int n_eq = p.num_equality();
    for(unsigned i = 0; i < m; ++i)
        result[i] = -c[n_eq + static_cast<int>(i)];
}

template <typename Problem>
timing bench_nlopt(const Problem& problem, std::uint32_t reps,
                   std::uint32_t seed_start, std::uint32_t seed_count)
{
    const auto lb = problem.lower_bounds();
    const auto ub = problem.upper_bounds();
    const auto x0 = problem.initial_point();
    const int n = problem.dimension();

    // Warmup.
    {
        nlopt::opt opt(nlopt::GN_ISRES, static_cast<unsigned>(n));
        opt.set_min_objective(nlopt_obj_callback<Problem>,
                              const_cast<Problem*>(&problem));
        opt.set_lower_bounds(std::vector<double>(lb.data(), lb.data() + n));
        opt.set_upper_bounds(std::vector<double>(ub.data(), ub.data() + n));
        opt.set_maxeval(50000);
        opt.set_ftol_rel(1e-12);
        opt.set_xtol_rel(1e-12);
        opt.set_population(0);
        if(problem.num_equality() > 0)
            opt.add_equality_mconstraint(
                nlopt_eq_mcallback<Problem>,
                const_cast<Problem*>(&problem),
                std::vector<double>(problem.num_equality(), 1e-6));
        if(problem.num_inequality() > 0)
            opt.add_inequality_mconstraint(
                nlopt_ineq_mcallback<Problem>,
                const_cast<Problem*>(&problem),
                std::vector<double>(problem.num_inequality(), 1e-6));
        std::vector<double> x(x0.data(), x0.data() + n);
        double fval;
        try { opt.optimize(x, fval); } catch(...) {}
    }

    std::vector<double> samples;
    samples.reserve(reps * seed_count);
    double last_fval = 0.0;
    std::uint32_t last_evals = 0;

    for(std::uint32_t s = 0; s < seed_count; ++s)
    {
        const std::uint32_t seed_value = seed_start + s;
        for(std::uint32_t r = 0; r < reps; ++r)
        {
            nlopt::opt opt(nlopt::GN_ISRES, static_cast<unsigned>(n));
            opt.set_min_objective(nlopt_obj_callback<Problem>,
                                  const_cast<Problem*>(&problem));
            opt.set_lower_bounds(std::vector<double>(lb.data(), lb.data() + n));
            opt.set_upper_bounds(std::vector<double>(ub.data(), ub.data() + n));
            opt.set_maxeval(50000);
            opt.set_ftol_rel(1e-12);
            opt.set_xtol_rel(1e-12);
            opt.set_population(0);
            nlopt_srand(seed_value);
            if(problem.num_equality() > 0)
                opt.add_equality_mconstraint(
                    nlopt_eq_mcallback<Problem>,
                    const_cast<Problem*>(&problem),
                    std::vector<double>(problem.num_equality(), 1e-6));
            if(problem.num_inequality() > 0)
                opt.add_inequality_mconstraint(
                    nlopt_ineq_mcallback<Problem>,
                    const_cast<Problem*>(&problem),
                    std::vector<double>(problem.num_inequality(), 1e-6));
            std::vector<double> x(x0.data(), x0.data() + n);

            const auto t0 = std::chrono::high_resolution_clock::now();
            try { opt.optimize(x, last_fval); } catch(...) {}
            const auto t1 = std::chrono::high_resolution_clock::now();

            samples.push_back(
                std::chrono::duration<double, std::micro>(t1 - t0).count());
            last_evals = static_cast<std::uint32_t>(opt.get_numevals());
        }
    }

    const double sum = std::reduce(samples.begin(), samples.end(), 0.0);
    const double mean = sum / static_cast<double>(samples.size());
    double var = 0.0;
    for(const double v : samples) var += (v - mean) * (v - mean);
    var /= static_cast<double>(samples.size());

    return timing{mean, std::sqrt(var), last_fval, last_evals, 0};
}

// ---------------------------------------------------------------------
// Templated argmin bench harness.
// ---------------------------------------------------------------------

template <typename Policy, typename Problem>
timing bench_argmin(const Problem& problem, std::uint32_t reps,
                     std::uint32_t seed_start, std::uint32_t seed_count)
{
    const auto x0 = problem.initial_point();
    argmin::solver_options opts;
    opts.max_iterations = 50000;
    opts.set_objective_threshold(1e-6);
    opts.set_step_threshold(1e-12);
    // Wire the stall criterion so production-alias / nlopt_faithful /
    // runarsson_yao_paper terminate when sigma collapses but the
    // ftol_reached predicate is preempted by a flat-objective regime.
    // 1e-9 default mirrors the production isres_policy stall semantics
    // and keeps the bench tractable on the 50000-iter ceiling cells.
    opts.set_stall_threshold(1e-9);

    // Warmup.
    {
        Policy policy;
        policy.options.seed = seed_start;
        argmin::step_budget_solver solver{policy, problem, x0, opts};
        solver.solve();
    }

    std::vector<double> samples;
    samples.reserve(reps * seed_count);
    double last_fval = 0.0;
    std::uint32_t last_iters = 0;

    for(std::uint32_t s = 0; s < seed_count; ++s)
    {
        const std::uint64_t seed_value = static_cast<std::uint64_t>(seed_start + s);
        Policy policy;
        policy.options.seed = seed_value;
        for(std::uint32_t r = 0; r < reps; ++r)
        {
            const auto t0 = std::chrono::high_resolution_clock::now();
            argmin::step_budget_solver solver{policy, problem, x0, opts};
            const auto result = solver.solve();
            const auto t1 = std::chrono::high_resolution_clock::now();

            last_fval = result.objective_value;
            last_iters = result.iterations;
            samples.push_back(
                std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
    }

    const double sum = std::reduce(samples.begin(), samples.end(), 0.0);
    const double mean = sum / static_cast<double>(samples.size());
    double var = 0.0;
    for(const double v : samples) var += (v - mean) * (v - mean);
    var /= static_cast<double>(samples.size());

    // Eval count for evolution strategies is reps-dependent; report
    // last iteration's eval count as a proxy (lambda * iterations).
    return timing{mean, std::sqrt(var), last_fval, last_iters, last_iters};
}

void print_header()
{
    std::println("  {:>26s}  {:>10s}  {:>10s}  {:>8s}  {:>14s}  {:>8s}  {:>8s}",
                 "solver", "wall_us", "stddev_us", "evals",
                 "objective", "iters", "z_us");
}

void print_row(std::string_view solver, const timing& t,
               const timing& nlopt_control)
{
    const double sd = std::max(nlopt_control.stddev_wall_us, 1e-12);
    const double z = (t.mean_wall_us - nlopt_control.mean_wall_us) / sd;
    std::println("  {:>26s}  {:10.2f}  {:10.2f}  {:8d}  {:14.6e}  {:8d}  {:+8.2f}",
                 solver,
                 t.mean_wall_us, t.stddev_wall_us,
                 t.evals, t.objective, t.iterations, z);
}

bool parse_uint_arg(std::string_view raw,
                    std::string_view key,
                    std::uint32_t& out)
{
    if(!raw.starts_with(key)) return false;
    auto rest = raw.substr(key.size());
    if(rest.starts_with('=')) rest = rest.substr(1);
    std::uint32_t v = 0;
    auto [_, ec] = std::from_chars(rest.data(),
                                   rest.data() + rest.size(), v);
    if(ec != std::errc{}) return false;
    out = v;
    return true;
}

}

int main(int argc, char** argv)
{
    std::uint32_t reps = 10;
    std::uint32_t seed_count = 10;
    std::uint32_t seed_start = 42;

    for(int i = 1; i < argc; ++i)
    {
        std::string_view arg{argv[i]};
        if((arg == "--reps" || arg == "-r") && i + 1 < argc)
        {
            std::from_chars(argv[i + 1],
                            argv[i + 1] + std::strlen(argv[i + 1]),
                            reps);
            ++i;
            continue;
        }
        if(parse_uint_arg(arg, "--reps", reps)) continue;
        if((arg == "--seed-count" || arg == "-s") && i + 1 < argc)
        {
            std::from_chars(argv[i + 1],
                            argv[i + 1] + std::strlen(argv[i + 1]),
                            seed_count);
            ++i;
            continue;
        }
        if(parse_uint_arg(arg, "--seed-count", seed_count)) continue;
        if((arg == "--seed-start") && i + 1 < argc)
        {
            std::from_chars(argv[i + 1],
                            argv[i + 1] + std::strlen(argv[i + 1]),
                            seed_start);
            ++i;
            continue;
        }
        if(parse_uint_arg(arg, "--seed-start", seed_start)) continue;
    }

    std::println("ISRES multi-variant micro-benchmark");
    std::println("  reps           = {}", reps);
    std::println("  seed_count     = {}", seed_count);
    std::println("  seed_start     = {}", seed_start);
    std::println("  trials/cell    = {} (reps x seeds)", reps * seed_count);
    std::println("");
    std::println("  Variants per problem (5 rows):");
    std::println("    1. isres (production alias) -> alternative::isres::nlopt_faithful_policy");
    std::println("    2. nlopt_faithful           -> alternative::isres::nlopt_faithful_policy");
    std::println("    3. original_argmin         -> alternative::isres::original_argmin_policy (frozen baseline)");
    std::println("    4. runarsson_yao_paper      -> alternative::isres::runarsson_yao_paper_policy");
    std::println("    5. nlopt_isres              -> NLopt 2.10.0 GN_ISRES (control; z-score reference)");

    auto print_problem = [&]<typename Problem>(std::string_view name,
                                               const Problem& problem)
    {
        std::println("\n=== {} ===", name);
        print_header();

        const auto nlopt_t = bench_nlopt(problem, reps, seed_start, seed_count);

        const auto t_alias = bench_argmin<argmin::isres_policy<>>(
            problem, reps, seed_start, seed_count);
        const auto t_faith = bench_argmin<
            argmin::alternative::isres::nlopt_faithful_policy<>>(
                problem, reps, seed_start, seed_count);
        const auto t_orig = bench_argmin<
            argmin::alternative::isres::original_argmin_policy<>>(
                problem, reps, seed_start, seed_count);
        const auto t_paper = bench_argmin<
            argmin::alternative::isres::runarsson_yao_paper_policy<>>(
                problem, reps, seed_start, seed_count);

        print_row("isres (production alias)", t_alias, nlopt_t);
        print_row("nlopt_faithful", t_faith, nlopt_t);
        print_row("original_argmin", t_orig, nlopt_t);
        print_row("runarsson_yao_paper", t_paper, nlopt_t);
        print_row("nlopt_isres", nlopt_t, nlopt_t);
    };

    print_problem("hs006_box (n=2, 1 eq)",
                  hs006_box{});
    print_problem("hs024_box (n=2, 3 ineq, f*=-1)",
                  hs024_box{});
    print_problem("hs035_box (n=3, 1 ineq, f*=1/9)",
                  hs035_box{});
    print_problem("hs076_box (n=4, 3 ineq, f*=-4.6818)",
                  hs076_box{});
    print_problem("rastrigin_2d (global, n=2, f*=0)",
                  rastrigin_box{2});
    print_problem("rastrigin_5d (global, n=5, f*=0)",
                  rastrigin_box{5});
    print_problem("schwefel_2d  (global, n=2, f*~0)",
                  schwefel_box{2});
    print_problem("schwefel_5d  (global, n=5, f*~0)",
                  schwefel_box{5});
    print_problem("bounds_degenerate (n=2, lb[1]==ub[1] within 1e-10)",
                  bounds_degenerate{});
}
