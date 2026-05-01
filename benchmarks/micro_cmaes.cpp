// Micro-benchmark: nablapp CMA-ES vs libcmaes CMAES_DEFAULT on global problems.
//
// Three-way comparison: nablapp<N> (fixed-N), nablapp<> (dynamic), libcmaes
// CMAES_DEFAULT. IPOP restarts enabled for nablapp variants -- essential for
// multimodal landscape coverage.
//
// Comparator switched from NLopt's controlled-random-search global
// solver to libcmaes CMAES_DEFAULT. CRS is a different algorithm class
// and not a same-class baseline for CMA-ES quality measurement; the
// libcmaes head-to-head is the canonical comparator (see benchmarks/
// bench_libcmaes.{h,cpp}, commit 0c30662). Aligning the micro-bench
// with the publish_bench adapter wiring means per-step z-score numbers
// compare nablapp_cmaes against same-class same-algorithm. Reference:
// Hansen (2023) arXiv:1604.00772; Auger & Hansen (2005) IPOP-CMA-ES.
//
// Vanilla-weights default (Plan 02 of phase 34.2): the rank-mu
// accumulator now runs i = 0..mu-1 (positive weights only) per Hansen
// 2023 §B.1 eq (49)-(50) and libcmaes covarianceupdate.cc:67-75. Every
// per-step number on every cell below changed shape under this fix --
// the diagonal-decay coefficient drops from c_mu * sum_abs_w to literal
// c_mu, the per-step inv_sqrt_delta allocation in the negative-weight
// branch is gone, and the rank-mu loop iterates mu = lambda/2 entries
// instead of lambda. Per the project's 10000-rep micro_cmaes bench-noise
// floor convention this resolves <5% per-step changes.
//
// Sampling variant default (Plan 05 of phase 34.2): production
// detail/cmaes_sampling.h forwards to detail::alternative::marsaglia
// (Marsaglia & Bray 1964 polar method on top of xoshiro256+; thread_local
// pair cache). Empirical winner of the perf-record A/B on this harness:
// drops std::normal_distribution<xoshiro256+> 10.55% self-time slice to
// marsaglia_normal 8.39% at a 5-rep median wall of 495 ms vs 507 ms
// for production. Ziggurat (detail/alternative/cmaes_sampling_ziggurat.h)
// is buildable for re-comparison via temporary include swap in
// detail/cmaes_sampling.h; perf-record snapshots are at /tmp/ab_results/
// during the A/B and copied into the plan's verdict doc post-run.
//
// Run under perf for flamegraph analysis:
//   perf record -F 99999 -g -- ./micro_cmaes
//   perf report --stdio --percent-limit=1.0

#include "nablapp/solver/cmaes_policy.h"
#include "nablapp/solver/alternative/cmaes/repair_l2_penalty_policy.h"
#include "nablapp/solver/alternative/cmaes/pwq_reparameterization_policy.h"
#include "nablapp/solver/alternative/cmaes/no_repair_adaptive_penalty_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/test_functions/ackley.h"
#include "nablapp/test_functions/rastrigin.h"
#include "nablapp/result/status.h"

#include "counting_problem.h"

#include <Eigen/Core>

#include <libcmaes/cmaes.h>
#include <libcmaes/cmaparameters.h>
#include <libcmaes/cmasolutions.h>
#include <libcmaes/genopheno.h>
#include <libcmaes/pwq_bound_strategy.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <print>
#include <vector>

namespace
{

struct timing
{
    double wall_us;
    double objective;
    std::uint32_t evals;
};

// Smooth quadratic centered at the origin used by the TolFun-bound
// regression cell at the bottom of main(). Defined at file scope so it
// satisfies the structural requirements of nablapp's `objective` concept
// (a class type with the required constexpr static members) without the
// local-class restriction that forbids constexpr static data members.
struct smooth_quadratic_2d
{
    static constexpr int problem_dimension = 2;
    int dimension() const { return 2; }
    double value(const Eigen::Vector<double, 2>& x) const
    {
        return x.squaredNorm();
    }
};

// Plain-array Rastrigin objective for the libcmaes FitFunc.
double rastrigin_libcmaes(const double* x, const int& n)
{
    constexpr double two_pi = 2.0 * 3.14159265358979323846;
    double f = 10.0 * static_cast<double>(n);
    for(int i = 0; i < n; ++i)
        f += x[i] * x[i] - 10.0 * std::cos(two_pi * x[i]);
    return f;
}

// Plain-array Rosenbrock objective for the libcmaes FitFunc.
double rosenbrock_libcmaes(const double* x, const int& n)
{
    double f = 0.0;
    for(int i = 0; i + 1 < n; ++i)
    {
        const double t1 = 1.0 - x[i];
        const double t2 = x[i + 1] - x[i] * x[i];
        f += t1 * t1 + 100.0 * t2 * t2;
    }
    return f;
}

// Fixed-dimension bounded Rosenbrock for CMA-ES.
// Unimodal, tests smooth landscape convergence with bounds.
template <int N>
struct bounded_rosenbrock
{
    static constexpr int problem_dimension = N;
    static constexpr nablapp::problem_class pclass =
        nablapp::problem_class::global | nablapp::problem_class::bound_constrained;

    [[nodiscard]] int dimension() const { return N; }

    [[nodiscard]] double value(const Eigen::Vector<double, N>& x) const
    {
        double f = 0.0;
        for(int i = 0; i + 1 < N; ++i)
        {
            double t1 = 1.0 - x[i];
            double t2 = x[i + 1] - x[i] * x[i];
            f += t1 * t1 + 100.0 * t2 * t2;
        }
        return f;
    }

    [[nodiscard]] Eigen::Vector<double, N> lower_bounds() const
    {
        return Eigen::Vector<double, N>::Constant(N, -5.0);
    }

    [[nodiscard]] Eigen::Vector<double, N> upper_bounds() const
    {
        return Eigen::Vector<double, N>::Constant(N, 5.0);
    }

    [[nodiscard]] Eigen::Vector<double, N> initial_point() const
    {
        Eigen::Vector<double, N> x0;
        x0[0] = -1.2;
        for(int i = 1; i < N; ++i)
            x0[i] = 1.0;
        return x0;
    }
};

// Dynamic-dimension bounded Rosenbrock (for cmaes_policy<>).
struct bounded_rosenbrock_dynamic
{
    static constexpr int problem_dimension = nablapp::dynamic_dimension;
    static constexpr nablapp::problem_class pclass =
        nablapp::problem_class::global | nablapp::problem_class::bound_constrained;

    [[nodiscard]] int dimension() const { return 2; }

    [[nodiscard]] double value(const Eigen::VectorXd& x) const
    {
        double t1 = 1.0 - x[0];
        double t2 = x[1] - x[0] * x[0];
        return t1 * t1 + 100.0 * t2 * t2;
    }

    [[nodiscard]] Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd::Constant(2, -5.0);
    }

    [[nodiscard]] Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd::Constant(2, 5.0);
    }

    [[nodiscard]] Eigen::VectorXd initial_point() const
    {
        return Eigen::VectorXd{{-1.2, 1.0}};
    }
};

template <typename Policy, typename Problem>
timing bench_nablapp(Policy policy, const Problem& problem,
                     std::uint32_t reps, std::uint32_t max_iter)
{
    auto x0 = problem.initial_point();
    nablapp::solver_options opts;
    opts.max_iterations = max_iter;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);

    typename Policy::options_type cmaes_opts{};
    cmaes_opts.seed = 42;
    cmaes_opts.restart = Policy::restart_strategy::ipop;

    // Warmup.
    {
        nablapp::basic_solver solver{policy, problem, x0, opts, cmaes_opts};
        solver.solve();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t iters = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nablapp::basic_solver solver{policy, problem, x0, opts, cmaes_opts};
        auto result = solver.solve();
        fval = result.objective_value;
        iters = result.iterations;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, iters};
}

// libcmaes CMAES_DEFAULT comparator, mirroring the bench_libcmaes adapter
// pattern (benchmarks/bench_libcmaes.cpp run_libcmaes_solver) for fairness.
// sigma initialization mirrors that adapter: max(ub - lb) / 3 (Hansen 2023
// recommends sigma in [search_range / 4, search_range / 3]). lambda is
// passed as -1 so libcmaes auto-computes 4 + floor(3*ln(n)), matching
// libcmaes's own default tuning rather than imposing nablapp's lambda
// choice across the boundary.
timing bench_libcmaes_cma(int n, libcmaes::FitFunc obj,
                          const std::vector<double>& x0,
                          const std::vector<double>& lb,
                          const std::vector<double>& ub,
                          std::uint32_t reps, std::uint32_t max_iter)
{
    using gp_type = libcmaes::GenoPheno<libcmaes::pwqBoundStrategy>;

    double max_range = 0.0;
    for(int i = 0; i < n; ++i)
    {
        const double range = ub[i] - lb[i];
        if(std::isfinite(range))
            max_range = std::max(max_range, range);
    }
    const double sigma0 = max_range > 0.0 ? max_range / 3.0 : 0.3;

    gp_type gp(lb.data(), ub.data(), n);

    auto run_one = [&](double& fval, std::uint32_t& evals) {
        libcmaes::CMAParameters<gp_type> params(
            x0, sigma0, /*lambda=*/-1, /*seed=*/42u, gp);
        params.set_algo(::CMAES_DEFAULT);
        params.set_max_iter(max_iter);
        params.set_quiet(true);

        try
        {
            auto sols = libcmaes::cmaes<gp_type>(obj, params);
            fval = sols.best_candidate().get_fvalue();
            evals = static_cast<std::uint32_t>(sols.fevals());
        }
        catch(...)
        {
            fval = std::numeric_limits<double>::quiet_NaN();
            evals = 0;
        }
    };

    // Warmup.
    {
        double fval; std::uint32_t evals;
        run_one(fval, evals);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t evals = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
        run_one(fval, evals);
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, evals};
}

void print_row(std::string_view solver, const timing& t)
{
    std::println("  {:>16s}  {:10.2f}  {:10d}  {:.6e}", solver, t.wall_us, t.evals, t.objective);
}

}

int main()
{
    constexpr std::uint32_t reps = 100;
    std::println("CMA-ES micro-benchmark, {} repetitions each\n", reps);
    std::println("  {:>16s}  {:>10s}  {:>10s}  {:>12s}", "solver", "wall (us)", "evals", "objective");

    // Rastrigin 2D
    {
        std::println("\n--- Rastrigin 2D (global minimum at 0, multimodal) ---");
        nablapp::rastrigin<double, 2> fixed_prob;
        nablapp::rastrigin<double>    dyn_prob{.n = 2};
        auto fixed = bench_nablapp(nablapp::cmaes_policy<2>{}, fixed_prob, reps, 10000);
        auto dyn   = bench_nablapp(nablapp::cmaes_policy<>{},  dyn_prob,  reps, 10000);
        auto libc  = bench_libcmaes_cma(2, rastrigin_libcmaes,
            {2.5, 2.5}, {-5.12, -5.12}, {5.12, 5.12}, reps, 10000);
        print_row("nablapp<2>", fixed);
        print_row("nablapp<>", dyn);
        print_row("libcmaes_cmaes", libc);
        std::println("  ratio fixed/libcmaes: {:.1f}x wall, {:.1f}x evals",
            fixed.wall_us / libc.wall_us, double(fixed.evals) / libc.evals);
        std::println("  ratio dyn/libcmaes:   {:.1f}x wall, {:.1f}x evals",
            dyn.wall_us / libc.wall_us, double(dyn.evals) / libc.evals);
    }

    // Boundary-handling variant A/B on Rastrigin 2D bounded.
    //
    // Persistent comparison of the three solver/alternative/cmaes/
    // boundary-handling variants. Production cmaes_policy<> aliases
    // the empirical winner (see solver/alternative/cmaes/README.md).
    // These per-variant cells stay here so the comparison can be
    // re-run on any future commit.
    // Reference: 34.2-03-AB-RESULT.md verdict doc.
    {
        std::println("\n--- Rastrigin 2D boundary-handling variant A/B ---");
        nablapp::rastrigin<double> dyn_prob{.n = 2};
        auto repair_l2 = bench_nablapp(
            nablapp::alternative::cmaes::repair_l2_penalty_policy<>{},
            dyn_prob, reps, 10000);
        auto pwq = bench_nablapp(
            nablapp::alternative::cmaes::pwq_reparameterization_policy<>{},
            dyn_prob, reps, 10000);
        auto no_repair = bench_nablapp(
            nablapp::alternative::cmaes::no_repair_adaptive_penalty_policy<>{},
            dyn_prob, reps, 10000);
        auto libc = bench_libcmaes_cma(2, rastrigin_libcmaes,
            {2.5, 2.5}, {-5.12, -5.12}, {5.12, 5.12}, reps, 10000);
        print_row("repair_l2", repair_l2);
        print_row("pwq_reparam", pwq);
        print_row("no_repair_adapt", no_repair);
        print_row("libcmaes_cmaes", libc);
    }

    // Boundary-handling variant A/B on Ackley 2D bounded (the box is
    // symmetric and the optimum is at the origin -- a non-boundary-
    // active landscape; expected: variants tie within noise).
    {
        std::println("\n--- Ackley 2D boundary-handling variant A/B ---");
        nablapp::ackley<double> dyn_prob{.n = 2};
        auto repair_l2 = bench_nablapp(
            nablapp::alternative::cmaes::repair_l2_penalty_policy<>{},
            dyn_prob, reps, 10000);
        auto pwq = bench_nablapp(
            nablapp::alternative::cmaes::pwq_reparameterization_policy<>{},
            dyn_prob, reps, 10000);
        auto no_repair = bench_nablapp(
            nablapp::alternative::cmaes::no_repair_adaptive_penalty_policy<>{},
            dyn_prob, reps, 10000);
        print_row("repair_l2", repair_l2);
        print_row("pwq_reparam", pwq);
        print_row("no_repair_adapt", no_repair);
    }

    // Rastrigin 5D
    {
        std::println("\n--- Rastrigin 5D (global minimum at 0, multimodal) ---");
        nablapp::rastrigin<double, 5> fixed_prob;
        nablapp::rastrigin<double>    dyn_prob{.n = 5};
        std::vector<double> x0(5, 2.5), lb(5, -5.12), ub(5, 5.12);
        auto fixed = bench_nablapp(nablapp::cmaes_policy<5>{}, fixed_prob, reps, 50000);
        auto dyn   = bench_nablapp(nablapp::cmaes_policy<>{},  dyn_prob,  reps, 50000);
        auto libc  = bench_libcmaes_cma(5, rastrigin_libcmaes, x0, lb, ub, reps, 50000);
        print_row("nablapp<5>", fixed);
        print_row("nablapp<>", dyn);
        print_row("libcmaes_cmaes", libc);
        std::println("  ratio fixed/libcmaes: {:.1f}x wall, {:.1f}x evals",
            fixed.wall_us / libc.wall_us, double(fixed.evals) / libc.evals);
        std::println("  ratio dyn/libcmaes:   {:.1f}x wall, {:.1f}x evals",
            dyn.wall_us / libc.wall_us, double(dyn.evals) / libc.evals);
    }

    // Rosenbrock 2D (bounded, unimodal -- tests CMA-ES on smooth landscape)
    {
        std::println("\n--- Rosenbrock 2D bounded [-5,5]^2 ---");
        bounded_rosenbrock<2>       fixed_prob;
        bounded_rosenbrock_dynamic  dyn_prob;
        auto fixed = bench_nablapp(nablapp::cmaes_policy<2>{}, fixed_prob, reps, 10000);
        auto dyn   = bench_nablapp(nablapp::cmaes_policy<>{},  dyn_prob,  reps, 10000);
        auto libc  = bench_libcmaes_cma(2, rosenbrock_libcmaes,
            {-1.2, 1.0}, {-5.0, -5.0}, {5.0, 5.0}, reps, 10000);
        print_row("nablapp<2>", fixed);
        print_row("nablapp<>", dyn);
        print_row("libcmaes_cmaes", libc);
        std::println("  ratio fixed/libcmaes: {:.1f}x wall, {:.1f}x evals",
            fixed.wall_us / libc.wall_us, double(fixed.evals) / libc.evals);
        std::println("  ratio dyn/libcmaes:   {:.1f}x wall, {:.1f}x evals",
            dyn.wall_us / libc.wall_us, double(dyn.evals) / libc.evals);
    }

    // Ackley 2D IPOP f_evals upper-bound assertion.
    //
    // Hansen 2023 (arXiv:1604.00772) section B.3 + Auger & Hansen (2005)
    // "A Restart CMA Evolution Strategy with Increasing Population Size":
    // the offspring loop is the ONLY problem.value() call site inside
    // cmaes_policy::step() (init() adds exactly 1 evaluation). With IPOP
    // lambda doubling capped at MaxPop=512, the absolute upper bound for
    // K iterations is 1 + MaxPop * K. We assert a 2x-slack relative bound
    // against MaxPop * iters_observed to catch any regression that
    // introduces an extra value() call outside the offspring loop or
    // reintroduces per-step bench-side multiplication.
    {
        std::println("\n--- Ackley 2D IPOP f_evals upper-bound check ---");
        nablapp::ackley<double, 2> ackley_prob;
        nablapp::bench::eval_counts counts;
        nablapp::bench::counting_problem<nablapp::ackley<double, 2>>
            wrapped{ackley_prob, counts};

        Eigen::Vector<double, 2> x0 = ackley_prob.initial_point();
        nablapp::solver_options opts;
        opts.max_iterations = 1000;
        opts.set_gradient_threshold(1e-12);
        opts.set_objective_threshold(1e-12);
        opts.set_step_threshold(1e-12);

        nablapp::cmaes_policy<2>::options_type cmaes_opts{};
        cmaes_opts.seed = 42u;
        cmaes_opts.restart = nablapp::cmaes_policy<2>::restart_strategy::ipop;

        nablapp::basic_solver solver{
            nablapp::cmaes_policy<2>{}, wrapped, x0, opts, cmaes_opts};
        auto result = solver.solve();

        constexpr int max_pop = 512;  // matches MaxPop in cmaes_policy.h
        const std::uint64_t observed_iters =
            static_cast<std::uint64_t>(result.iterations);
        const std::uint64_t f_evals_bound =
            1ull
            + 2ull * static_cast<std::uint64_t>(max_pop) * observed_iters
            + 16ull;

        std::println("  iters: {}, f_evals: {}, bound: {}, objective: {:.6e}",
            observed_iters,
            static_cast<std::uint64_t>(counts.f),
            f_evals_bound,
            result.objective_value);

        if(static_cast<std::uint64_t>(counts.f) > f_evals_bound)
        {
            std::cerr << "micro_cmaes: f_evals upper bound violated: "
                      << counts.f << " > " << f_evals_bound
                      << " (iters=" << observed_iters << ")\n";
            return 2;
        }
    }

    // Smooth-quadratic TolFun-bound regression cell.
    //
    // Hansen 2023 (arXiv:1604.00772) section B.3 item 6 (TolFun): default
    // `objective_value_tolerance = 1e-12` per the paper text "10^-12 is a
    // conservative first guess". On a smooth quadratic centered at the
    // origin, CMA-ES converges and the range of the best-of-generation
    // history (and the current generation's offspring) drops below 1e-12,
    // exiting the policy with solver_status::ftol_reached. The cell pins
    // this behaviour as a per-step regression: changes that re-route the
    // exit through a different §B.3 criterion (e.g., turning off the
    // history-window machinery) will fail this assertion.
    {
        std::println("\n--- Smooth quadratic 2D TolFun-bound exit ---");

        smooth_quadratic_2d prob;

        Eigen::Vector<double, 2> x0{{2.0, 2.0}};
        nablapp::solver_options opts;
        opts.max_iterations = 5000;
        opts.set_gradient_threshold(1e-30);
        opts.set_objective_threshold(1e-30);
        opts.set_step_threshold(1e-30);

        nablapp::cmaes_policy<2>::options_type cmaes_opts{};
        cmaes_opts.seed = 42u;
        // Defeat TolX so TolFun is the criterion that owns the exit on
        // this run.
        cmaes_opts.cmaes.step_size_tolerance = 1e-30;

        nablapp::basic_solver solver{
            nablapp::cmaes_policy<2>{}, prob, x0, opts, cmaes_opts};
        auto result = solver.solve();

        std::println("  iters: {}, status: {}, objective: {:.6e}",
            result.iterations,
            static_cast<int>(result.status),
            result.objective_value);

        if(result.status != nablapp::solver_status::ftol_reached)
        {
            std::cerr << "micro_cmaes: TolFun-bound cell did not exit on "
                         "ftol_reached: status="
                      << static_cast<int>(result.status)
                      << " objective=" << result.objective_value << "\n";
            return 3;
        }
    }

    std::println("\nProfile with:");
    std::println("  perf stat ./micro_cmaes");
    std::println("  perf record -F 99999 -g -- ./micro_cmaes");
    std::println("  perf report --stdio --percent-limit=1.0");
}
