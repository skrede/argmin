// Active-set QP micro-benchmark: Givens QR update vs full recompute.
//
// Measures per-solve wall time on two problem scales:
//   IK-scale:   n=6,  m_ineq=3  (typical inverse kinematics)
//   NMPC-scale: n=20, m_ineq=10 (typical model-predictive control)
//
// Compares the stateful solver (with Givens incremental QR) against
// the free-function solver (full Householder QR each iteration).
//
// Reference: N&W Algorithm 16.1, pp. 460-463 (active-set method).
//            N&W Algorithm 16.3 (QR update via Givens rotations).

#include "argmin/detail/active_set_qp.h"

#include <Eigen/Core>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace argmin::detail;

namespace
{

struct qp_problem
{
    Eigen::MatrixXd G;
    Eigen::VectorXd d;
    Eigen::MatrixXd A_eq;
    Eigen::VectorXd b_eq;
    Eigen::MatrixXd A_ineq;
    Eigen::VectorXd b_ineq;
    Eigen::VectorXd x0;
};

// Generate a random convex QP with box-like inequality constraints.
// G = A^T A + eps*I (ensures positive definiteness).
// Inequality constraints: random normals with feasible x0.
qp_problem make_random_qp(int n, int m_ineq, std::mt19937& rng)
{
    std::normal_distribution<double> normal(0.0, 1.0);

    qp_problem prob;
    Eigen::MatrixXd A_rand(n, n);
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < n; ++j)
            A_rand(i, j) = normal(rng);

    prob.G = A_rand.transpose() * A_rand +
             0.1 * Eigen::MatrixXd::Identity(n, n);

    prob.d.resize(n);
    for(int i = 0; i < n; ++i)
        prob.d[i] = normal(rng);

    prob.A_eq.resize(0, n);
    prob.b_eq.resize(0);

    // Random inequality constraints: a_i^T x >= b_i
    // Choose a_i random, b_i = a_i^T x0 - slack so x0 is feasible.
    prob.x0 = Eigen::VectorXd::Zero(n);
    prob.A_ineq.resize(m_ineq, n);
    prob.b_ineq.resize(m_ineq);
    for(int i = 0; i < m_ineq; ++i)
    {
        for(int j = 0; j < n; ++j)
            prob.A_ineq(i, j) = normal(rng);
        double slack = std::abs(normal(rng)) + 0.1;
        prob.b_ineq[i] = prob.A_ineq.row(i).dot(prob.x0) - slack;
    }

    return prob;
}

struct timing_result
{
    double mean_us;
    double stddev_us;
};

// Benchmark the stateful solver (with Givens QR updates).
timing_result bench_stateful(
    const std::vector<qp_problem>& problems,
    uint32_t warmup,
    uint32_t iterations)
{
    const int n = problems[0].G.rows();
    const int m = problems[0].A_ineq.rows();
    active_set_qp_solver<double> solver(n, m);

    // Warm up
    for(uint32_t i = 0; i < warmup; ++i)
    {
        const auto& p = problems[i % problems.size()];
        solver.solve(p.G, p.d, p.A_eq, p.b_eq, p.A_ineq, p.b_ineq, p.x0);
    }

    // Measure
    std::vector<double> times(iterations);
    for(uint32_t i = 0; i < iterations; ++i)
    {
        const auto& p = problems[i % problems.size()];
        auto t0 = std::chrono::high_resolution_clock::now();
        solver.solve(p.G, p.d, p.A_eq, p.b_eq, p.A_ineq, p.b_ineq, p.x0);
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    double sum = 0.0;
    for(auto t : times) sum += t;
    double mean = sum / static_cast<double>(iterations);

    double var = 0.0;
    for(auto t : times) var += (t - mean) * (t - mean);
    double stddev = std::sqrt(var / static_cast<double>(iterations));

    return {mean, stddev};
}

// Benchmark the free-function solver (full Householder QR each iteration).
timing_result bench_free_function(
    const std::vector<qp_problem>& problems,
    uint32_t warmup,
    uint32_t iterations)
{
    // Warm up
    for(uint32_t i = 0; i < warmup; ++i)
    {
        const auto& p = problems[i % problems.size()];
        solve_qp(p.G, p.d, p.A_eq, p.b_eq, p.A_ineq, p.b_ineq, p.x0);
    }

    // Measure
    std::vector<double> times(iterations);
    for(uint32_t i = 0; i < iterations; ++i)
    {
        const auto& p = problems[i % problems.size()];
        auto t0 = std::chrono::high_resolution_clock::now();
        solve_qp(p.G, p.d, p.A_eq, p.b_eq, p.A_ineq, p.b_ineq, p.x0);
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    double sum = 0.0;
    for(auto t : times) sum += t;
    double mean = sum / static_cast<double>(iterations);

    double var = 0.0;
    for(auto t : times) var += (t - mean) * (t - mean);
    double stddev = std::sqrt(var / static_cast<double>(iterations));

    return {mean, stddev};
}

void run_scale(
    const char* label,
    int n,
    int m_ineq,
    uint32_t warmup,
    uint32_t iterations)
{
    std::mt19937 rng(42);
    constexpr uint32_t n_problems = 50;
    std::vector<qp_problem> problems;
    problems.reserve(n_problems);
    for(uint32_t i = 0; i < n_problems; ++i)
        problems.push_back(make_random_qp(n, m_ineq, rng));

    auto stateful = bench_stateful(problems, warmup, iterations);
    auto free_fn = bench_free_function(problems, warmup, iterations);

    double speedup = free_fn.mean_us / stateful.mean_us;

    std::printf("  %-14s | %10.2f +/- %5.2f | %10.2f +/- %5.2f | %5.2fx\n",
        label,
        free_fn.mean_us, free_fn.stddev_us,
        stateful.mean_us, stateful.stddev_us,
        speedup);
}

}

int main()
{
    constexpr uint32_t warmup = 200;
    constexpr uint32_t iterations = 2000;

    std::printf("Active-Set QP Micro-Benchmark\n");
    std::printf("=============================\n");
    std::printf("  %-14s | %18s | %18s | %s\n",
        "Scale", "Full QR (us)", "Givens (us)", "Speedup");
    std::printf("  %-14s-+-%-18s-+-%-18s-+-%s\n",
        "--------------", "------------------",
        "------------------", "-------");

    run_scale("IK (6x3)",      6,  3, warmup, iterations);
    run_scale("NMPC (20x10)", 20, 10, warmup, iterations);
    run_scale("Large (40x20)", 40, 20, warmup, iterations);

    return 0;
}
