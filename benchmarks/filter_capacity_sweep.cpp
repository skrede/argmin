// Filter capacity / at-capacity-policy sweep.
//
// The filter set backing the filter-SQP policies (filter_slsqp,
// filter_trsqp, filter_nw_sqp) stores a non-dominated frontier of
// rejected (objective, violation) pairs in a std::vector. Dominance
// pruning bounds its growth in practice, but push_back can allocate at
// steady state and the worst-case entry count is unbounded. Bounding it
// requires a fixed capacity plus a defined at-capacity policy -- and the
// capacity value AND the drop heuristic must be selected by evidence,
// not by fiat.
//
// This driver measures reality, then sweeps the design space:
//
//   Phase A (measure): run the real filter policies on the filter-lineage
//   HS constrained suite with an effectively-unbounded capacity, recording
//   the natural high-water entry count per cell plus the reference
//   convergence (objective, constraint violation, outer iterations).
//
//   Phase B (sweep): re-run each cell under candidate at-capacity
//   mechanisms -- (0) reject the incoming trial, (1) drop the maximal-
//   violation frontier entry, (2) FIFO ring (drop the oldest) -- across a
//   grid of caps spanning small forced-eviction values up to generous
//   multiples of the observed high-water. Per (mechanism, cap) cell the
//   driver reports whether eviction EVER fired and how many suite cells
//   diverged from the unbounded reference trajectory.
//
// The instrumentation lives behind ARGMIN_FILTER_CAPACITY_SWEEP in
// detail/filter_acceptance.h: a process-global capacity/policy config and
// a high-water/eviction record that this driver sets before, and reads
// after, each solve. The shipping library never compiles that path.
//
// Reference: Hock & Schittkowski 1981, Test Examples for Nonlinear
//            Programming Codes (HS024/026/028/035/039/040/043/050/071/076);
//            Fletcher & Leyffer 2002, Math. Program. 91:239-269 (filter
//            method); Wachter & Biegler 2006, Math. Program. 106:25-57
//            Section 2.3 (filter frontier).

#define ARGMIN_FILTER_CAPACITY_SWEEP 1

#include "argmin/detail/filter_acceptance.h"

#include "argmin/solver/filter_nw_sqp_policy.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/filter_trsqp_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/sqp_mode.h"
#include "argmin/solver/options.h"

#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <functional>

namespace
{

struct cell_result
{
    double f{};
    double cv{};
    std::uint32_t iters{};
    std::size_t highwater{};
    std::size_t evictions{};
};

// A single (policy, HS problem) cell: a closure that runs the real solver
// under a chosen capacity + at-capacity policy and returns the resulting
// convergence plus the filter high-water / eviction record.
struct suite_cell
{
    std::string name;
    std::function<cell_result(std::size_t cap, int overflow_policy)> run;
};

template <template <int> class Policy, typename Problem>
cell_result solve_cell(const Problem& problem, std::size_t cap, int overflow_policy)
{
    auto& st = argmin::detail::filter_sweep_state();
    st.capacity = cap;
    st.overflow_policy = overflow_policy;
    st.highwater = 0;
    st.evictions = 0;

    auto x0 = problem.initial_point();
    argmin::solver_options<> opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);

    argmin::step_budget_solver solver{
        Policy<Problem::problem_dimension>{}, problem, x0, opts};
    auto result = solver.solve(opts);

    return {result.objective_value, result.constraint_violation,
            result.iterations, st.highwater, st.evictions};
}

template <template <int> class Policy, typename Problem>
suite_cell make_cell(std::string name, Problem problem)
{
    return suite_cell{
        std::move(name),
        [problem](std::size_t cap, int overflow_policy) {
            return solve_cell<Policy>(problem, cap, overflow_policy);
        }};
}

// The ten filter-lineage HS cells: the failing family that motivates
// filter acceptance (HS024/035/039/040/043/050) plus the reference cells
// used as the non-regression bar (HS026/028/071/076).
template <template <int> class Policy>
std::vector<suite_cell> build_suite()
{
    std::vector<suite_cell> cells;
    cells.push_back(make_cell<Policy>("HS024", argmin::hs024<>{}));
    cells.push_back(make_cell<Policy>("HS026", argmin::hs026<>{}));
    cells.push_back(make_cell<Policy>("HS028", argmin::hs028<>{}));
    cells.push_back(make_cell<Policy>("HS035", argmin::hs035<>{}));
    cells.push_back(make_cell<Policy>("HS039", argmin::hs039<>{}));
    cells.push_back(make_cell<Policy>("HS040", argmin::hs040<>{}));
    cells.push_back(make_cell<Policy>("HS043", argmin::hs043<>{}));
    cells.push_back(make_cell<Policy>("HS050", argmin::hs050<>{}));
    cells.push_back(make_cell<Policy>("HS071", argmin::hs071<>{}));
    cells.push_back(make_cell<Policy>("HS076", argmin::hs076<>{}));
    return cells;
}

// filter_trsqp mode-fixed aliases so the mode-templated policy fits the
// single-parameter policy-template contract used by the runner.
template <int N>
using filter_trsqp_accurate = argmin::filter_trsqp_policy<N, argmin::sqp_mode::accurate>;
template <int N>
using filter_trsqp_fast = argmin::filter_trsqp_policy<N, argmin::sqp_mode::fast>;

constexpr std::size_t unbounded_cap = 1u << 30;

bool diverged(const cell_result& ref, const cell_result& trial)
{
    const double f_rel = std::abs(ref.f) > 1.0 ? std::abs(ref.f) : 1.0;
    if(std::abs(ref.f - trial.f) > 1e-9 * f_rel)
        return true;
    if(std::abs(ref.cv - trial.cv) > 1e-9)
        return true;
    if(ref.iters != trial.iters)
        return true;
    return false;
}

struct policy_report
{
    std::string policy;
    std::vector<std::string> cell_names;
    std::vector<cell_result> reference; // unbounded per cell
    std::size_t highwater_max{};
};

policy_report run_measure(const char* policy_name, std::vector<suite_cell>& cells)
{
    policy_report rep;
    rep.policy = policy_name;
    std::printf("\n=== %s : Phase A measure (unbounded capacity) ===\n", policy_name);
    std::printf("%-8s %-16s %-14s %-8s %-10s\n",
                "cell", "f", "cv", "iters", "highwater");
    for(auto& c : cells)
    {
        cell_result r = c.run(unbounded_cap, 0);
        rep.cell_names.push_back(c.name);
        rep.reference.push_back(r);
        rep.highwater_max = std::max(rep.highwater_max, r.highwater);
        std::printf("%-8s %-16.8e %-14.4e %-8u %-10zu\n",
                    c.name.c_str(), r.f, r.cv, r.iters, r.highwater);
    }
    std::printf("  high-water (max over suite): %zu\n", rep.highwater_max);
    return rep;
}

void run_sweep(std::vector<suite_cell>& cells, const policy_report& rep,
               const std::vector<std::size_t>& caps,
               const std::array<const char*, 3>& mech_names)
{
    std::printf("\n=== %s : Phase B sweep (cap x mechanism) ===\n",
                rep.policy.c_str());
    std::printf("%-6s %-18s %-14s %-16s\n",
                "cap", "mechanism", "suite_evict", "cells_diverged");
    for(std::size_t cap : caps)
    {
        for(int mech = 0; mech < 3; ++mech)
        {
            std::size_t total_evict = 0;
            int diverged_cells = 0;
            for(std::size_t i = 0; i < cells.size(); ++i)
            {
                cell_result r = cells[i].run(cap, mech);
                total_evict += r.evictions;
                if(diverged(rep.reference[i], r))
                    ++diverged_cells;
            }
            std::printf("%-6zu %-18s %-14zu %-16d\n",
                        cap, mech_names[static_cast<std::size_t>(mech)],
                        total_evict, diverged_cells);
        }
    }
}

template <template <int> class Policy>
void run_policy(const char* policy_name)
{
    auto cells = build_suite<Policy>();
    policy_report rep = run_measure(policy_name, cells);

    // Caps: small forced-eviction values (below any realistic high-water)
    // to exercise + differentiate the mechanisms, then generous multiples
    // of the observed high-water where eviction is expected never to fire.
    const std::size_t hw = rep.highwater_max == 0 ? 1 : rep.highwater_max;
    std::vector<std::size_t> caps{4, 8,
                                  hw,          // tight: exactly the high-water
                                  2 * hw,      // 2x
                                  4 * hw,      // 4x
                                  8 * hw};     // 8x
    const std::array<const char*, 3> mech_names{
        "reject", "drop_max_theta", "ring_fifo"};
    run_sweep(cells, rep, caps, mech_names);
}

}

int main()
{
    std::printf("Filter capacity / at-capacity-policy sweep\n");
    std::printf("Filter-lineage HS suite: "
                "HS024/026/028/035/039/040/043/050/071/076\n");

    run_policy<argmin::filter_slsqp_policy>("filter_slsqp");
    run_policy<filter_trsqp_accurate>("filter_trsqp[accurate]");
    run_policy<filter_trsqp_fast>("filter_trsqp[fast]");
    run_policy<argmin::filter_nw_sqp_policy>("filter_nw_sqp");
    return 0;
}
