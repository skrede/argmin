// kthohr/optim comparison benchmarks for nablapp benchmark suite.
//
// Per D-01: native API adapter, no common interface. Wires kthohr/optim
// unconstrained / bound-constrained / constrained algorithms through the
// problem_registry iteration. This file is the initial scaffold —
// algorithm-by-algorithm dispatch is wired in a follow-on plan once
// counting_problem<P> exists.
//
// Sign convention: kthohr/optim treats all inequality constraints as
// c(x) <= 0; nablapp uses c_ineq(x) >= 0 feasible. Constraint wiring
// negates constraint values on the adapter boundary.
//
// OPTIM_ENABLE_EIGEN_WRAPPERS is supplied as a compile definition by the
// optim::optim INTERFACE target wired in benchmarks/CMakeLists.txt.

#include "bench_optim.h"
#include "benchmark_result.h"
#include "problem_registry.h"

#include "nablapp/formulation/concepts.h"

#include <optim.hpp>

#include <vector>

namespace nablapp::bench
{

void run_optim_benchmarks(std::vector<benchmark_result>& results, const bench_config& config)
{
    // Initial scaffold: dispatch structure only.
    // A follow-on plan fills BFGS / L-BFGS / CG / Newton / GD variants on
    // unconstrained and bound-constrained, and SUMT on constrained,
    // routing all callbacks through counting_problem<P>.
    (void)results;
    (void)config;
}

}
