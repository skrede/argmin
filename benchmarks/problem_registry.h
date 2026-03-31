#ifndef HPP_GUARD_NABLAPP_BENCHMARKS_PROBLEM_REGISTRY_H
#define HPP_GUARD_NABLAPP_BENCHMARKS_PROBLEM_REGISTRY_H

// Compile-time registry of all benchmark test problems.
//
// Provides for_each_problem() which invokes a callable on every registered
// problem, and for_each_problem_of_class() which filters by problem_class.
// This avoids type-erasure overhead while allowing the driver to iterate
// all problems generically.

#include "nablapp/test_functions/beale.h"
#include "nablapp/test_functions/booth.h"
#include "nablapp/test_functions/ackley.h"
#include "nablapp/test_functions/schwefel.h"
#include "nablapp/test_functions/griewank.h"
#include "nablapp/test_functions/rosenbrock.h"
#include "nablapp/test_functions/rastrigin.h"
#include "nablapp/test_functions/himmelblau.h"
#include "nablapp/test_functions/problem_class.h"
#include "nablapp/test_functions/hock_schittkowski.h"
#include "nablapp/test_functions/more_garbow_hillstrom.h"

#include <string_view>
#include <utility>

namespace nablapp::bench
{

// Invokes fn(name, problem_instance) for each registered problem.
// fn signature: void(std::string_view name, auto&& problem)
template <typename Fn>
void for_each_problem(Fn&& fn)
{
    // Unconstrained (fixed dimension)
    fn("beale", nablapp::beale<>{});
    fn("booth", nablapp::booth<>{});
    fn("himmelblau", nablapp::himmelblau<>{});

    // Unconstrained (variable dimension)
    fn("rosenbrock_2", nablapp::rosenbrock<>{.n = 2});
    fn("rosenbrock_10", nablapp::rosenbrock<>{.n = 10});

    // MGH unconstrained
    fn("powell_singular", nablapp::powell_singular<>{});
    fn("brown_badly_scaled", nablapp::brown_badly_scaled<>{});
    fn("trigonometric_5", nablapp::trigonometric<>{.n = 5});
    fn("wood", nablapp::wood<>{});
    fn("helical_valley", nablapp::helical_valley<>{});
    fn("penalty_i_4", nablapp::penalty_i<>{.n = 4});
    fn("variably_dimensioned_5", nablapp::variably_dimensioned<>{.n = 5});
    fn("extended_rosenbrock_4", nablapp::extended_rosenbrock<>{.n = 4});

    // Bound-constrained
    fn("hs001", nablapp::hs001<>{});
    fn("hs002", nablapp::hs002<>{});
    fn("hs005", nablapp::hs005<>{});

    // Inequality-constrained
    fn("hs024", nablapp::hs024<>{});
    fn("hs035", nablapp::hs035<>{});
    fn("hs043", nablapp::hs043<>{});
    fn("hs076", nablapp::hs076<>{});

    // Equality-constrained
    fn("hs006", nablapp::hs006<>{});
    fn("hs007", nablapp::hs007<>{});
    fn("hs026", nablapp::hs026<>{});
    fn("hs028", nablapp::hs028<>{});
    fn("hs039", nablapp::hs039<>{});
    fn("hs040", nablapp::hs040<>{});
    fn("hs048", nablapp::hs048<>{});
    fn("hs050", nablapp::hs050<>{});
    fn("hs051", nablapp::hs051<>{});

    // Mixed (equality + inequality + bounds)
    fn("hs071", nablapp::hs071<>{});

    // Global (variable dimension)
    fn("ackley_2", nablapp::ackley<>{.n = 2});
    fn("ackley_10", nablapp::ackley<>{.n = 10});
    fn("rastrigin_2", nablapp::rastrigin<>{.n = 2});
    fn("rastrigin_10", nablapp::rastrigin<>{.n = 10});
    fn("schwefel_2", nablapp::schwefel<>{.n = 2});
    fn("griewank_2", nablapp::griewank<>{.n = 2});
}

// Filter variant: invokes fn only for problems matching the given class.
template <typename Fn>
void for_each_problem_of_class(problem_class filter, Fn&& fn)
{
    for_each_problem([&](std::string_view name, auto&& prob) {
        if(has_class(prob.pclass, filter))
            fn(name, std::forward<decltype(prob)>(prob));
    });
}

}

#endif
