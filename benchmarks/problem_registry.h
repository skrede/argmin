#ifndef HPP_GUARD_ARGMIN_BENCHMARKS_PROBLEM_REGISTRY_H
#define HPP_GUARD_ARGMIN_BENCHMARKS_PROBLEM_REGISTRY_H

// Compile-time registry of all benchmark test problems.
//
// Provides for_each_problem() which invokes a callable on every registered
// problem, and for_each_problem_of_class() which filters by problem_class.
// This avoids type-erasure overhead while allowing the driver to iterate
// all problems generically.

#include "argmin/test_functions/beale.h"
#include "argmin/test_functions/booth.h"
#include "argmin/test_functions/ackley.h"
#include "argmin/test_functions/schwefel.h"
#include "argmin/test_functions/griewank.h"
#include "argmin/test_functions/nmpc_lqr.h"
#include "argmin/test_functions/rosenbrock.h"
#include "argmin/test_functions/rastrigin.h"
#include "argmin/test_functions/himmelblau.h"
#include "argmin/test_functions/problem_class.h"
#include "argmin/test_functions/ik_pose_batch.h"
#include "argmin/test_functions/hock_schittkowski.h"
#include "argmin/test_functions/more_garbow_hillstrom.h"

#include <string_view>
#include <utility>

namespace argmin::bench
{

// Invokes fn(name, problem_instance) for each registered problem.
// fn signature: void(std::string_view name, auto&& problem)
template <typename Fn>
void for_each_problem(Fn&& fn)
{
    // Unconstrained (fixed dimension)
    fn("beale", argmin::beale<>{});
    fn("booth", argmin::booth<>{});
    fn("himmelblau", argmin::himmelblau<>{});

    // Unconstrained (variable dimension)
    fn("rosenbrock_2", argmin::rosenbrock<>{.n = 2});
    fn("rosenbrock_10", argmin::rosenbrock<>{.n = 10});

    // MGH unconstrained
    fn("powell_singular", argmin::powell_singular<>{});
    fn("brown_badly_scaled", argmin::brown_badly_scaled<>{});
    fn("trigonometric_5", argmin::trigonometric<>{.n = 5});
    fn("wood", argmin::wood<>{});
    fn("helical_valley", argmin::helical_valley<>{});
    fn("penalty_i_4", argmin::penalty_i<>{.n = 4});
    fn("variably_dimensioned_5", argmin::variably_dimensioned<>{.n = 5});
    fn("extended_rosenbrock_4", argmin::extended_rosenbrock<>{.n = 4});

    // Hock-Schittkowski constrained and bound-constrained problems
    fn("hs001", argmin::hs001<>{});
    fn("hs002", argmin::hs002<>{});
    fn("hs005", argmin::hs005<>{});
    fn("hs006", argmin::hs006<>{});
    fn("hs007", argmin::hs007<>{});
    fn("hs021", argmin::hs021<>{});
    fn("hs023", argmin::hs023<>{});
    fn("hs024", argmin::hs024<>{});
    fn("hs026", argmin::hs026<>{});
    fn("hs027", argmin::hs027<>{});
    fn("hs028", argmin::hs028<>{});
    fn("hs029", argmin::hs029<>{});
    fn("hs030", argmin::hs030<>{});
    fn("hs031", argmin::hs031<>{});
    fn("hs034", argmin::hs034<>{});
    fn("hs035", argmin::hs035<>{});
    fn("hs036", argmin::hs036<>{});
    fn("hs037", argmin::hs037<>{});
    fn("hs038", argmin::hs038<>{});
    fn("hs039", argmin::hs039<>{});
    fn("hs040", argmin::hs040<>{});
    fn("hs043", argmin::hs043<>{});
    fn("hs044", argmin::hs044<>{});
    fn("hs048", argmin::hs048<>{});
    fn("hs050", argmin::hs050<>{});
    fn("hs051", argmin::hs051<>{});
    fn("hs052", argmin::hs052<>{});
    fn("hs071", argmin::hs071<>{});
    fn("hs076", argmin::hs076<>{});

    // Global (variable dimension)
    fn("ackley_2", argmin::ackley<>{.n = 2});
    fn("ackley_10", argmin::ackley<>{.n = 10});
    fn("rastrigin_2", argmin::rastrigin<>{.n = 2});
    fn("rastrigin_10", argmin::rastrigin<>{.n = 10});
    fn("schwefel_2", argmin::schwefel<>{.n = 2});
    fn("griewank_2", argmin::griewank<>{.n = 2});

    // Application-shaped (NMPC, IK pose-batch)
    fn("nmpc_lqr_h10",    argmin::nmpc_lqr<10>{});
    fn("nmpc_lqr_h20",    argmin::nmpc_lqr<20>{});
    fn("ik_pose_batch_6", argmin::ik_pose_batch<6, 5>{});
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
