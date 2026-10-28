#ifndef HPP_GUARD_ARGMIN_TESTS_PROPERTY_PROPERTY_JUDGE_H
#define HPP_GUARD_ARGMIN_TESTS_PROPERTY_PROPERTY_JUDGE_H

#include "property_support.h"

#include "oracle_tolerances.h"

#include "cases/case_options.h"

#include "oracle/result_oracle.h"

#include "argmin/solver/options.h"
#include "argmin/solver/step_budget_solver.h"

#include <Eigen/Core>

#include <rapidcheck.h>

#include <string>
#include <cstdint>
#include <algorithm>

namespace argmin::property
{

// On a bound-only box mma can stop moving at a non-stationary point and run out its whole budget, so
// the moving-asymptote properties take a smaller budget than the case table to bound their runtime.
inline solver_options<> moving_asymptote_property_options()
{
    solver_options<> opts = cases::moving_asymptote_options();
    opts.max_iterations = 2000;
    return opts;
}

template <typename Result>
void log_result(const Result& result, const oracle::verdict<double>& v)
{
    const oracle::kkt_legs<double>& legs = v.legs;
    RC_LOG() << "status " << static_cast<std::int32_t>(result.status) << " after " << result.iterations << " iterations, stationarity " << legs.stationarity << ", violations " << legs.equality_violation << ' '
             << legs.inequality_violation << ' ' << legs.bound_violation << ", complementarity " << legs.complementarity << ", dual sign " << legs.dual_sign_breach << "\nx "
             << result.x.transpose().format(Eigen::IOFormat(Eigen::FullPrecision)) << '\n';
}

// Every status is judged, so a status that makes no claim passes and a false claim fails. A cap holds a
// stronger claim to a weaker one; the suites name each cap they pass and why.
template <typename Policy, typename Problem>
void judge_solve(const Policy& policy, const Problem& problem, const solver_options<>& options, oracle::claim cap = oracle::claim::optimality)
{
    step_budget_solver solver{policy, problem, problem.initial_point(), options};
    const auto result = solver.solve(options);
    const oracle::tolerances<double> tol = oracle::tolerances_of(options);
    const oracle::verdict<double> v = oracle::judge(problem, result, tol);
    const oracle::claim held = std::min(v.claimed, cap);
    RC_TAG(static_cast<std::int32_t>(result.status));
    log_result(result, v);
    RC_ASSERT(result.x.size() == static_cast<Eigen::Index>(problem.dimension()));
    RC_ASSERT(result.x.allFinite());
    RC_ASSERT((held == v.claimed ? v.failing : oracle::claim_failure(held, v.legs, tol)) == std::string());
}

template <typename Generator, typename Policy>
void check_judged(const std::string& description, const Generator& generator, const Policy& policy, const solver_options<>& options, oracle::claim cap = oracle::claim::optimality)
{
    check(description, [&] { judge_solve(policy, *generator, options, cap); });
}

}

#endif
