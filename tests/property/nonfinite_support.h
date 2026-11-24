#ifndef HPP_GUARD_ARGMIN_TESTS_PROPERTY_NONFINITE_SUPPORT_H
#define HPP_GUARD_ARGMIN_TESTS_PROPERTY_NONFINITE_SUPPORT_H

#include "property_support.h"

#include "oracle/result_oracle.h"
#include "oracle/callback_tracer.h"
#include "oracle/injection_schedule.h"

#include "argmin/solver/options.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/result/status.h"
#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <rapidcheck.h>

#include <cmath>
#include <string>
#include <vector>
#include <cstdint>

namespace argmin::property
{

inline const std::vector<oracle::callback>& objective_and_constraint_callbacks()
{
    static const std::vector<oracle::callback> callbacks{oracle::callback::value, oracle::callback::gradient, oracle::callback::constraints, oracle::callback::constraint_jacobian};
    return callbacks;
}

inline rc::Gen<oracle::injection> injection_into(const std::vector<oracle::callback>& callbacks)
{
    return rc::gen::exec(
        [&callbacks]
        {
            const oracle::callback target = *rc::gen::elementOf(callbacks);
            const oracle::non_finite kind = *rc::gen::element(oracle::non_finite::nan, oracle::non_finite::positive_infinity, oracle::non_finite::negative_infinity);
            const oracle::injection_mode mode = *rc::gen::element(oracle::injection_mode::first_call, oracle::injection_mode::kth_call, oracle::injection_mode::from_kth_call);
            const std::uint32_t k = mode == oracle::injection_mode::first_call ? 1u : *rc::gen::inRange<std::uint32_t>(2, 40);
            return oracle::injection{target, kind, mode, k};
        });
}

// The first call of each callback is at the start, so an injection there is a non-finite start; any later one is a
// non-finite trial, which may neither back a claim nor become the returned point. A population method that never
// evaluates the start sees every injection as a trial.
enum class start_evaluation : std::uint8_t
{
    first_call,
    never
};

template <typename Result>
std::string non_finite_failure(const Result& result, const oracle::injection_schedule& schedule, start_evaluation start)
{
    if(schedule.injects_first_call() && start == start_evaluation::first_call)
        return result.status == solver_status::non_finite_evaluation ? std::string() : "a non-finite start did not end non_finite_evaluation";
    const bool claims = oracle::claim_of(result.status) > oracle::claim::none;
    if(claims && !(std::isfinite(result.objective_value) && std::isfinite(result.constraint_violation)))
        return "a claim with a non-finite objective or violation";
    if(schedule.only_injected_at(result.x))
        return "the returned point is one at which only a non-finite value was seen";
    return std::string();
}

inline void log_injection(const oracle::injection& planned)
{
    RC_LOG() << "callback " << static_cast<std::int32_t>(planned.target) << ", kind " << static_cast<std::int32_t>(planned.kind) << ", mode " << static_cast<std::int32_t>(planned.mode) << ", k " << planned.k
             << '\n';
}

template <typename Policy, typename Problem>
void check_non_finite_once(const Policy& policy, const Problem& problem, const oracle::injection& planned, const solver_options<>& options, start_evaluation start)
{
    const oracle::traced<Problem> traced{problem, planned};
    step_budget_solver solver{policy, traced, traced.initial_point(), options};
    const solve_result<double> result = solver.solve(options);
    log_injection(planned);
    RC_LOG() << "status " << static_cast<std::int32_t>(result.status) << " after " << result.iterations << ", f " << result.objective_value << ", cv " << result.constraint_violation << ", injected "
             << traced.injections().injected_count() << '\n';
    RC_TAG(static_cast<std::int32_t>(result.status));
    RC_ASSERT(non_finite_failure(result, traced.injections(), start) == std::string());
}

template <typename Generator, typename Policy>
void check_non_finite(const std::string& key, const Generator& generator, const Policy& policy, const solver_options<>& options,
                      const std::vector<oracle::callback>& callbacks = objective_and_constraint_callbacks(), start_evaluation start = start_evaluation::first_call)
{
    check(key + " never reports a non-finite value as success or as its point", [&] { check_non_finite_once(policy, *generator, *injection_into(callbacks), options, start); });
}

}

#endif
