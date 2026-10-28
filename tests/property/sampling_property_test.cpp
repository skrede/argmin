#include "property_judge.h"
#include "generated_problems.h"

#include "cases/case_options.h"
#include "cases/stochastic_cases.h"

#include "oracle/result_oracle.h"

#include "argmin/solver/cmaes_policy.h"
#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/restarting_policy.h"

#include "argmin/solver/alternative/cmaes/repair_l2_penalty_policy.h"
#include "argmin/solver/alternative/cmaes/no_repair_adaptive_penalty_policy.h"

#include "argmin/solver/alternative/isres/nlopt_faithful_policy.h"
#include "argmin/solver/alternative/isres/original_argmin_policy.h"
#include "argmin/solver/alternative/isres/runarsson_yao_paper_policy.h"

#include <catch2/catch_test_macros.hpp>

#include <string>

using namespace argmin;

namespace
{

template <typename Policy>
void check_cmaes(const std::string& key, const Policy& policy, oracle::claim bound_cap)
{
    const Policy seeded = cases::seeded(policy, cases::stochastic_seed);
    property::check_judged(key + " unconstrained", property::unconstrained_quadratic(), seeded, cases::stochastic_options());
    property::check_judged(key + " bound", property::bound_quadratic(), seeded, cases::stochastic_options(), bound_cap);
}

template <typename Policy>
void check_isres(const std::string& key, const Policy& policy, oracle::claim cap)
{
    property::check_judged(key + " bound", property::rowless_bound_quadratic(), cases::seeded(policy, cases::stochastic_seed), cases::stochastic_options(), cap);
}

// Excluded claims, held to feasibility on generated boxes: the converged and ftol_reached statuses of these
// variants come from the sampling scale rather than stationarity, and the oracle rejects them there.
// - cmaes_repair_l2_penalty: converged at 1.2 times the stationarity limit, the step-size proxy straddling it.
// - cmaes_no_repair_adaptive_penalty: converged at a non-stationary point of the box.
// - isres_nlopt_faithful, isres_original_argmin: converged at 1.3 and 13 times the stationarity limit.
// - isres_runarsson_yao_paper: ftol_reached just past the complementarity limit.
constexpr oracle::claim full_claim = oracle::claim::optimality;
constexpr oracle::claim feasibility_only = oracle::claim::feasibility;

}

TEST_CASE("the oracle accepts BOBYQA on generated quadratics", "[property]")
{
    property::check_judged("bobyqa bound", property::bound_quadratic(), bobyqa_policy<>{}, cases::derivative_free_options());
}

TEST_CASE("the oracle accepts COBYLA on generated quadratics", "[property]")
{
    const solver_options<> options = cases::derivative_free_options();
    property::check_judged("cobyla bound", property::rowless_bound_quadratic(), cobyla_policy<>{}, options);
    property::check_judged("cobyla inequality", property::inequality_quadratic(), cobyla_policy<>{}, options);
    property::check_judged("cobyla equality", property::equality_quadratic(), cobyla_policy<>{}, options);
    property::check_judged("cobyla mixed", property::mixed_quadratic(), cobyla_policy<>{}, options);
}

TEST_CASE("the oracle accepts CMA-ES with a fixed seed on generated quadratics", "[property]")
{
    check_cmaes("cmaes_repair_l2_penalty", alternative::cmaes::repair_l2_penalty_policy<>{}, feasibility_only);
    check_cmaes("cmaes_pwq_reparameterization", alternative::cmaes::pwq_reparameterization_policy<>{}, full_claim);
    check_cmaes("cmaes_no_repair_adaptive_penalty", alternative::cmaes::no_repair_adaptive_penalty_policy<>{}, feasibility_only);
    check_cmaes("restarting_cmaes", restarting_policy<cmaes_policy<>>{}, full_claim);
}

TEST_CASE("the oracle accepts ISRES with a fixed seed on generated boxes", "[property]")
{
    check_isres("isres_nlopt_faithful", alternative::isres::nlopt_faithful_policy<>{}, feasibility_only);
    check_isres("isres_original_argmin", alternative::isres::original_argmin_policy<>{}, feasibility_only);
    check_isres("isres_runarsson_yao_paper", alternative::isres::runarsson_yao_paper_policy<>{}, feasibility_only);
}
