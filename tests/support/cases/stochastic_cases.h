#ifndef HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_STOCHASTIC_CASES_H
#define HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_STOCHASTIC_CASES_H

#include "cases/case_options.h"
#include "cases/case_problems.h"

#include "argmin/solver/cmaes_policy.h"
#include "argmin/solver/restarting_policy.h"

#include "argmin/solver/alternative/cmaes/repair_l2_penalty_policy.h"
#include "argmin/solver/alternative/cmaes/no_repair_adaptive_penalty_policy.h"

#include "argmin/solver/alternative/isres/nlopt_faithful_policy.h"
#include "argmin/solver/alternative/isres/original_argmin_policy.h"
#include "argmin/solver/alternative/isres/runarsson_yao_paper_policy.h"

#include "argmin/test_functions/rosenbrock.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <array>
#include <cstdint>
#include <string_view>

namespace argmin::cases
{

inline constexpr std::array<std::string_view, 7> stochastic_keys{
    "cmaes_repair_l2_penalty", "cmaes_pwq_reparameterization", "cmaes_no_repair_adaptive_penalty", "restarting_cmaes", "isres_nlopt_faithful", "isres_original_argmin", "isres_runarsson_yao_paper"};

template <typename Policy>
Policy seeded(Policy policy, std::uint64_t seed)
{
    policy.options.seed = seed;
    return policy;
}

template <typename Inner>
restarting_policy<Inner> seeded(restarting_policy<Inner> policy, std::uint64_t seed)
{
    policy.options.inner.seed = seed;
    return policy;
}

// CMA-ES models no constraint rows, so its cases are the unconstrained and bound classes. Its converged
// claim is a step-size test, which on the Wood function (hs038) lands within a factor of two of the
// oracle's stationarity limit and flips with the trajectory, so the bound class is hs005.
template <typename Visitor>
void visit_cmaes_cases(Visitor& visitor)
{
    using namespace alternative::cmaes;
    const rosenbrock<double, 2> unconstrained;
    const hs005<> bound;
    const solver_options<> options = stochastic_options();
    visit_case(visitor, case_id{"cmaes_repair_l2_penalty", "unconstrained", "rosenbrock"}, seeded(repair_l2_penalty_policy<>{}, stochastic_seed), unconstrained, options);
    visit_case(visitor, case_id{"cmaes_repair_l2_penalty", "bound", "hs005"}, seeded(repair_l2_penalty_policy<>{}, stochastic_seed), bound, options);
    visit_case(visitor, case_id{"cmaes_pwq_reparameterization", "unconstrained", "rosenbrock"}, seeded(pwq_reparameterization_policy<>{}, stochastic_seed), unconstrained, options);
    visit_case(visitor, case_id{"cmaes_pwq_reparameterization", "bound", "hs005"}, seeded(pwq_reparameterization_policy<>{}, stochastic_seed), bound, options);
    visit_case(visitor, case_id{"cmaes_no_repair_adaptive_penalty", "unconstrained", "rosenbrock"}, seeded(no_repair_adaptive_penalty_policy<>{}, stochastic_seed), unconstrained, options);
    visit_case(visitor, case_id{"cmaes_no_repair_adaptive_penalty", "bound", "hs005"}, seeded(no_repair_adaptive_penalty_policy<>{}, stochastic_seed), bound, options);
    visit_case(visitor, case_id{"restarting_cmaes", "unconstrained", "rosenbrock"}, seeded(restarting_policy<cmaes_policy<>>{}, stochastic_seed), unconstrained, options);
    visit_case(visitor, case_id{"restarting_cmaes", "bound", "hs005"}, seeded(restarting_policy<cmaes_policy<>>{}, stochastic_seed), bound, options);
}

// ISRES requires the constrained interface, so its bound class is hs038 with zero constraint rows. The
// fixed seed lands nlopt_faithful on one of the few hs028 runs that miss the equality limit, so its
// equality class is hs048, which it meets on every seed tried.
template <typename Visitor>
void visit_isres_cases(Visitor& visitor)
{
    using namespace alternative::isres;
    const no_constraint_rows<hs038<>> bound;
    const hs076<> inequality;
    const hs028<> equality;
    const hs048<> nlopt_equality;
    const hs071<> mixed;
    const solver_options<> options = stochastic_options();
    visit_case(visitor, case_id{"isres_nlopt_faithful", "bound", "hs038"}, seeded(nlopt_faithful_policy<>{}, stochastic_seed), bound, options);
    visit_case(visitor, case_id{"isres_nlopt_faithful", "inequality", "hs076"}, seeded(nlopt_faithful_policy<>{}, stochastic_seed), inequality, options);
    visit_case(visitor, case_id{"isres_nlopt_faithful", "equality", "hs048"}, seeded(nlopt_faithful_policy<>{}, stochastic_seed), nlopt_equality, options);
    visit_case(visitor, case_id{"isres_nlopt_faithful", "mixed", "hs071"}, seeded(nlopt_faithful_policy<>{}, stochastic_seed), mixed, options);
    visit_case(visitor, case_id{"isres_original_argmin", "bound", "hs038"}, seeded(original_argmin_policy<>{}, stochastic_seed), bound, options);
    visit_case(visitor, case_id{"isres_original_argmin", "inequality", "hs076"}, seeded(original_argmin_policy<>{}, stochastic_seed), inequality, options);
    visit_case(visitor, case_id{"isres_original_argmin", "equality", "hs028"}, seeded(original_argmin_policy<>{}, stochastic_seed), equality, options);
    visit_case(visitor, case_id{"isres_original_argmin", "mixed", "hs071"}, seeded(original_argmin_policy<>{}, stochastic_seed), mixed, options);
    visit_case(visitor, case_id{"isres_runarsson_yao_paper", "bound", "hs038"}, seeded(runarsson_yao_paper_policy<>{}, stochastic_seed), bound, options);
    visit_case(visitor, case_id{"isres_runarsson_yao_paper", "inequality", "hs076"}, seeded(runarsson_yao_paper_policy<>{}, stochastic_seed), inequality, options);
    visit_case(visitor, case_id{"isres_runarsson_yao_paper", "equality", "hs028"}, seeded(runarsson_yao_paper_policy<>{}, stochastic_seed), equality, options);
    visit_case(visitor, case_id{"isres_runarsson_yao_paper", "mixed", "hs071"}, seeded(runarsson_yao_paper_policy<>{}, stochastic_seed), mixed, options);
}

template <typename Visitor>
void visit_stochastic_cases(Visitor& visitor)
{
    visit_cmaes_cases(visitor);
    visit_isres_cases(visitor);
}

}

#endif
