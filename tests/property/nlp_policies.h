#ifndef HPP_GUARD_ARGMIN_TESTS_PROPERTY_NLP_POLICIES_H
#define HPP_GUARD_ARGMIN_TESTS_PROPERTY_NLP_POLICIES_H

#include "cases/case_options.h"
#include "cases/stochastic_cases.h"

#include "argmin/solver/sqp_mode.h"
#include "argmin/solver/lm_policy.h"
#include "argmin/solver/mma_policy.h"
#include "argmin/solver/cmaes_policy.h"
#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/multistart_policy.h"
#include "argmin/solver/restarting_policy.h"
#include "argmin/solver/byrd_lbfgsb_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/filter_trsqp_policy.h"
#include "argmin/solver/projected_gn_policy.h"
#include "argmin/solver/filter_nw_sqp_policy.h"
#include "argmin/solver/ccsa_quadratic_policy.h"
#include "argmin/solver/augmented_lagrangian_policy.h"
#include "argmin/solver/projected_gradient_gn_policy.h"

#include "argmin/solver/alternative/gcmma/rho_wval_policy.h"
#include "argmin/solver/alternative/gcmma/raa_augmented_policy.h"
#include "argmin/solver/alternative/gcmma/move_limit_shrink_policy.h"

#include "argmin/solver/alternative/cmaes/repair_l2_penalty_policy.h"
#include "argmin/solver/alternative/cmaes/pwq_reparameterization_policy.h"
#include "argmin/solver/alternative/cmaes/no_repair_adaptive_penalty_policy.h"

#include "argmin/solver/alternative/isres/nlopt_faithful_policy.h"
#include "argmin/solver/alternative/isres/original_argmin_policy.h"
#include "argmin/solver/alternative/isres/runarsson_yao_paper_policy.h"

#include "argmin/types.h"

namespace argmin::property
{

template <sqp_mode Mode>
using tr_sqp = tr_sqp_policy<double, dynamic_dimension, Mode>;

template <sqp_mode Mode>
using filter_trsqp = filter_trsqp_policy<double, dynamic_dimension, Mode>;

using restarting_cmaes = restarting_policy<cmaes_policy<>>;

// A short stagnation limit ends runs on a convex quadratic, so a solve restarts with an escalated population before
// it is reset; the cap bounds the population growth.
inline restarting_cmaes restarting_cmaes_that_restarts()
{
    restarting_cmaes policy = cases::seeded(restarting_cmaes{}, cases::stochastic_seed);
    policy.options.stagnation_limit = 3;
    policy.options.max_restarts = 3;
    return policy;
}

template <typename Policy>
Policy seeded(Policy policy)
{
    return cases::seeded(policy, cases::stochastic_seed);
}

// Bit equality needs no convergence, so the stochastic runs stop well short of the oracle cases' budget.
inline solver_options<> short_stochastic_options()
{
    solver_options<> options = cases::stochastic_options();
    options.max_iterations = 300;
    return options;
}

}

#endif
