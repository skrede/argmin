#include "nlp_policies.h"
#include "nonfinite_support.h"
#include "generated_problems.h"

#include "cases/case_options.h"

#include <catch2/catch_test_macros.hpp>

#include <vector>

using namespace argmin;
using namespace argmin::property;

namespace
{

const std::vector<oracle::callback>& value_callbacks()
{
    static const std::vector<oracle::callback> callbacks{oracle::callback::value};
    return callbacks;
}

const std::vector<oracle::callback>& gradient_callbacks()
{
    static const std::vector<oracle::callback> callbacks{oracle::callback::value, oracle::callback::gradient};
    return callbacks;
}

const std::vector<oracle::callback>& value_and_constraint_callbacks()
{
    static const std::vector<oracle::callback> callbacks{oracle::callback::value, oracle::callback::constraints};
    return callbacks;
}

}

TEST_CASE("the CMA-ES solvers never report a non-finite value as success or as their point", "[property]")
{
    const solver_options<> options = short_stochastic_options();
    check_non_finite("cmaes_repair_l2_penalty", bound_quadratic(), seeded(alternative::cmaes::repair_l2_penalty_policy<>{}), options, value_callbacks());
    check_non_finite("cmaes_pwq_reparameterization", bound_quadratic(), seeded(alternative::cmaes::pwq_reparameterization_policy<>{}), options, value_callbacks());
    check_non_finite("cmaes_no_repair_adaptive_penalty", bound_quadratic(), seeded(alternative::cmaes::no_repair_adaptive_penalty_policy<>{}), options, value_callbacks());
}

TEST_CASE("the ISRES solvers never report a non-finite value as success or as their point", "[property]")
{
    const solver_options<> options = short_stochastic_options();
    check_non_finite("isres_nlopt_faithful", inequality_quadratic(), seeded(alternative::isres::nlopt_faithful_policy<>{}), options, value_and_constraint_callbacks(), start_evaluation::never);
    check_non_finite("isres_original_argmin", inequality_quadratic(), seeded(alternative::isres::original_argmin_policy<>{}), options, value_and_constraint_callbacks(), start_evaluation::never);
    check_non_finite("isres_runarsson_yao_paper", inequality_quadratic(), seeded(alternative::isres::runarsson_yao_paper_policy<>{}), options, value_and_constraint_callbacks(), start_evaluation::never);
}

TEST_CASE("the gradient bound solvers and the augmented Lagrangian never report a non-finite value as success or as their point", "[property]")
{
    const solver_options<> options = cases::gradient_options();
    check_non_finite("lbfgsb", bound_quadratic(), lbfgsb_policy<>{}, options, gradient_callbacks());
    check_non_finite("byrd_lbfgsb", bound_quadratic(), byrd_lbfgsb_policy<>{}, options, gradient_callbacks());
    check_non_finite("augmented_lagrangian_lbfgsb", mixed_quadratic(), augmented_lagrangian_policy<lbfgsb_policy<>>{}, options);
}

TEST_CASE("the derivative-free solvers never report a non-finite value as success or as their point", "[property]")
{
    const solver_options<> options = cases::derivative_free_options();
    check_non_finite("bobyqa", bound_quadratic(), bobyqa_policy<>{}, options, value_callbacks());
    check_non_finite("cobyla", inequality_quadratic(), cobyla_policy<>{}, options, value_and_constraint_callbacks());
}

TEST_CASE("the decorators never report a non-finite value as success or as their point", "[property]")
{
    check_non_finite("multistart_lbfgsb", bound_quadratic(), multistart_policy<lbfgsb_policy<>>{}, cases::gradient_options(), gradient_callbacks());
    check_non_finite("restarting_cmaes", bound_quadratic(), restarting_cmaes_that_restarts(), short_stochastic_options(), value_callbacks());
}
