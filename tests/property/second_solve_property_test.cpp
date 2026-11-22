#include "nlp_policies.h"
#include "reset_support.h"
#include "generated_problems.h"
#include "generated_least_squares.h"

#include <catch2/catch_test_macros.hpp>

using namespace argmin;
using namespace argmin::property;

TEST_CASE("a second solve on one options object equals the first for the gradient solvers", "[property]")
{
    const solver_options<> options = cases::gradient_options();
    check_second_solve("lbfgsb", bound_quadratic(), lbfgsb_policy<>{}, options);
    check_second_solve("byrd_lbfgsb", bound_quadratic(), byrd_lbfgsb_policy<>{}, options);
    check_second_solve("multistart_lbfgsb", bound_quadratic(), multistart_policy<lbfgsb_policy<>>{}, options);
    check_second_solve("augmented_lagrangian_lbfgsb", mixed_quadratic(), augmented_lagrangian_policy<lbfgsb_policy<>>{}, options);
    check_second_solve("kraft_slsqp", mixed_quadratic(), kraft_slsqp_policy<>{}, options);
    check_second_solve("filter_slsqp", mixed_quadratic(), filter_slsqp_policy<>{}, options);
    check_second_solve("nw_sqp", mixed_quadratic(), nw_sqp_policy<>{}, options);
    check_second_solve("filter_nw_sqp", mixed_quadratic(), filter_nw_sqp_policy<>{}, options);
    check_second_solve("tr_sqp_fast", mixed_quadratic(), tr_sqp<sqp_mode::fast>{}, options);
    check_second_solve("tr_sqp_accurate", mixed_quadratic(), tr_sqp<sqp_mode::accurate>{}, options);
    check_second_solve("filter_trsqp_fast", mixed_quadratic(), filter_trsqp<sqp_mode::fast>{}, options);
    check_second_solve("filter_trsqp_accurate", mixed_quadratic(), filter_trsqp<sqp_mode::accurate>{}, options);
}

TEST_CASE("a second solve on one options object equals the first for the moving-asymptote, least-squares and derivative-free solvers", "[property]")
{
    const auto problems = inequality_quadratic(start_kind::feasible);
    const auto boxed = boxed_least_squares(box_kind::around_start);
    check_second_solve("mma", problems, mma_policy<>{}, cases::gradient_options());
    check_second_solve("ccsa_quadratic", problems, ccsa_quadratic_policy<>{}, cases::gradient_options());
    check_second_solve("gcmma_move_limit_shrink", problems, alternative::gcmma::move_limit_shrink_policy<>{}, cases::gradient_options());
    check_second_solve("gcmma_raa_augmented", problems, alternative::gcmma::raa_augmented_policy<>{}, cases::gradient_options());
    check_second_solve("gcmma_rho_wval", problems, alternative::gcmma::rho_wval_policy<>{}, cases::gradient_options());
    check_second_solve("lm", unconstrained_least_squares(), lm_policy<>{}, cases::least_squares_options());
    check_second_solve("projected_gn", boxed, projected_gn_policy<>{}, cases::least_squares_options());
    check_second_solve("projected_gradient_gn", boxed, projected_gradient_gn_policy<>{}, cases::least_squares_options());
    check_second_solve("bobyqa", bound_quadratic(), bobyqa_policy<>{}, cases::derivative_free_options());
    check_second_solve("cobyla", inequality_quadratic(), cobyla_policy<>{}, cases::derivative_free_options());
}

TEST_CASE("a second solve on one options object equals the first for the stochastic solvers and the restarting decorator", "[property]")
{
    const solver_options<> options = short_stochastic_options();
    check_second_solve("cmaes_repair_l2_penalty", bound_quadratic(), seeded(alternative::cmaes::repair_l2_penalty_policy<>{}), options);
    check_second_solve("cmaes_pwq_reparameterization", bound_quadratic(), seeded(alternative::cmaes::pwq_reparameterization_policy<>{}), options);
    check_second_solve("cmaes_no_repair_adaptive_penalty", bound_quadratic(), seeded(alternative::cmaes::no_repair_adaptive_penalty_policy<>{}), options);
    check_second_solve("restarting_cmaes", bound_quadratic(), restarting_cmaes_that_restarts(), options);
    check_second_solve("isres_nlopt_faithful", inequality_quadratic(), seeded(alternative::isres::nlopt_faithful_policy<>{}), options);
    check_second_solve("isres_original_argmin", inequality_quadratic(), seeded(alternative::isres::original_argmin_policy<>{}), options);
    check_second_solve("isres_runarsson_yao_paper", inequality_quadratic(), seeded(alternative::isres::runarsson_yao_paper_policy<>{}), options);
}
