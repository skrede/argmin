#include "warm_sets.h"
#include "nlp_policies.h"
#include "reset_support.h"
#include "generated_problems.h"
#include "generated_least_squares.h"

#include <catch2/catch_test_macros.hpp>

using namespace argmin;
using namespace argmin::property;

// Excluded: dense_admm_qp, sparse_admm_qp and dense_active_set_qp have no reset; a re-solve warm-starts by design.

TEST_CASE("reset_clear equals a fresh solve for the gradient bound solvers and the augmented Lagrangian", "[property]")
{
    check_reset("lbfgsb", bound_quadratic(), lbfgsb_policy<>{}, cases::gradient_options());
    check_reset("byrd_lbfgsb", bound_quadratic(), byrd_lbfgsb_policy<>{}, cases::gradient_options());
    check_reset("multistart_lbfgsb", bound_quadratic(), multistart_policy<lbfgsb_policy<>>{}, cases::gradient_options());
    check_reset("augmented_lagrangian_lbfgsb", mixed_quadratic(), augmented_lagrangian_policy<lbfgsb_policy<>>{}, cases::gradient_options());
}

TEST_CASE("reset_clear equals a fresh solve for the SQP solvers", "[property]")
{
    check_reset("kraft_slsqp", mixed_quadratic(), kraft_slsqp_policy<>{}, cases::gradient_options());
    check_reset("filter_slsqp", mixed_quadratic(), filter_slsqp_policy<>{}, cases::gradient_options());
    check_reset("nw_sqp", mixed_quadratic(), nw_sqp_policy<>{}, cases::gradient_options());
    check_reset("filter_nw_sqp", mixed_quadratic(), filter_nw_sqp_policy<>{}, cases::gradient_options());
    check_reset("tr_sqp_fast", mixed_quadratic(), tr_sqp<sqp_mode::fast>{}, cases::gradient_options());
    check_reset("tr_sqp_accurate", mixed_quadratic(), tr_sqp<sqp_mode::accurate>{}, cases::gradient_options());
    check_reset("filter_trsqp_fast", mixed_quadratic(), filter_trsqp<sqp_mode::fast>{}, cases::gradient_options());
    check_reset("filter_trsqp_accurate", mixed_quadratic(), filter_trsqp<sqp_mode::accurate>{}, cases::gradient_options());
}

TEST_CASE("reset_clear equals a fresh solve for the moving-asymptote solvers", "[property]")
{
    // Bit equality needs no convergence and gcmma move_limit_shrink crawls through a large budget, so these
    // runs stop at 200 iterations.
    const auto problems = inequality_quadratic(start_kind::feasible);
    const solver_options<> options = cases::gradient_options();
    check_reset("mma", problems, mma_policy<>{}, options);
    check_reset("ccsa_quadratic", problems, ccsa_quadratic_policy<>{}, options);
    check_reset("gcmma_move_limit_shrink", problems, alternative::gcmma::move_limit_shrink_policy<>{}, options);
    check_reset("gcmma_raa_augmented", problems, alternative::gcmma::raa_augmented_policy<>{}, options);
    check_reset("gcmma_rho_wval", problems, alternative::gcmma::rho_wval_policy<>{}, options);
}

TEST_CASE("reset_clear equals a fresh solve for the stochastic solvers and the restarting decorator", "[property]")
{
    const solver_options<> options = short_stochastic_options();
    check_reset("cmaes_repair_l2_penalty", bound_quadratic(), seeded(alternative::cmaes::repair_l2_penalty_policy<>{}), options);
    check_reset("cmaes_pwq_reparameterization", bound_quadratic(), seeded(alternative::cmaes::pwq_reparameterization_policy<>{}), options);
    check_reset("cmaes_no_repair_adaptive_penalty", bound_quadratic(), seeded(alternative::cmaes::no_repair_adaptive_penalty_policy<>{}), options);
    check_reset("restarting_cmaes", bound_quadratic(), restarting_cmaes_that_restarts(), options);
    check_reset("isres_nlopt_faithful", inequality_quadratic(), seeded(alternative::isres::nlopt_faithful_policy<>{}), options);
    check_reset("isres_original_argmin", inequality_quadratic(), seeded(alternative::isres::original_argmin_policy<>{}), options);
    check_reset("isres_runarsson_yao_paper", inequality_quadratic(), seeded(alternative::isres::runarsson_yao_paper_policy<>{}), options);
}

TEST_CASE("reset and reset_clear equal a fresh solve for the least-squares and derivative-free solvers", "[property]")
{
    const auto boxed = boxed_least_squares(box_kind::around_start);
    for(const reset_kind kind : {reset_kind::clearing, reset_kind::warm})
    {
        check_reset("lm", unconstrained_least_squares(), lm_policy<>{}, cases::least_squares_options(), kind);
        check_reset("projected_gn", boxed, projected_gn_policy<>{}, cases::least_squares_options(), kind);
        check_reset("projected_gradient_gn", boxed, projected_gradient_gn_policy<>{}, cases::least_squares_options(), kind);
        check_reset("bobyqa", bound_quadratic(), bobyqa_policy<>{}, cases::derivative_free_options(), kind);
        check_reset("cobyla", inequality_quadratic(), cobyla_policy<>{}, cases::derivative_free_options(), kind);
    }
}

TEST_CASE("reset equals a fresh start with its warm set carried over for the gradient bound solvers and the augmented Lagrangian", "[property]")
{
    const solver_options<> options = cases::gradient_options();
    check_warm_reset("lbfgsb", bound_quadratic(), lbfgsb_policy<>{}, options, warm::curvature_pairs);
    check_warm_reset("byrd_lbfgsb", bound_quadratic(), byrd_lbfgsb_policy<>{}, options, warm::curvature_pairs);
    // A multistart solve ends only after all its starts, so its steps are capped far above one run's.
    check_warm_reset("multistart_lbfgsb", bound_quadratic(), multistart_policy<lbfgsb_policy<>>{}, options, warm::of_inner(warm::curvature_pairs), 400);
    check_warm_reset("augmented_lagrangian_lbfgsb", mixed_quadratic(), augmented_lagrangian_policy<lbfgsb_policy<>>{}, options, warm::multipliers_penalty_and_inner_solver);
}

TEST_CASE("reset equals a fresh start with its warm set carried over for the SQP solvers", "[property]")
{
    const solver_options<> options = cases::gradient_options();
    check_warm_reset("kraft_slsqp", mixed_quadratic(), kraft_slsqp_policy<>{}, options, warm::hessian_and_penalty);
    check_warm_reset("filter_slsqp", mixed_quadratic(), filter_slsqp_policy<>{}, options, warm::hessian);
    check_warm_reset("nw_sqp", mixed_quadratic(), nw_sqp_policy<>{}, options, warm::hessian_and_penalty);
    check_warm_reset("filter_nw_sqp", mixed_quadratic(), filter_nw_sqp_policy<>{}, options, warm::hessian_and_penalty);
    check_warm_reset("tr_sqp_fast", mixed_quadratic(), tr_sqp<sqp_mode::fast>{}, options, warm::hessian_and_multipliers);
    check_warm_reset("tr_sqp_accurate", mixed_quadratic(), tr_sqp<sqp_mode::accurate>{}, options, warm::hessian_and_multipliers);
    check_warm_reset("filter_trsqp_fast", mixed_quadratic(), filter_trsqp<sqp_mode::fast>{}, options, warm::hessian_and_multipliers);
    check_warm_reset("filter_trsqp_accurate", mixed_quadratic(), filter_trsqp<sqp_mode::accurate>{}, options, warm::hessian_and_multipliers);
}

TEST_CASE("reset equals a fresh start with its warm set carried over for the moving-asymptote solvers", "[property]")
{
    const auto problems = inequality_quadratic(start_kind::feasible);
    const solver_options<> options = cases::gradient_options();
    check_warm_reset("mma", problems, mma_policy<>{}, options, warm::none);
    check_warm_reset("ccsa_quadratic", problems, ccsa_quadratic_policy<>{}, options, warm::trust_radii);
    check_warm_reset("gcmma_move_limit_shrink", problems, alternative::gcmma::move_limit_shrink_policy<>{}, options, warm::none);
    check_warm_reset("gcmma_raa_augmented", problems, alternative::gcmma::raa_augmented_policy<>{}, options, warm::none);
    check_warm_reset("gcmma_rho_wval", problems, alternative::gcmma::rho_wval_policy<>{}, options, warm::none);
}

TEST_CASE("reset equals a fresh start for the stochastic solvers and the restarting decorator", "[property]")
{
    const solver_options<> options = short_stochastic_options();
    check_warm_reset("cmaes_repair_l2_penalty", bound_quadratic(), seeded(alternative::cmaes::repair_l2_penalty_policy<>{}), options, warm::none);
    check_warm_reset("cmaes_pwq_reparameterization", bound_quadratic(), seeded(alternative::cmaes::pwq_reparameterization_policy<>{}), options, warm::none);
    check_warm_reset("cmaes_no_repair_adaptive_penalty", bound_quadratic(), seeded(alternative::cmaes::no_repair_adaptive_penalty_policy<>{}), options, warm::none);
    check_warm_reset("restarting_cmaes", bound_quadratic(), restarting_cmaes_that_restarts(), options, warm::none, 200);
    check_warm_reset("restarting_cmaes unboxed", unconstrained_quadratic(), restarting_cmaes_that_restarts(), options, warm::none, 200);
    check_warm_reset("isres_nlopt_faithful", inequality_quadratic(), seeded(alternative::isres::nlopt_faithful_policy<>{}), options, warm::none);
    check_warm_reset("isres_original_argmin", inequality_quadratic(), seeded(alternative::isres::original_argmin_policy<>{}), options, warm::none);
    check_warm_reset("isres_runarsson_yao_paper", inequality_quadratic(), seeded(alternative::isres::runarsson_yao_paper_policy<>{}), options, warm::none);
}
