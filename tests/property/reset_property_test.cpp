#include "reset_support.h"
#include "generated_problems.h"
#include "generated_least_squares.h"

#include "cases/case_options.h"

#include "argmin/solver/sqp_mode.h"
#include "argmin/solver/lm_policy.h"
#include "argmin/solver/mma_policy.h"
#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/multistart_policy.h"
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

#include "argmin/types.h"

#include <catch2/catch_test_macros.hpp>

using namespace argmin;

namespace
{

template <sqp_mode Mode>
using tr_sqp = tr_sqp_policy<double, dynamic_dimension, Mode>;

template <sqp_mode Mode>
using filter_trsqp = filter_trsqp_policy<double, dynamic_dimension, Mode>;

}

// Excluded from both reset properties:
// - cmaes_repair_l2_penalty, cmaes_pwq_reparameterization, cmaes_no_repair_adaptive_penalty, restarting_cmaes,
//   isres_nlopt_faithful, isres_original_argmin and isres_runarsson_yao_paper: the random stream continues across
//   a reset, so a reset run draws other samples than a fresh run with the same seed, and the CMA-ES reset_clear
//   draws a new mean inside the box on purpose.
// - dense_admm_qp, sparse_admm_qp and dense_active_set_qp: they have no reset; a re-solve warm-starts by design.
// Excluded from the warm reset() property: every other key outside the derivative-free and least-squares families,
// because their reset() keeps algorithm state by contract (curvature pairs, quasi-Newton matrices, trust radii and
// asymptotes), and the moving-asymptote family also keeps stale asymptotes across it.

TEST_CASE("reset_clear equals a fresh solve for the gradient bound solvers and the augmented Lagrangian", "[property]")
{
    property::check_reset("lbfgsb", property::bound_quadratic(), lbfgsb_policy<>{}, cases::gradient_options());
    property::check_reset("byrd_lbfgsb", property::bound_quadratic(), byrd_lbfgsb_policy<>{}, cases::gradient_options());
    property::check_reset("multistart_lbfgsb", property::bound_quadratic(), multistart_policy<lbfgsb_policy<>>{}, cases::gradient_options());
    property::check_reset("augmented_lagrangian_lbfgsb", property::mixed_quadratic(), augmented_lagrangian_policy<lbfgsb_policy<>>{}, cases::gradient_options());
}

TEST_CASE("reset_clear equals a fresh solve for the SQP solvers", "[property]")
{
    property::check_reset("kraft_slsqp", property::mixed_quadratic(), kraft_slsqp_policy<>{}, cases::gradient_options());
    property::check_reset("filter_slsqp", property::mixed_quadratic(), filter_slsqp_policy<>{}, cases::gradient_options());
    property::check_reset("nw_sqp", property::mixed_quadratic(), nw_sqp_policy<>{}, cases::gradient_options());
    property::check_reset("filter_nw_sqp", property::mixed_quadratic(), filter_nw_sqp_policy<>{}, cases::gradient_options());
    property::check_reset("tr_sqp_fast", property::mixed_quadratic(), tr_sqp<sqp_mode::fast>{}, cases::gradient_options());
    property::check_reset("tr_sqp_accurate", property::mixed_quadratic(), tr_sqp<sqp_mode::accurate>{}, cases::gradient_options());
    property::check_reset("filter_trsqp_fast", property::mixed_quadratic(), filter_trsqp<sqp_mode::fast>{}, cases::gradient_options());
    property::check_reset("filter_trsqp_accurate", property::mixed_quadratic(), filter_trsqp<sqp_mode::accurate>{}, cases::gradient_options());
}

TEST_CASE("reset_clear equals a fresh solve for the moving-asymptote solvers", "[property]")
{
    // Bit equality needs no convergence and gcmma move_limit_shrink crawls through a large budget, so these
    // runs stop at 200 iterations.
    const auto problems = property::inequality_quadratic(property::start_kind::feasible);
    const solver_options<> options = cases::gradient_options();
    property::check_reset("mma", problems, mma_policy<>{}, options);
    property::check_reset("ccsa_quadratic", problems, ccsa_quadratic_policy<>{}, options);
    property::check_reset("gcmma_move_limit_shrink", problems, alternative::gcmma::move_limit_shrink_policy<>{}, options);
    property::check_reset("gcmma_raa_augmented", problems, alternative::gcmma::raa_augmented_policy<>{}, options);
    property::check_reset("gcmma_rho_wval", problems, alternative::gcmma::rho_wval_policy<>{}, options);
}

TEST_CASE("reset and reset_clear equal a fresh solve for the least-squares and derivative-free solvers", "[property]")
{
    const auto boxed = property::boxed_least_squares(property::box_kind::around_start);
    for(const property::reset_kind kind : {property::reset_kind::clearing, property::reset_kind::warm})
    {
        property::check_reset("lm", property::unconstrained_least_squares(), lm_policy<>{}, cases::least_squares_options(), kind);
        property::check_reset("projected_gn", boxed, projected_gn_policy<>{}, cases::least_squares_options(), kind);
        property::check_reset("projected_gradient_gn", boxed, projected_gradient_gn_policy<>{}, cases::least_squares_options(), kind);
        property::check_reset("bobyqa", property::bound_quadratic(), bobyqa_policy<>{}, cases::derivative_free_options(), kind);
        property::check_reset("cobyla", property::inequality_quadratic(), cobyla_policy<>{}, cases::derivative_free_options(), kind);
    }
}
