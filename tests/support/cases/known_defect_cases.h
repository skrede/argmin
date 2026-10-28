#ifndef HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_KNOWN_DEFECT_CASES_H
#define HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_KNOWN_DEFECT_CASES_H

#include "cases/case_options.h"

#include <array>
#include <algorithm>
#include <string_view>

namespace argmin::cases
{

struct known_defect
{
    std::string_view key;
    std::string_view problem_class;
    std::string_view reason;
};

// Each entry is skipped by the oracle coverage and asserted by an expected-fail case under tests/defects.
inline constexpr std::array<known_defect, 7> known_defect_cases{{
    {"projected_gradient_gn",
     "bound",
     "projecting the full Gauss-Newton step onto the box crawls along the free variable once a bound is active, so the "
     "bounded Rosenbrock case runs out of iterations far from stationarity"},
    {"tr_sqp_fast",
     "inequality",
     "the trust region collapses at the hs076 optimum, so the run ends trust_region_step_rejected and claims nothing at a "
     "point that meets the converged tolerance"},
    {"tr_sqp_fast",
     "mixed",
     "the trust region collapses short of the hs071 solution, so the run ends trust_region_step_rejected with "
     "stationarity forty times the gradient tolerance"},
    {"tr_sqp_accurate",
     "mixed",
     "the trust region collapses short of the hs071 solution, so the run ends trust_region_step_rejected with "
     "stationarity sixty times the gradient tolerance"},
    {"mma",
     "bound",
     "with zero constraint rows the iterate stops changing at a non-stationary point of the Wood function, so the run "
     "ends max_iterations"},
    {"gcmma_move_limit_shrink",
     "bound",
     "the shrinking move limit slows progress on the Wood function so far that the run ends max_iterations short of "
     "stationarity"},
    {"gcmma_move_limit_shrink",
     "inequality",
     "the shrinking move limit stalls progress on hs076, which the other variants solve in under 30 iterations, so the "
     "run ends max_iterations far from stationarity"},
}};

inline constexpr std::array<std::string_view, 32> solver_keys{"lbfgsb",
                                                              "byrd_lbfgsb",
                                                              "kraft_slsqp",
                                                              "nw_sqp",
                                                              "filter_slsqp",
                                                              "filter_nw_sqp",
                                                              "tr_sqp_fast",
                                                              "tr_sqp_accurate",
                                                              "filter_trsqp_fast",
                                                              "filter_trsqp_accurate",
                                                              "lm",
                                                              "projected_gn",
                                                              "projected_gradient_gn",
                                                              "mma",
                                                              "ccsa_quadratic",
                                                              "gcmma_move_limit_shrink",
                                                              "gcmma_raa_augmented",
                                                              "gcmma_rho_wval",
                                                              "bobyqa",
                                                              "cobyla",
                                                              "cmaes_repair_l2_penalty",
                                                              "cmaes_pwq_reparameterization",
                                                              "cmaes_no_repair_adaptive_penalty",
                                                              "isres_nlopt_faithful",
                                                              "isres_original_argmin",
                                                              "isres_runarsson_yao_paper",
                                                              "augmented_lagrangian_lbfgsb",
                                                              "multistart_lbfgsb",
                                                              "restarting_cmaes",
                                                              "dense_admm_qp",
                                                              "sparse_admm_qp",
                                                              "dense_active_set_qp"};

inline bool is_solver_key(std::string_view key)
{
    return std::ranges::find(solver_keys, key) != solver_keys.end();
}

inline bool same_case(const known_defect& defect, const case_id& id)
{
    return defect.key == id.key && defect.problem_class == id.problem_class;
}

inline bool is_known_defect(const case_id& id)
{
    return std::ranges::any_of(known_defect_cases, [&](const known_defect& defect) { return same_case(defect, id); });
}

}

#endif
