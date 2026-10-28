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
inline constexpr std::array<known_defect, 1> known_defect_cases{{
    {"projected_gradient_gn",
     "bound",
     "projecting the full Gauss-Newton step onto the box crawls along the free variable once a bound is active, so the "
     "bounded Rosenbrock case runs out of iterations far from stationarity"},
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
