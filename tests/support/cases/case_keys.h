#ifndef HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_CASE_KEYS_H
#define HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_CASE_KEYS_H

#include <array>
#include <algorithm>
#include <string_view>

namespace argmin::cases
{

// One key per public solver. The cmaes_policy, gcmma_policy and isres_policy aliases are covered through
// the alternative each one names, and the inventory scan maps every solver header on disk to these keys.
inline constexpr std::array<std::string_view, 32> case_keys{"lbfgsb",
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

static_assert(case_keys.size() == 32);

inline bool is_solver_key(std::string_view key)
{
    return std::ranges::find(case_keys, key) != case_keys.end();
}

}

#endif
