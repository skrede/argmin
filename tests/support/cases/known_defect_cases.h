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
inline constexpr std::array<known_defect, 21> known_defect_cases{{
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
    {"cobyla",
     "bound",
     "from the default initial radius the linear models stall near the saddle of the Wood function at f = 7.877, so the run "
     "ends max_iterations far from stationarity"},
    {"cmaes_pwq_reparameterization", "bound", "on hs005 the run reports converged after 65 generations and returns the initial point, which is not stationary"},
    {"cmaes_no_repair_adaptive_penalty", "bound", "on hs005 the step size collapses at a non-stationary interior point and the run reports converged"},
    {"isres_nlopt_faithful",
     "bound",
     "the step-size collapse ends the run converged on the Wood function with stationarity about 150 times the gradient "
     "tolerance"},
    {"isres_nlopt_faithful", "inequality", "the run reports converged at a point of hs076 that violates an inequality row and is far from stationary"},
    {"isres_nlopt_faithful", "mixed", "the run reports converged at a point of hs071 that violates both constraint kinds and is far from stationary"},
    {"isres_original_argmin", "bound", "the run ends max_iterations on the Wood function far from stationarity"},
    {"isres_original_argmin", "inequality", "the run reports converged after 34 generations on hs076 far from stationarity"},
    {"isres_original_argmin",
     "equality",
     "the best point never moves from the best of the initial population, so the run ends max_iterations at an "
     "infeasible point of hs028"},
    {"isres_original_argmin",
     "mixed",
     "the best point never moves from the best of the initial population, so the run ends max_iterations at an "
     "infeasible point of hs071"},
    {"isres_runarsson_yao_paper", "bound", "the run ends max_iterations on the Wood function far from stationarity"},
    {"isres_runarsson_yao_paper", "inequality", "the run ends ftol_reached on hs076 with stationarity far above the tolerance"},
    {"isres_runarsson_yao_paper",
     "equality",
     "the best point never moves from the best of the initial population, so the run ends ftol_reached at an infeasible "
     "point of hs028"},
    {"isres_runarsson_yao_paper",
     "mixed",
     "the best point never moves from the best of the initial population, so the run ends ftol_reached at an infeasible "
     "point of hs071"},
}};

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
