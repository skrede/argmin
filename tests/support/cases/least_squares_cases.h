#ifndef HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_LEAST_SQUARES_CASES_H
#define HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_LEAST_SQUARES_CASES_H

#include "cases/case_options.h"
#include "cases/case_problems.h"

#include "argmin/solver/lm_policy.h"
#include "argmin/solver/projected_gn_policy.h"
#include "argmin/solver/projected_gradient_gn_policy.h"

#include "argmin/test_functions/small_dense.h"

#include <string_view>

namespace argmin::cases
{

// Levenberg-Marquardt never reads a box, so a bounded problem is not one of its cases.
template <typename Visitor>
void visit_lm(Visitor& visitor)
{
    const rosenbrock_ls2 rosenbrock{};
    const sd_ls012<> extended{};
    visit_case(visitor, case_id{"lm", "unconstrained", "rosenbrock_ls2"}, lm_policy<>{}, rosenbrock, least_squares_options());
    visit_case(visitor, case_id{"lm", "unconstrained", "sd_ls012"}, lm_policy<>{}, extended, least_squares_options());
}

template <typename Visitor, typename Policy>
void visit_projected(Visitor& visitor, std::string_view key, const Policy& policy)
{
    const infinite_box<rosenbrock_ls2> rosenbrock{};
    const infinite_box<sd_ls012<>> extended{};
    const bounded_rosenbrock_ls2 bounded{};
    visit_case(visitor, case_id{key, "unconstrained", "rosenbrock_ls2"}, policy, rosenbrock, least_squares_options());
    visit_case(visitor, case_id{key, "unconstrained", "sd_ls012"}, policy, extended, least_squares_options());
    visit_case(visitor, case_id{key, "bound", "bounded_rosenbrock_ls2"}, policy, bounded, least_squares_options());
}

template <typename Visitor>
void visit_least_squares_cases(Visitor& visitor)
{
    visit_lm(visitor);
    visit_projected(visitor, "projected_gn", projected_gn_policy<>{});
    visit_projected(visitor, "projected_gradient_gn", projected_gradient_gn_policy<>{});
}

}

#endif
