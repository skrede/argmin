#include "cases/qp_cases.h"
#include "cases/case_keys.h"
#include "cases/sqp_cases.h"
#include "cases/gradient_cases.h"
#include "cases/stochastic_cases.h"
#include "cases/least_squares_cases.h"
#include "cases/derivative_free_cases.h"
#include "cases/moving_asymptote_cases.h"
#include "cases/trust_region_sqp_cases.h"

#include <catch2/catch_test_macros.hpp>

#include <vector>
#include <algorithm>
#include <string_view>

using namespace argmin;

namespace
{

template <typename Keys>
void append(std::vector<std::string_view>& all, const Keys& keys)
{
    all.insert(all.end(), keys.begin(), keys.end());
}

std::vector<std::string_view> family_keys()
{
    std::vector<std::string_view> all;
    append(all, cases::gradient_keys);
    append(all, cases::least_squares_keys);
    append(all, cases::sqp_keys);
    append(all, cases::trust_region_sqp_keys);
    append(all, cases::moving_asymptote_keys);
    append(all, cases::derivative_free_keys);
    append(all, cases::stochastic_keys);
    append(all, cases::qp_keys);
    return all;
}

}

TEST_CASE("the family case tables together cover every solver key exactly once", "[oracle]")
{
    std::vector<std::string_view> covered = family_keys();
    std::vector<std::string_view> expected(cases::case_keys.begin(), cases::case_keys.end());
    std::ranges::sort(covered);
    std::ranges::sort(expected);
    CHECK(covered == expected);
}
