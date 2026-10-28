#ifndef HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_CASE_VERDICT_H
#define HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_CASE_VERDICT_H

#include "oracle_tolerances.h"

#include "cases/case_options.h"

#include "oracle/result_oracle.h"

#include <catch2/catch_test_macros.hpp>

#include <optional>
#include <string_view>

namespace argmin::cases
{

class selecting_visitor
{
public:
    selecting_visitor(std::string_view key, std::string_view problem_class) : key_(key), problem_class_(problem_class) {}

    template <typename Problem, typename Options, typename MakeSolver>
    void operator()(const case_id& id, const Problem& problem, const Options& options, const MakeSolver& make_solver)
    {
        if(id.key != key_ || id.problem_class != problem_class_)
            return;
        auto solver = make_solver();
        const auto result = solver.solve(options);
        verdict_.emplace(oracle::judge(problem, result, oracle::tolerances_of(options)));
    }

    const std::optional<oracle::verdict<double>>& verdict() const { return verdict_; }

private:
    std::string_view key_;
    std::string_view problem_class_;
    std::optional<oracle::verdict<double>> verdict_;
};

// An expected-fail case solves the very case the coverage skips, with its problem, options and seed.
template <typename VisitFamily>
std::optional<oracle::verdict<double>> judge_case(VisitFamily visit_family, std::string_view key, std::string_view problem_class)
{
    selecting_visitor selector(key, problem_class);
    visit_family(selector);
    return selector.verdict();
}

// A case missing from the table leaves no verdict and asserts nothing, so an expected-fail case built on
// this check reports it as an unexpected pass.
template <typename VisitFamily>
void check_case_claim(VisitFamily visit_family, std::string_view key, std::string_view problem_class)
{
    const std::optional<oracle::verdict<double>> v = judge_case(visit_family, key, problem_class);
    if(!v.has_value())
        return;
    INFO(key << ' ' << problem_class);
    CAPTURE(v->legs.stationarity, v->failing);
    CHECK(v->claimed != oracle::claim::none);
    CHECK(v->ok);
}

}

#endif
