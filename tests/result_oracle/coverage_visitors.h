#ifndef HPP_GUARD_ARGMIN_TESTS_RESULT_ORACLE_COVERAGE_VISITORS_H
#define HPP_GUARD_ARGMIN_TESTS_RESULT_ORACLE_COVERAGE_VISITORS_H

#include "margin_table.h"
#include "oracle_tolerances.h"

#include "cases/case_options.h"
#include "cases/known_defect_cases.h"

#include "oracle/result_oracle.h"

#include <catch2/catch_test_macros.hpp>

#include <vector>
#include <algorithm>
#include <string_view>

namespace argmin::cases
{

class judging_visitor
{
public:
    explicit judging_visitor(margin_table& margins) : margins_(margins) {}

    template <typename Problem, typename Options, typename MakeSolver>
    void operator()(const case_id& id, const Problem& problem, const Options& options, const MakeSolver& make_solver)
    {
        if(is_known_defect(id))
            return;
        auto solver = make_solver();
        const auto result = solver.solve(options);
        const oracle::tolerances<double> tol = oracle::tolerances_of(options);
        const oracle::verdict<double> v = oracle::judge(problem, result, tol);
        margins_.record(id, result.status, v, tol);
        INFO(id.key << ' ' << id.problem_class << ' ' << id.problem);
        CAPTURE(static_cast<int>(result.status), v.failing, v.legs.stationarity, v.legs.equality_violation, v.legs.inequality_violation, v.legs.bound_violation, v.legs.complementarity, v.legs.dual_sign_breach);
        CHECK(v.claimed != oracle::claim::none);
        CHECK(v.ok);
    }

private:
    margin_table& margins_;
};

struct collecting_visitor
{
    template <typename Problem, typename Options, typename MakeSolver>
    void operator()(const case_id& id, const Problem&, const Options&, const MakeSolver&)
    {
        visited.push_back(id);
    }

    std::vector<case_id> visited;
};

inline bool visits_key(const std::vector<case_id>& visited, std::string_view key)
{
    return std::ranges::any_of(visited, [&](const case_id& id) { return id.key == key; });
}

// An entry naming one of this family's keys must match a visited case, so a renamed or removed case
// cannot leave its defect listed and silently skipped.
template <typename VisitFamily>
void check_known_defects_match(VisitFamily visit_family)
{
    collecting_visitor collector;
    visit_family(collector);
    for(const case_id& id : collector.visited)
        CHECK(is_solver_key(id.key));
    for(const known_defect& defect : known_defect_cases)
    {
        INFO(defect.key << ' ' << defect.problem_class);
        CHECK(is_solver_key(defect.key));
        if(visits_key(collector.visited, defect.key))
            CHECK(std::ranges::any_of(collector.visited, [&](const case_id& id) { return same_case(defect, id); }));
    }
}

}

#endif
