#include "coverage_visitors.h"

#include "margin_table.h"
#include "oracle_tolerances.h"

#include "cases/qp_cases.h"
#include "cases/known_defect_cases.h"

#include "oracle/qp_oracle.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <algorithm>

using namespace argmin;

namespace
{

// The ratios are against the unscaled tolerances, so each reads as the safety factor the case would need.
void record_qp_margins(cases::margin_table& margins, const cases::case_id& id, const cases::qp_case& c, const qp_result<double>& result, const oracle::qp_tolerances<double>& tol)
{
    const Eigen::VectorXd x = result.x;
    const Eigen::VectorXd ax = c.A * x;
    const Eigen::VectorXd px = c.P * x;
    const Eigen::VectorXd aty = c.A.transpose() * result.y;
    const double dual_tol = tol.absolute + tol.relative * std::max({oracle::inf_norm(px), oracle::inf_norm(aty), oracle::inf_norm(c.q)});
    const double primal_tol = tol.absolute + tol.relative * oracle::inf_norm(ax);
    const std::int32_t status = static_cast<std::int32_t>(result.status);
    margins.record_leg(id, status, "stationarity", oracle::inf_norm(px + c.q + aty), dual_tol);
    margins.record_leg(id, status, "primal_violation", oracle::inf_norm((c.l - ax).cwiseMax(ax - c.u).cwiseMax(0.0)), primal_tol);
}

bool makes_claim(qp_solve_status status)
{
    return status != qp_solve_status::solved_inaccurate && status != qp_solve_status::max_iterations;
}

class qp_judging_visitor
{
public:
    explicit qp_judging_visitor(cases::margin_table& margins) : margins_(margins) {}

    template <typename Options, typename MakeSolver>
    void operator()(const cases::case_id& id, const cases::qp_case& c, const Options& options, const MakeSolver& make_solver)
    {
        if(cases::is_known_defect(id))
            return;
        auto solver = make_solver();
        const auto result = cases::solve_qp(solver, c, options);
        INFO(id.key << ' ' << id.problem_class << ' ' << id.problem);
        CHECK(result.has_value());
        if(!result.has_value())
            return;
        const oracle::qp_tolerances<double> tol = oracle::qp_tolerances_of(options);
        record_qp_margins(margins_, id, c, *result, tol);
        const oracle::qp_verdict v = oracle::judge_qp(c.P, c.q, c.A, c.l, c.u, *result, tol);
        CAPTURE(static_cast<int>(result->status), v.failing);
        CHECK(makes_claim(result->status));
        CHECK(v.ok);
    }

private:
    cases::margin_table& margins_;
};

}

TEST_CASE("oracle accepts every QP solver result in the case table", "[oracle]")
{
    cases::margin_table margins("qp");
    qp_judging_visitor judge(margins);
    cases::visit_qp_cases(judge);
}

TEST_CASE("every listed QP known defect matches a case in the table", "[oracle]")
{
    cases::check_known_defects_match([](auto& visitor) { cases::visit_qp_cases(visitor); });
}

TEST_CASE("the QP case table visits exactly its keys", "[oracle]")
{
    cases::check_visits_exactly([](auto& visitor) { cases::visit_qp_cases(visitor); }, cases::qp_keys);
}
