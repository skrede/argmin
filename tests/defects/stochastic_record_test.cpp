#include "cases/case_options.h"

#include "oracle/kkt_point.h"
#include "oracle/kkt_measure.h"

#include "argmin/solver/options.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/solver/alternative/isres/nlopt_faithful_policy.h"
#include "argmin/solver/alternative/isres/original_argmin_policy.h"
#include "argmin/solver/alternative/isres/runarsson_yao_paper_policy.h"

#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <algorithm>

using namespace argmin;

namespace
{

// The lowest objective among the evaluated points within the tolerance, their violation over the rows and the box
// measured by the oracle.
template <typename Problem>
class recording : public Problem
{
public:
    using vector = Eigen::Vector<double, Problem::problem_dimension>;

    explicit recording(double tolerance)
        : m_tolerance{tolerance}
        , m_best_within{std::numeric_limits<double>::infinity()}
    {}

    double value(const vector& x) const
    {
        const double f = Problem::value(x);
        oracle::kkt_legs<double> legs;
        oracle::measure_feasibility(oracle::evaluate_point(static_cast<const Problem&>(*this), x), legs);
        if(std::max({legs.equality_violation, legs.inequality_violation, legs.bound_violation}) <= m_tolerance)
            m_best_within = std::min(m_best_within, f);
        return f;
    }

    double best_within() const { return m_best_within; }

private:
    double m_tolerance;
    mutable double m_best_within;
};

template <typename Policy, typename Problem>
void check_returns_best_evaluated()
{
    const solver_options<> options = cases::stochastic_options();
    const recording<Problem> problem{options.feasibility_tolerance};
    Policy policy{};
    policy.options.seed = cases::stochastic_seed;
    step_budget_solver solver{policy, problem, problem.initial_point(), options};
    const auto result = solver.solve(options);

    REQUIRE(std::isfinite(problem.best_within()));
    CHECK(result.objective_value == problem.best_within());
    CHECK(result.constraint_violation <= options.feasibility_tolerance);
    CHECK(std::isnan(result.gradient_norm));
}

}

TEST_CASE("ISRES nlopt_faithful returns the best point it evaluated within the feasibility tolerance", "[stochastic]")
{
    SECTION("hs071") { check_returns_best_evaluated<alternative::isres::nlopt_faithful_policy<double, 4>, hs071<>>(); }
    SECTION("hs026") { check_returns_best_evaluated<alternative::isres::nlopt_faithful_policy<double, 3>, hs026<>>(); }
}

TEST_CASE("ISRES runarsson_yao_paper returns the best point it evaluated within the feasibility tolerance", "[stochastic]")
{
    SECTION("hs071") { check_returns_best_evaluated<alternative::isres::runarsson_yao_paper_policy<double, 4>, hs071<>>(); }
    SECTION("hs026") { check_returns_best_evaluated<alternative::isres::runarsson_yao_paper_policy<double, 3>, hs026<>>(); }
}

TEST_CASE("ISRES original_argmin returns the best point it evaluated within the feasibility tolerance", "[stochastic]")
{
    SECTION("hs071") { check_returns_best_evaluated<alternative::isres::original_argmin_policy<double, 4>, hs071<>>(); }
    SECTION("hs026") { check_returns_best_evaluated<alternative::isres::original_argmin_policy<double, 3>, hs026<>>(); }
}
