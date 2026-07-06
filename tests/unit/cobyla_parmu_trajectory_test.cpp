// COBYLA merit-penalty (parmu) monotonicity pin.
//
// Powell's COBYLA merit function is  phi(x) = f(x) + parmu * RESMAX(x),
// where RESMAX is the MAXIMUM constraint violation. The penalty parmu is
// raised only through  barmu = (predicted objective INCREASE along the
// step) / (predicted reduction in the maximum violation) -- i.e. only when
// driving toward feasibility costs objective (Powell 1994, "A direct search
// optimization method that models the objective and constraint functions by
// linear interpolation", section 5). It follows that a step which reduces
// BOTH the objective and the maximum violation carries a non-positive
// predicted objective increase and therefore must NOT raise parmu.
//
// The pre-rewrite driver inverted this: it computed the raise factor from the
// predicted objective DECREASE (prerec = -g.d) rather than the predicted
// objective increase, so it inflated parmu on precisely the objective-and-
// feasibility-improving steps where a faithful COBYLA leaves it untouched
// (worst observed growth factor 6.14e10 on HS024). The Powell-faithful driver
// now computes barmu = sum / prerec, where sum is the k=(m+1) term of the
// linear-model reduction (sum = g.d, the predicted objective CHANGE) and
// prerec = resmax_old - resnew >= 0 (the predicted reduction in the MAXIMUM
// violation). parmu is raised (to 2*barmu) only when parmu < 1.5*barmu, which
// requires barmu > 0, i.e. sum > 0: a predicted objective INCREASE. A step that
// improves the objective has g.d <= 0, so barmu <= 0 and the raise never fires.
//
// Hand-computed witness (HS048, first joint-improving step): the objective
// falls 62 -> 51.8784 and the maximum violation falls 2 -> 8.9e-16 while parmu
// stays 0. Across HS048 that run takes 31 joint-improving steps and grows parmu
// on none of them; the same holds across HS050/051/024 (the aggregate the CHECK
// below asserts).
//
// This pin drives the live policy on the pinned Hock-Schittkowski problems,
// records the (parmu, objective, max-violation) stream, and asserts that no
// joint-improving step grew parmu. The checked-in NLopt (f2c'd Powell)
// reference optima in oracles/cobyla_parmu_trajectory.csv are the bar the
// corrected driver's convergence is pinned against separately; here they anchor
// the problem set and confirm the oracle plumbing resolves.

#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/options.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <map>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

using Catch::Approx;
using namespace argmin;

namespace
{

// Minimal oracle CSV reader: '#' lines are comments; data lines are
// "name,v1,v2,...". Returns name -> values. Mirrors the reader used by the
// CMA-ES and MMA oracle pins.
std::map<std::string, std::vector<double>> load_oracle(const std::string& path)
{
    std::map<std::string, std::vector<double>> rows;
    std::ifstream in(path);
    if(!in.is_open())
        return rows;
    std::string line;
    while(std::getline(in, line))
    {
        if(line.empty() || line.front() == '#')
            continue;
        std::istringstream ss(line);
        std::string name;
        std::getline(ss, name, ',');
        std::vector<double> vals;
        std::string tok;
        while(std::getline(ss, tok, ','))
            vals.push_back(std::stod(tok));
        rows.emplace(name, std::move(vals));
    }
    return rows;
}

// Maximum constraint violation in the argmin sense: equalities contribute
// |c_i|, inequalities (g >= 0 feasible) contribute max(0, -g_i).
template <typename Problem>
double max_violation(const Problem& problem, const Eigen::VectorXd& x)
{
    const int neq = problem.num_equality();
    const int nineq = problem.num_inequality();
    if(neq + nineq == 0)
        return 0.0;
    Eigen::VectorXd c(neq + nineq);
    problem.constraints(x, c);
    double v = 0.0;
    for(int i = 0; i < neq; ++i)
        v = std::max(v, std::abs(c[i]));
    for(int i = neq; i < neq + nineq; ++i)
        v = std::max(v, std::max(0.0, -c[i]));
    return v;
}

// Drive the COBYLA policy directly, accumulating joint-improving steps and
// the count on which parmu grew. A "joint-improving" step is one whose
// accepted incumbent strictly lowers the objective while not increasing the
// maximum violation.
template <typename Problem>
void accumulate(int max_steps, int& joint_steps, int& parmu_grew_on_joint,
                double& worst_growth_factor)
{
    Problem problem;
    Eigen::VectorXd x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = max_steps;

    cobyla_policy pol;
    auto s = pol.init(problem, x0, opts);

    constexpr double tol = 1e-12;
    for(int it = 0; it < max_steps; ++it)
    {
        const double parmu_before = s.parmu;
        const double f_before = s.objective_value;
        const double v_before = max_violation(problem, s.x);

        pol.step(s);

        const double parmu_after = s.parmu;
        const double f_after = s.objective_value;
        const double v_after = max_violation(problem, s.x);

        const bool f_down = f_after < f_before - tol;
        const bool v_not_up = v_after <= v_before + tol;
        if(f_down && v_not_up)
        {
            ++joint_steps;
            if(parmu_after > parmu_before * (1.0 + 1e-9) + tol)
            {
                ++parmu_grew_on_joint;
                const double factor = (parmu_before > tol)
                                          ? parmu_after / parmu_before
                                          : parmu_after;
                worst_growth_factor = std::max(worst_growth_factor, factor);
            }
        }
    }
}

}

// The corrected merit direction is the bar: no joint objective-and-feasibility
// improving step raises parmu. This passed once the driver adopted Powell's
// barmu = sum/prerec direction (see the witness in the file header).
TEST_CASE("cobyla parmu never grows on a joint objective-and-feasibility "
          "improving step",
          "[cobyla][oracle-pin]")
{
    // The oracle anchors the problem set and confirms the checked-in NLopt
    // reference optima resolve next to the test binary.
    const auto oracle = load_oracle("oracles/cobyla_parmu_trajectory.csv");
    REQUIRE(oracle.contains("hs024_fopt"));
    REQUIRE(oracle.contains("hs048_fopt"));
    REQUIRE(oracle.contains("hs051_fopt"));

    int joint_steps = 0;
    int parmu_grew_on_joint = 0;
    double worst_growth = 0.0;

    accumulate<hs048<>>(150, joint_steps, parmu_grew_on_joint, worst_growth);
    accumulate<hs050<>>(150, joint_steps, parmu_grew_on_joint, worst_growth);
    accumulate<hs051<>>(150, joint_steps, parmu_grew_on_joint, worst_growth);
    accumulate<hs024<>>(150, joint_steps, parmu_grew_on_joint, worst_growth);

    // The instrument is only meaningful if the driver actually took
    // joint-improving steps.
    REQUIRE(joint_steps > 0);

    INFO("joint-improving steps: " << joint_steps
         << "; parmu grew on: " << parmu_grew_on_joint
         << "; worst growth factor: " << worst_growth);
    CHECK(parmu_grew_on_joint == 0);
}
