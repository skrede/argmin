// Sweep harness for the MMA/GCMMA/CCSA bounded-dual elastics scale knob
// (options_type::dual_bound_scale).
//
// The bounded-dual elastics (Svanberg 2002 relaxed subproblem, a_i = 0
// instance) box each constraint multiplier at
//   c_i = dual_bound_scale * max(|g_i(x0)|, 1)
// so an inequality-infeasible outer iterate cannot make the subproblem
// infeasible and drive the Lagrange dual unbounded (which otherwise burns
// the inner solver's iteration cap on ever-growing multipliers).
//
// The scale multiplier is empirical: c_i must be large enough that no
// false subproblem-infeasibility is reported (the true multiplier mu_i*
// is never clipped) yet small enough to keep the boxed dual well-
// conditioned. This harness sweeps dual_bound_scale over the grid
//   {0.1, 1, 10, 1000 (Svanberg mmasub default magnitude), 10000}
// plus an unbounded control (+infinity, the pre-elastic classic dual),
// each scaled by the per-constraint reference max(|g_i(x0)|, 1).
//
// Problem set: inequality-infeasible-start problems (x0 violates one or
// more inequalities), the case the elastics exist to handle -- HS024 from
// an infeasible interior point, and two synthetic infeasible-start cells.
//
// Metric (feasibility-gated): for each cell/scale, run the policy by
// manual init+step so the policy state (dual multipliers, working bounds)
// is observed directly. Record
//   - reached_feasible + outer iterations to first feasibility,
//   - final objective and its error vs the known optimum,
//   - max over the run of the absolute multiplier norm |y|_inf (the
//     unbounded control blows this up -- the DoS signature),
//   - max over the run of the saturation ratio max_i(y_i / c_i) (whether
//     the elastic engaged, and whether it stays pinned at the bound).
// A good scale reaches feasibility in few outer iterations with a bounded,
// well-conditioned multiplier (saturation ratio well below 1 at the
// solution) and no clipping of the true multiplier (which shows up as a
// failure to reach feasibility or a wrong optimum).
//
// Output: a JSON document (default mma_elastic_dual_sweep.json in CWD,
// override with --output PATH) plus a per-row stdout table. Non-ctest,
// one-shot, read-only on argmin.
//
// The shipped dual_bound_scale default is set from this data and re-locked
// only after the upstream separable-approximation math has settled.
//
// Reference: Svanberg (2002) "A class of globally convergent optimization
//            methods based on conservative convex separable
//            approximations", SIAM J. Optim. 12(2):555-573, Section 2
//            (the relaxed subproblem). Hock and Schittkowski (1981).

#include "argmin/solver/ccsa_quadratic_policy.h"
#include "argmin/solver/mma_policy.h"
#include "argmin/solver/alternative/gcmma/rho_wval_policy.h"
#include "argmin/solver/options.h"
#include "argmin/test_functions/hock_schittkowski.h"
#include "argmin/types.h"

#include <Eigen/Core>

#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <algorithm>

namespace
{

using argmin::solver_options;

// Synthetic infeasible-start cell A: min (x0-2)^2 + (x1-2)^2 subject to
// the single inequality x0 + x1 <= 1 (argmin convention c0 = 1 - x0 - x1
// >= 0), box [-10, 10]^2. Start x0 = (5, 5) is deeply infeasible
// (c0 = -9). The optimum is the projection of (2, 2) onto x0 + x1 = 1:
// x* = (0.5, 0.5), f* = 4.5.
struct halfplane_cell
{
    static constexpr int problem_dimension = 2;
    int dimension() const { return 2; }
    int num_equality() const { return 0; }
    int num_inequality() const { return 1; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        const double a = x[0] - 2.0, b = x[1] - 2.0;
        return a * a + b * b;
    }
    void gradient(const Eigen::Vector<double, 2>& x,
                  Eigen::Vector<double, 2>& g) const
    {
        g[0] = 2.0 * (x[0] - 2.0);
        g[1] = 2.0 * (x[1] - 2.0);
    }
    void constraints(const Eigen::Vector<double, 2>& x, auto& c) const
    {
        c[0] = 1.0 - x[0] - x[1];
    }
    void constraint_jacobian(const Eigen::Vector<double, 2>&, auto& J) const
    {
        J(0, 0) = -1.0; J(0, 1) = -1.0;
    }
    Eigen::Vector<double, 2> lower_bounds() const
    {
        return Eigen::Vector<double, 2>::Constant(2, -10.0);
    }
    Eigen::Vector<double, 2> upper_bounds() const
    {
        return Eigen::Vector<double, 2>::Constant(2, 10.0);
    }
    Eigen::Vector<double, 2> initial_point() const
    {
        return Eigen::Vector<double, 2>{{5.0, 5.0}};
    }
    double optimal_value() const { return 4.5; }
};

// Synthetic infeasible-start cell B: min x0^2 + x1^2 subject to x0 >= 3
// (argmin convention c0 = x0 - 3 >= 0), box [-10, 10]^2. Start x0 =
// (0, 0) violates the constraint (c0 = -3). The optimum is x* = (3, 0),
// f* = 9.
struct halfspace_cell
{
    static constexpr int problem_dimension = 2;
    int dimension() const { return 2; }
    int num_equality() const { return 0; }
    int num_inequality() const { return 1; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        return x[0] * x[0] + x[1] * x[1];
    }
    void gradient(const Eigen::Vector<double, 2>& x,
                  Eigen::Vector<double, 2>& g) const
    {
        g[0] = 2.0 * x[0];
        g[1] = 2.0 * x[1];
    }
    void constraints(const Eigen::Vector<double, 2>& x, auto& c) const
    {
        c[0] = x[0] - 3.0;
    }
    void constraint_jacobian(const Eigen::Vector<double, 2>&, auto& J) const
    {
        J(0, 0) = 1.0; J(0, 1) = 0.0;
    }
    Eigen::Vector<double, 2> lower_bounds() const
    {
        return Eigen::Vector<double, 2>::Constant(2, -10.0);
    }
    Eigen::Vector<double, 2> upper_bounds() const
    {
        return Eigen::Vector<double, 2>::Constant(2, 10.0);
    }
    Eigen::Vector<double, 2> initial_point() const
    {
        return Eigen::Vector<double, 2>{{0.0, 0.0}};
    }
    double optimal_value() const { return 9.0; }
};

// HS024 from an inequality-infeasible interior start x0 = (0.5, 4.0):
// c[0] = 0.5/sqrt(3) - 4 = -3.71 (violated), c[2] = 6 - 0.5 - sqrt(3)*4
// = -1.43 (violated). Optimum x* = (3, sqrt(3)), f* = -1.
struct hs024_infeasible : argmin::hs024<>
{
    Eigen::Vector<double, 2> initial_point() const
    {
        return Eigen::Vector<double, 2>{{0.5, 4.0}};
    }
};

struct cell_result
{
    std::string problem;
    std::string policy;
    double scale{};
    bool reached_feasible{};
    int iters_to_feasible{};
    double f_final{};
    double f_err{};
    double max_mult_inf{};    // max over run of |y|_inf
    double max_sat_ratio{};   // max over run of max_i(y_i / c_i)
};

constexpr int kMaxIters = 500;
constexpr double kFeasTol = 1e-6;

// Manual init + step loop so the policy dual state (y_dual, c_dual) is
// observed. Returns the feasibility-gated cell metrics.
template <typename Policy, typename Problem>
cell_result run_cell(const std::string& pname, const std::string& polname,
                     Policy policy, const Problem& problem, double scale)
{
    constexpr int N = Problem::problem_dimension;
    Eigen::Vector<double, N> x0 = problem.initial_point();
    solver_options<> opts;
    opts.set_gradient_threshold(1e-6);
    opts.max_iterations = kMaxIters;
    opts.set_step_threshold(1e-15);
    typename Policy::options_type popts;
    popts.dual_bound_scale = scale;
    auto s = policy.init(problem, x0, opts, popts);

    cell_result cr;
    cr.problem = pname;
    cr.policy = polname;
    cr.scale = scale;
    cr.reached_feasible = false;
    cr.iters_to_feasible = kMaxIters;
    cr.max_mult_inf = 0.0;
    cr.max_sat_ratio = 0.0;

    double best_f = std::numeric_limits<double>::infinity();
    for(int k = 0; k < kMaxIters; ++k)
    {
        auto r = policy.step(s);
        const double y_inf =
            s.y_dual.size() > 0 ? s.y_dual.cwiseAbs().maxCoeff() : 0.0;
        cr.max_mult_inf = std::max(cr.max_mult_inf, y_inf);
        double sat = 0.0;
        for(int i = 0; i < s.y_dual.size(); ++i)
        {
            const double ci = s.c_dual.size() > i ? s.c_dual[i] : 0.0;
            if(ci > 0.0 && ci < std::numeric_limits<double>::infinity())
                sat = std::max(sat, s.y_dual[i] / ci);
        }
        cr.max_sat_ratio = std::max(cr.max_sat_ratio, sat);

        const double cv = r.constraint_violation;
        if(cv < kFeasTol)
        {
            if(!cr.reached_feasible)
            {
                cr.reached_feasible = true;
                cr.iters_to_feasible = k + 1;
            }
            best_f = std::min(best_f, r.objective_value);
        }
    }

    cr.f_final = cr.reached_feasible ? best_f : s.f;
    const double fstar = problem.optimal_value();
    cr.f_err = std::abs(cr.f_final - fstar) / std::max(std::abs(fstar), 1.0);
    return cr;
}

template <typename Problem>
void run_problem(const std::string& pname, const Problem& problem,
                 const std::vector<double>& grid,
                 std::vector<cell_result>& out)
{
    constexpr int N = Problem::problem_dimension;
    for(double c : grid)
    {
        out.push_back(run_cell(pname, "ccsa",
            argmin::ccsa_quadratic_policy<N>{}, problem, c));
        out.push_back(run_cell(pname, "mma",
            argmin::mma_policy<N>{}, problem, c));
        out.push_back(run_cell(pname, "gcmma_rho_wval",
            argmin::alternative::gcmma::rho_wval_policy<N>{}, problem, c));
    }
}

void print_table(const std::vector<cell_result>& rows)
{
    std::cout << "\n=== dual_bound_scale sweep (inequality-infeasible start) ===\n";
    std::cout << std::left
              << std::setw(12) << "problem"
              << std::setw(16) << "policy"
              << std::setw(10) << "scale"
              << std::setw(8) << "feas?"
              << std::setw(10) << "it_feas"
              << std::setw(14) << "f_final"
              << std::setw(12) << "f_err"
              << std::setw(14) << "max|y|"
              << std::setw(12) << "max_sat" << "\n";
    for(const auto& r : rows)
    {
        std::cout << std::left
                  << std::setw(12) << r.problem
                  << std::setw(16) << r.policy
                  << std::setw(10) << std::defaultfloat << r.scale
                  << std::setw(8) << (r.reached_feasible ? "yes" : "NO")
                  << std::setw(10) << r.iters_to_feasible
                  << std::setw(14) << std::setprecision(5) << r.f_final
                  << std::setw(12) << std::scientific << std::setprecision(2)
                  << r.f_err
                  << std::setw(14) << r.max_mult_inf
                  << std::setw(12) << r.max_sat_ratio
                  << std::defaultfloat << "\n";
    }
}

void write_json(const std::string& path, const std::vector<cell_result>& rows)
{
    std::ofstream out(path);
    out << std::scientific << std::setprecision(8);
    out << "{\n  \"rows\": [\n";
    bool first = true;
    for(const auto& r : rows)
    {
        if(!first) out << ",\n";
        first = false;
        out << "    {\"problem\": \"" << r.problem << "\", \"policy\": \""
            << r.policy << "\", \"scale\": " << r.scale
            << ", \"reached_feasible\": " << (r.reached_feasible ? "true" : "false")
            << ", \"iters_to_feasible\": " << r.iters_to_feasible
            << ", \"f_final\": " << r.f_final
            << ", \"f_err\": " << r.f_err
            << ", \"max_mult_inf\": " << r.max_mult_inf
            << ", \"max_sat_ratio\": " << r.max_sat_ratio << "}";
    }
    out << "\n  ]\n}\n";
}

}

int main(int argc, char** argv)
{
    std::string output = "mma_elastic_dual_sweep.json";
    for(int i = 1; i + 1 < argc; ++i)
        if(std::string(argv[i]) == "--output") output = argv[i + 1];

    const double inf = std::numeric_limits<double>::infinity();
    const std::vector<double> grid{0.1, 1.0, 10.0, 1000.0, 10000.0, inf};

    std::vector<cell_result> rows;
    run_problem("hs024_inf", hs024_infeasible{}, grid, rows);
    run_problem("halfplane", halfplane_cell{}, grid, rows);
    run_problem("halfspace", halfspace_cell{}, grid, rows);

    print_table(rows);
    write_json(output, rows);
    std::cout << "\nJSON written to: " << output << "\n";
    return 0;
}
