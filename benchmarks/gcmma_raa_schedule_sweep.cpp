// Sweep harness for the GCMMA raa-augmented penalty schedule knob
// (raa_augmented_policy::options_type::raa_svanberg_schedule).
//
// Two schedules are compared:
//   - false: the pre-fix schedule -- fixed raa floor (raa_min = 1e-7) and
//     non-conservative growth raa' = raa + 2*delta.
//   - true:  Svanberg 2002 eq. 3.6-3.9 -- the raa floor recomputed each
//     outer iteration from the current gradient as
//     (0.1/n) * sum_j |dg_j| * range_j, and growth raa' = 1.1*(raa + delta).
//
// The documented failure the Svanberg schedule targets: a fixed 1e-7 raa
// floor wastes inner conservativity budget on large-scale objectives (the
// penalty is negligible relative to a 1e6-scaled objective, so the inner
// loop must grow it from scratch each outer iteration). The gradient-scaled
// floor tracks the objective magnitude, so the schedule should be robust
// across objective scale.
//
// Problem set: the MMA-compatible (inequality + box) Hock-Schittkowski cells
// HS024, HS035, HS076, each at objective scales {1, 1e3, 1e6}. Scaling the
// objective by s multiplies f and grad f by s (the optimizer and optimum are
// unchanged; only the penalty-vs-objective magnitude balance changes).
//
// Metric: outer iterations to reach a relative objective error
// |f - f*| / max(|f*|, 1) < 1e-2, and the smallest relative error reached
// over the run (convergence quality). A scale-robust schedule reaches the
// optimum in a scale-independent iteration count; the fixed-floor schedule
// should degrade (more iterations, or failure) as the objective scale grows.
//
// Output: a JSON document (default gcmma_raa_schedule_sweep.json in CWD,
// override with --output PATH) plus a per-row stdout table. Non-ctest,
// one-shot, read-only on argmin.
//
// Reference: Svanberg (2002) "A class of globally convergent optimization
//            methods based on conservative convex separable
//            approximations", SIAM J. Optim. 12(2):555-573, eqs. 3.6-3.9.
//            Hock and Schittkowski (1981).

#include "argmin/solver/alternative/gcmma/raa_augmented_policy.h"
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

// Objective-scaled wrapper: forwards the base problem but multiplies the
// objective value and gradient by a constant factor (constraints, bounds,
// x0 unchanged). The optimizer and optimal point are invariant; only the
// objective-vs-penalty magnitude balance changes.
template <typename Base>
struct scaled_objective
{
    static constexpr int problem_dimension = Base::problem_dimension;
    Base base{};
    double factor{1.0};

    int dimension() const { return base.dimension(); }
    int num_equality() const { return base.num_equality(); }
    int num_inequality() const { return base.num_inequality(); }

    double value(const Eigen::Vector<double, problem_dimension>& x) const
    {
        return factor * base.value(x);
    }
    void gradient(const Eigen::Vector<double, problem_dimension>& x,
                  Eigen::Vector<double, problem_dimension>& g) const
    {
        base.gradient(x, g);
        g *= factor;
    }
    void constraints(const Eigen::Vector<double, problem_dimension>& x,
                     auto& c) const { base.constraints(x, c); }
    void constraint_jacobian(const Eigen::Vector<double, problem_dimension>& x,
                             auto& J) const { base.constraint_jacobian(x, J); }
    auto lower_bounds() const { return base.lower_bounds(); }
    auto upper_bounds() const { return base.upper_bounds(); }
    auto initial_point() const { return base.initial_point(); }
    double optimal_value() const { return factor * base.optimal_value(); }
};

struct cell_result
{
    std::string problem;
    double scale{};
    bool svanberg{};
    bool reached{};
    int iters_to_target{};
    double rel_err_floor{};
};

constexpr int kMaxIters = 500;
constexpr double kTargetRel = 1e-2;

template <typename Problem>
cell_result run_cell(const std::string& pname, const Problem& problem,
                     double scale, bool svanberg)
{
    constexpr int N = Problem::problem_dimension;
    Eigen::Vector<double, N> x0 = problem.initial_point();
    solver_options<> opts;
    opts.max_iterations = kMaxIters;
    opts.set_gradient_threshold(1e-6);
    opts.set_step_threshold(1e-14);

    argmin::alternative::gcmma::raa_augmented_policy<N> policy;
    typename argmin::alternative::gcmma::raa_augmented_policy<N>::options_type
        popts;
    popts.raa_svanberg_schedule = svanberg;
    auto s = policy.init(problem, x0, opts, popts);

    const double fstar = problem.optimal_value();
    const double denom = std::max(std::abs(fstar), 1.0);

    cell_result cr;
    cr.problem = pname;
    cr.scale = scale;
    cr.svanberg = svanberg;
    cr.reached = false;
    cr.iters_to_target = kMaxIters;
    cr.rel_err_floor = std::numeric_limits<double>::infinity();

    for(int k = 0; k < kMaxIters; ++k)
    {
        auto r = policy.step(s);
        const double cv = r.constraint_violation;
        if(cv < 1e-4)
        {
            const double rel = std::abs(r.objective_value - fstar) / denom;
            cr.rel_err_floor = std::min(cr.rel_err_floor, rel);
            if(!cr.reached && rel < kTargetRel)
            {
                cr.reached = true;
                cr.iters_to_target = k + 1;
            }
        }
    }
    return cr;
}

template <typename Base>
void run_problem(const std::string& pname, std::vector<cell_result>& out)
{
    const std::vector<double> scales{1.0, 1e3, 1e6};
    for(double sc : scales)
        for(bool sv : {false, true})
        {
            scaled_objective<Base> p;
            p.factor = sc;
            out.push_back(run_cell(pname, p, sc, sv));
        }
}

void print_table(const std::vector<cell_result>& rows)
{
    std::cout << "\n=== raa schedule sweep (Svanberg 3.6-3.9 vs fixed) ===\n";
    std::cout << std::left
              << std::setw(10) << "problem"
              << std::setw(10) << "scale"
              << std::setw(12) << "schedule"
              << std::setw(10) << "reached"
              << std::setw(12) << "it_target"
              << std::setw(14) << "rel_floor" << "\n";
    for(const auto& r : rows)
    {
        std::cout << std::left
                  << std::setw(10) << r.problem
                  << std::setw(10) << std::scientific << std::setprecision(0)
                  << r.scale
                  << std::setw(12) << (r.svanberg ? "svanberg" : "fixed")
                  << std::setw(10) << (r.reached ? "yes" : "NO")
                  << std::setw(12) << std::defaultfloat << r.iters_to_target
                  << std::setw(14) << std::scientific << std::setprecision(2)
                  << r.rel_err_floor << "\n";
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
        out << "    {\"problem\": \"" << r.problem << "\", \"scale\": "
            << r.scale << ", \"svanberg\": " << (r.svanberg ? "true" : "false")
            << ", \"reached\": " << (r.reached ? "true" : "false")
            << ", \"iters_to_target\": " << r.iters_to_target
            << ", \"rel_err_floor\": " << r.rel_err_floor << "}";
    }
    out << "\n  ]\n}\n";
}

}

int main(int argc, char** argv)
{
    std::string output = "gcmma_raa_schedule_sweep.json";
    for(int i = 1; i + 1 < argc; ++i)
        if(std::string(argv[i]) == "--output") output = argv[i + 1];

    std::vector<cell_result> rows;
    run_problem<argmin::hs024<>>("HS024", rows);
    run_problem<argmin::hs035<>>("HS035", rows);
    run_problem<argmin::hs076<>>("HS076", rows);

    print_table(rows);
    write_json(output, rows);
    std::cout << "\nJSON written to: " << output << "\n";
    return 0;
}
