// Sweep harness for the two MMA subproblem-regularizer knobs introduced
// with the Svanberg distance-scaled p/q coefficients:
//
//   - raai (mma_policy::options_type::raai): the distance-scaled
//     regularizer constant in
//       p_ij = dxU^2 (1.001 g+ + 0.001 g- + raai/range_j),
//       q_ij = dxL^2 (0.001 g+ + 1.001 g- + raai/range_j).
//     Grid: {0, 1e-6, 1e-5 (Svanberg starting hypothesis), 1e-4}.
//
//   - asymptote_min_fraction (ASYMIN): the lower clamp on the asymptote
//     distance as a fraction of range_j.
//     Grid: {1e-4 (incumbent), 1e-3, 1e-2 (Svanberg notes 2.16)}.
//
// Problem set: the MMA-compatible (inequality + box) Hock-Schittkowski
// cells HS024, HS035, HS043, HS076, plus a small-gradient-terminal
// quartic where |grad f| -> 0 at the optimum. The quartic is the cell
// that most stresses raai: as the objective gradient vanishes the p/q
// coefficients are dominated by the raai/range term, and a flat additive
// epsilon (the pre-fix form) would collapse the reciprocal minimizer onto
// the asymptote-window midpoint and wander.
//
// Metric (raai): terminal stationarity floor = the smallest projected KKT
// residual reached over the run, plus a terminal-wandering measure =
// max |f_{k} - f_{k-1}| over the final tail_window iterations. A good raai
// drives the KKT floor down without inducing low-amplitude terminal
// wandering; raai = 0 is the flat-regularizer failure mode to beat.
//
// Metric (ASYMIN): oscillation/destabilization onset = the number of
// iterations on which the objective increased (non-monotone events) and
// the largest such increase, over the run. A tighter ASYMIN should keep
// the asymptote distance in the reciprocal-model-meaningful regime and
// suppress asymptote-collapse oscillation.
//
// Each cell is run by manual stepping (init + step loop) so the full
// objective / KKT trajectory is observed rather than only the terminal
// solve_result. Both knobs are runtime options -- no rebuild between
// values.
//
// Output: a JSON document (default mma_regularizer_sweep.json in CWD,
// override with --output PATH) plus a per-row stdout table. Non-ctest,
// one-shot, read-only on argmin.
//
// The shipped provisional defaults (raai, asymptote_min_fraction) are set
// from this data and re-locked only after the upstream separable-
// approximation math has settled.
//
// Reference: Svanberg (2007) "MMA and GCMMA -- two methods for nonlinear
//            optimization", KTH lecture notes, eqs. 2.3-2.4, 2.16.
//            Hock and Schittkowski (1981) "Test Examples for Nonlinear
//            Programming Codes", LNEMS vol. 187, Springer.

#include "argmin/solver/mma_policy.h"
#include "argmin/test_functions/hock_schittkowski.h"
#include "argmin/types.h"

#include <Eigen/Core>

#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>

namespace
{

using argmin::mma_policy;
using argmin::solver_options;

// Small-gradient-terminal quartic: min (x0-1)^4 + (x1-1)^4 subject to the
// (inactive at the optimum) inequality 3 - x0 - x1 >= 0, box [0, 3]^2.
// The optimum is x = (1, 1), f* = 0, where grad f = 4(x-1)^3 = 0.
struct quartic_terminal
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int dimension() const { return 2; }
    int num_equality() const { return 0; }
    int num_inequality() const { return 1; }

    double value(const Eigen::VectorXd& x) const
    {
        const double a = x[0] - 1.0, b = x[1] - 1.0;
        return a * a * a * a + b * b * b * b;
    }
    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        g.resize(2);
        const double a = x[0] - 1.0, b = x[1] - 1.0;
        g[0] = 4.0 * a * a * a;
        g[1] = 4.0 * b * b * b;
    }
    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = 3.0 - x[0] - x[1];
    }
    void constraint_jacobian(const Eigen::VectorXd&, Eigen::MatrixXd& J) const
    {
        J.resize(1, 2);
        J(0, 0) = -1.0;
        J(0, 1) = -1.0;
    }
    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd{{0.0, 0.0}}; }
    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd{{3.0, 3.0}}; }
    Eigen::VectorXd initial_point() const { return Eigen::VectorXd{{0.2, 0.2}}; }
    double optimal_value() const { return 0.0; }
};

struct cell_result
{
    std::string problem;
    double knob_value{};
    double f_final{};
    double f_err{};
    double kkt_floor{};      // smallest KKT residual over the run
    double kkt_final{};      // KKT residual at the last iteration
    double tail_wander{};    // max |df| over the tail window
    int nonmonotone{};       // count of iterations with f increase
    double max_increase{};   // largest single-iteration f increase
    int iterations{};
};

constexpr int kMaxIters = 400;
constexpr int kTailWindow = 20;

template <typename Problem, int N>
cell_result run_cell(const std::string& name, const Problem& problem,
                     const Eigen::VectorXd& x0, double fstar,
                     typename mma_policy<N>::options_type popts,
                     double knob_value)
{
    mma_policy<N> policy;
    solver_options<> opts;
    auto s = policy.init(problem, Eigen::Vector<double, N>(x0), opts, popts);

    std::vector<double> fs;
    std::vector<double> kkts;
    double f_prev = problem.value(Eigen::Vector<double, N>(x0));
    int nonmono = 0;
    double max_inc = 0.0;

    for(int k = 0; k < kMaxIters; ++k)
    {
        auto r = policy.step(s);
        const double f = r.objective_value;
        const double kkt = r.kkt_residual.value_or(r.gradient_norm);
        fs.push_back(f);
        kkts.push_back(kkt);
        if(f > f_prev)
        {
            ++nonmono;
            max_inc = std::max(max_inc, f - f_prev);
        }
        f_prev = f;
    }

    cell_result cr;
    cr.problem = name;
    cr.knob_value = knob_value;
    cr.f_final = fs.back();
    cr.f_err = std::abs(fs.back() - fstar) / std::max(std::abs(fstar), 1.0);
    cr.kkt_floor = *std::min_element(kkts.begin(), kkts.end());
    cr.kkt_final = kkts.back();
    cr.nonmonotone = nonmono;
    cr.max_increase = max_inc;
    cr.iterations = kMaxIters;

    double wander = 0.0;
    const int start = std::max(1, static_cast<int>(fs.size()) - kTailWindow);
    for(int k = start; k < static_cast<int>(fs.size()); ++k)
        wander = std::max(wander, std::abs(fs[k] - fs[k - 1]));
    cr.tail_wander = wander;
    return cr;
}

// Run every problem cell for one options configuration.
std::vector<cell_result> run_all(
    double raai, double asymin, double knob_value)
{
    std::vector<cell_result> out;

    auto set = [&](auto popts) {
        popts.raai = raai;
        popts.asymptote_min_fraction = asymin;
        return popts;
    };

    {
        argmin::hs024<> p; using P = argmin::hs024<>;
        constexpr int N = P::problem_dimension;
        out.push_back(run_cell<P, N>("HS024", p, p.initial_point(),
            p.optimal_value(), set(mma_policy<N>::options_type{}), knob_value));
    }
    {
        argmin::hs035<> p; using P = argmin::hs035<>;
        constexpr int N = P::problem_dimension;
        out.push_back(run_cell<P, N>("HS035", p, p.initial_point(),
            p.optimal_value(), set(mma_policy<N>::options_type{}), knob_value));
    }
    {
        argmin::hs043<> p; using P = argmin::hs043<>;
        constexpr int N = P::problem_dimension;
        out.push_back(run_cell<P, N>("HS043", p, p.initial_point(),
            p.optimal_value(), set(mma_policy<N>::options_type{}), knob_value));
    }
    {
        argmin::hs076<> p; using P = argmin::hs076<>;
        constexpr int N = P::problem_dimension;
        out.push_back(run_cell<P, N>("HS076", p, p.initial_point(),
            p.optimal_value(), set(mma_policy<N>::options_type{}), knob_value));
    }
    {
        quartic_terminal p;
        constexpr int N = argmin::dynamic_dimension;
        out.push_back(run_cell<quartic_terminal, N>("quartic", p,
            p.initial_point(), p.optimal_value(),
            set(mma_policy<N>::options_type{}), knob_value));
    }
    return out;
}

void print_table(const std::string& knob,
                 const std::vector<std::vector<cell_result>>& passes)
{
    std::cout << "\n=== " << knob << " sweep ===\n";
    std::cout << std::left << std::setw(10) << "problem"
              << std::setw(12) << knob
              << std::setw(14) << "f_err"
              << std::setw(14) << "kkt_floor"
              << std::setw(14) << "kkt_final"
              << std::setw(14) << "tail_wander"
              << std::setw(10) << "nonmono"
              << std::setw(14) << "max_incr" << "\n";
    for(const auto& pass : passes)
        for(const auto& r : pass)
        {
            std::cout << std::left << std::setw(10) << r.problem
                      << std::setw(12) << std::scientific << std::setprecision(1)
                      << r.knob_value
                      << std::setw(14) << r.f_err
                      << std::setw(14) << r.kkt_floor
                      << std::setw(14) << r.kkt_final
                      << std::setw(14) << r.tail_wander
                      << std::setw(10) << std::defaultfloat << r.nonmonotone
                      << std::setw(14) << std::scientific << std::setprecision(1)
                      << r.max_increase << "\n";
        }
}

void write_json(const std::string& path,
                const std::vector<std::vector<cell_result>>& raai_passes,
                const std::vector<std::vector<cell_result>>& asymin_passes)
{
    std::ofstream out(path);
    out << std::scientific << std::setprecision(8);
    auto dump = [&](const char* tag,
                    const std::vector<std::vector<cell_result>>& passes) {
        out << "  \"" << tag << "\": [\n";
        bool first = true;
        for(const auto& pass : passes)
            for(const auto& r : pass)
            {
                if(!first) out << ",\n";
                first = false;
                out << "    {\"problem\": \"" << r.problem << "\", \"value\": "
                    << r.knob_value << ", \"f_err\": " << r.f_err
                    << ", \"kkt_floor\": " << r.kkt_floor
                    << ", \"kkt_final\": " << r.kkt_final
                    << ", \"tail_wander\": " << r.tail_wander
                    << ", \"nonmonotone\": " << r.nonmonotone
                    << ", \"max_increase\": " << r.max_increase << "}";
            }
        out << "\n  ]";
    };
    out << "{\n";
    dump("raai", raai_passes);
    out << ",\n";
    dump("asymin", asymin_passes);
    out << "\n}\n";
}

}

int main(int argc, char** argv)
{
    std::string output = "mma_regularizer_sweep.json";
    for(int i = 1; i + 1 < argc; ++i)
        if(std::string(argv[i]) == "--output") output = argv[i + 1];

    // Pass A: sweep raai at the incumbent ASYMIN = 1e-4.
    const std::vector<double> raai_grid{0.0, 1e-6, 1e-5, 1e-4};
    std::vector<std::vector<cell_result>> raai_passes;
    for(double v : raai_grid)
        raai_passes.push_back(run_all(v, 1e-4, v));

    // Pass B: sweep ASYMIN at the raai starting hypothesis 1e-5.
    const std::vector<double> asymin_grid{1e-4, 1e-3, 1e-2};
    std::vector<std::vector<cell_result>> asymin_passes;
    for(double v : asymin_grid)
        asymin_passes.push_back(run_all(1e-5, v, v));

    print_table("raai", raai_passes);
    print_table("asymptote_min_fraction", asymin_passes);
    write_json(output, raai_passes, asymin_passes);
    std::cout << "\nJSON written to: " << output << "\n";
    return 0;
}
