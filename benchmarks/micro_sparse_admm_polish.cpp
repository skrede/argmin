// Sparse ADMM QP polish micro-benchmark (argmin-only, informative).
//
// Reproduces the per-step cost of the reduced-KKT polish on the control-shaped
// linear-MPC family, and A/Bs the pattern-gated reuse of that polish factor. The
// polish re-analyzes and re-factorizes a freshly shaped reduced KKT whenever the
// active-set pattern, the delta or the pose changes, so its cost is isolated by
// differencing a polish-on run against a polish-off run over an identical problem
// sequence; the reuse A/B differences a reuse-enabled polish-on run against a
// reuse-disabled one over the same sequence, so the saving the reuse buys on a
// hit is measured directly against the always-re-analyzing path.
//
// Two warm-resolve regimes are measured per cell, matching how the solver is
// exercised in practice:
//   pertIC  -- the linear term q is held fixed and the initial-condition bounds
//              are perturbed by fresh bounded noise each step, the regime that
//              keeps the iteration count pinned at the termination-check floor;
//   rollout -- a closed-loop model-predictive rollout that applies the first
//              input, propagates the plant, slides the tracking reference and
//              re-solves, the regime whose moving reference balloons the ADMM
//              iteration count so the kernel dominates and the polish share
//              falls.
//
// For each regime and cell the bench prints the median userspace instruction
// count with polish on and off, the derived polish share of per-step cost, and
// the consecutive-step fraction of identical polish active-set patterns (the
// reuse hit rate a pattern-gated cache would see), derived exactly as the
// solver derives its own active set from the returned unscaled duals. It then
// prints the reuse A/B: the per-step median with the reuse seam enabled against
// disabled and the instruction saving between them, the direct measure of the
// pattern-gated reuse lever in each regime.
//
// This target is INFORMATIVE: it prints numbers and asserts nothing. A gated
// argmin-only measurement belongs under tests/ for deletability; this ungated
// reproduction belongs here. Counts come from the userspace hardware
// instruction counter; when that counter cannot arm the bench prints the
// unavailable note rather than a free-pass zero.
//
// Reference: Stellato, Banjac, Goulart, Bemporad, Boyd (2020), "OSQP: An
//            operator splitting solver for quadratic programs," Math. Prog.
//            Comp. 12:637-672, Section 5.3 (polish); Borrelli, Bemporad and
//            Morari, Predictive Control for Linear and Hybrid Systems,
//            Cambridge 2017, Chapter 11 (the non-condensed stacked formulation).

#include "perf_instruction_counter.h"
#include "sparse_control_qp_family.h"

#include "argmin/qp/qp_types.h"
#include "argmin/qp/sparse_admm_qp.h"
#include "argmin/options/sparse_qp_options.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <cstdint>
#include <algorithm>

namespace
{

using argmin::bench::instructions_unavailable;
using argmin::bench::perf_instruction_counter;
using argmin::control_qp_data::control_qp;
using argmin::control_qp_data::make_control_qp;

// One warm-resolve control step: the vectors-only data a resolve consumes.
struct step_problem
{
    Eigen::VectorXd q;
    Eigen::VectorXd l;
    Eigen::VectorXd u;
};

// Per-row active-set tag: 0 inactive, 1 lower active, 2 upper active. The
// solver's polish reads exactly this off the unscaled duals, so replicating it
// here gives the pattern a pattern-gated reuse cache would key on.
using active_pattern = std::vector<std::uint8_t>;

// The polish active-set derivation lifted verbatim from try_polish: a dual
// magnitude must clear a relative threshold before its sign pins a bound.
active_pattern pattern_of(const argmin::qp_result<double>& out)
{
    const int m = static_cast<int>(out.y.size());
    const double y_inf = m > 0 ? out.y.cwiseAbs().maxCoeff() : 0.0;
    const double active_tol = 1e-12 * std::max(1.0, y_inf);
    active_pattern pattern(static_cast<std::size_t>(m), 0);
    for(int i = 0; i < m; ++i)
    {
        if(out.y[i] < -active_tol)
            pattern[static_cast<std::size_t>(i)] = 1;
        else if(out.y[i] > active_tol)
            pattern[static_cast<std::size_t>(i)] = 2;
    }
    return pattern;
}

// Write the initial-condition bounds (both sides, they are an equality) so a
// resolve poses x_0 at the supplied state.
void set_initial_condition(const control_qp& cell, const Eigen::VectorXd& x0,
                           Eigen::VectorXd& l, Eigen::VectorXd& u)
{
    for(int j = 0; j < cell.n_state; ++j)
    {
        const int row = cell.initial_row(j);
        l[row] = x0[j];
        u[row] = x0[j];
    }
}

// pertIC: q fixed, initial-condition bounds jittered by fresh bounded noise
// each step. Deterministic given the cell, so the polish-on and polish-off
// measured passes see a bit-identical sequence.
std::vector<step_problem> build_pertic(const control_qp& cell, int steps)
{
    std::mt19937 rng(0x9E3779B9u);
    std::uniform_real_distribution<double> noise(-0.05, 0.05);
    std::vector<step_problem> sequence;
    sequence.reserve(static_cast<std::size_t>(steps));
    for(int k = 0; k < steps; ++k)
    {
        Eigen::VectorXd perturbed = cell.x0;
        for(int j = 0; j < cell.n_state; ++j)
            perturbed[j] += noise(rng);
        Eigen::VectorXd l = cell.l;
        Eigen::VectorXd u = cell.u;
        set_initial_condition(cell, perturbed, l, u);
        sequence.push_back({cell.q, l, u});
    }
    return sequence;
}

// Closed-loop rollout: drive the plant once with polish-on solves to obtain the
// realistic (q, l, u) schedule, then hand that recorded schedule to both
// measured passes so the polish-on and polish-off differences are taken over an
// identical workload. The recorded state trajectory depends only weakly on the
// polish (an accepted polish barely moves the applied input), and the schedule
// is what both passes replay regardless.
std::vector<step_problem> build_rollout(const control_qp& cell, int steps)
{
    argmin::sparse_admm_qp_solver<double> solver;
    argmin::sparse_qp_options opts;
    argmin::qp_result<double> out;

    Eigen::VectorXd state = cell.x0;
    Eigen::VectorXd q = cell.gradient_for(cell.reference_at(0));
    Eigen::VectorXd l = cell.l;
    Eigen::VectorXd u = cell.u;
    set_initial_condition(cell, state, l, u);
    solver.solve_into(cell.P, q, cell.A, l, u, out, opts);

    std::vector<step_problem> sequence;
    sequence.reserve(static_cast<std::size_t>(steps));
    sequence.push_back({q, l, u});

    for(int t = 1; t < steps; ++t)
    {
        Eigen::VectorXd u0(cell.n_input);
        for(int axis = 0; axis < cell.n_input; ++axis)
            u0[axis] = out.x[cell.input_index(0, axis)];
        state = cell.Ad * state + cell.Bd * u0;

        q = cell.gradient_for(cell.reference_at(t));
        l = cell.l;
        u = cell.u;
        set_initial_condition(cell, state, l, u);
        solver.resolve_into(q, l, u, out, opts);
        sequence.push_back({q, l, u});
    }
    return sequence;
}

struct measured_pass
{
    std::vector<std::int64_t> instructions;
    std::vector<int> iterations;
    double hit_rate{0.0};
    bool armed{false};
};

// Pose on the first recorded problem, then measure each subsequent warm
// resolve. The active-set hit rate is derived only on the polish-carrying pass,
// where the returned duals reflect the polished iterate.
measured_pass run_pass(const control_qp& cell, const std::vector<step_problem>& sequence,
                       bool polish, bool reuse, bool collect_hits)
{
    argmin::sparse_admm_qp_solver<double> solver;
    solver.set_polish_reuse(reuse);
    argmin::sparse_qp_options opts;
    opts.polish = polish;
    argmin::qp_result<double> out;

    solver.solve_into(cell.P, sequence[0].q, cell.A, sequence[0].l, sequence[0].u, out, opts);

    perf_instruction_counter counter;
    measured_pass pass;
    pass.armed = counter.armed();

    active_pattern previous;
    bool have_previous = false;
    std::size_t identical = 0;
    std::size_t compared = 0;

    for(std::size_t k = 1; k < sequence.size(); ++k)
    {
        const std::int64_t before = counter.read();
        solver.resolve_into(sequence[k].q, sequence[k].l, sequence[k].u, out, opts);
        const std::int64_t after = counter.read();

        std::int64_t delta = instructions_unavailable;
        if(before >= 0 && after >= 0)
            delta = after - before;
        pass.instructions.push_back(delta);
        pass.iterations.push_back(out.iterations);

        if(collect_hits)
        {
            active_pattern current = pattern_of(out);
            if(have_previous)
            {
                ++compared;
                if(current == previous)
                    ++identical;
            }
            previous = std::move(current);
            have_previous = true;
        }
    }

    if(collect_hits && compared > 0)
        pass.hit_rate = static_cast<double>(identical) / static_cast<double>(compared);
    return pass;
}

double median(std::vector<std::int64_t> values)
{
    if(values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2;
    if(values.size() % 2 == 1)
        return static_cast<double>(values[mid]);
    return 0.5 * (static_cast<double>(values[mid - 1]) + static_cast<double>(values[mid]));
}

double median_int(const std::vector<int>& values)
{
    std::vector<std::int64_t> widened(values.begin(), values.end());
    return median(std::move(widened));
}

void report_regime(const char* label, const control_qp& cell,
                   const std::vector<step_problem>& sequence)
{
    // The polish share isolates the intrinsic re-analyzing polish cost, so it is
    // measured with the reuse seam OFF; the reuse lever is reported separately by
    // report_reuse. This keeps the share independent of the reuse hit rate.
    const measured_pass on = run_pass(cell, sequence, true, false, true);
    const measured_pass off = run_pass(cell, sequence, false, false, false);

    if(!on.armed || !off.armed)
    {
        std::printf("    %-8s | instruction counter unavailable (raise perf_event_paranoid<=2)\n",
                    label);
        std::printf("    %-8s | iters med on %.0f off %.0f | hit-rate %5.1f%%\n",
                    label, median_int(on.iterations), median_int(off.iterations),
                    100.0 * on.hit_rate);
        return;
    }

    const double med_on = median(on.instructions);
    const double med_off = median(off.instructions);
    const double share = med_on > 0.0 ? 100.0 * (med_on - med_off) / med_on : 0.0;

    std::printf("    %-8s | on %9.0f | off %9.0f | polish %5.1f%% | "
                "iters %.0f | hit-rate %5.1f%%\n",
                label, med_on, med_off, share, median_int(on.iterations),
                100.0 * on.hit_rate);
}

// Reuse A/B: polish stays on for both passes; the seam alone differs. reuse-on
// takes the pattern-gated fast path on a hit, reuse-off re-analyzes every call
// (the pre-reuse behavior). The saving is (reuse-off - reuse-on) instructions per
// step: positive means the reuse paid, and a near-zero value on a ~0%-hit regime
// is the evidence the miss-path compare is effectively free rather than a net
// per-step regression.
void report_reuse(const char* label, const control_qp& cell,
                  const std::vector<step_problem>& sequence)
{
    const measured_pass on = run_pass(cell, sequence, true, true, false);
    const measured_pass off = run_pass(cell, sequence, true, false, false);

    if(!on.armed || !off.armed)
    {
        std::printf("    %-8s | reuse A/B: instruction counter unavailable\n", label);
        return;
    }

    const double med_on = median(on.instructions);
    const double med_off = median(off.instructions);
    const double saved = med_off - med_on;
    const double pct = med_off > 0.0 ? 100.0 * saved / med_off : 0.0;

    std::printf("    %-8s | reuse-on %9.0f | reuse-off %9.0f | saved %+9.0f (%+5.1f%%)\n",
                label, med_on, med_off, saved, pct);
}

void run_cell(int n_input, int horizon, int steps)
{
    const control_qp cell =
        make_control_qp(horizon, 2 * n_input, n_input, false, 1000003u);

    std::printf("  cell ni=%d N=%d | vars %d | rows %d\n", n_input, horizon,
                cell.n_variables(), cell.n_rows());

    const std::vector<step_problem> pertic = build_pertic(cell, steps);
    const std::vector<step_problem> rollout = build_rollout(cell, steps);
    report_regime("pertIC", cell, pertic);
    report_regime("rollout", cell, rollout);
    report_reuse("pertIC", cell, pertic);
    report_reuse("rollout", cell, rollout);
}

}

int main()
{
    constexpr int steps = 101;

    std::printf("Sparse ADMM QP polish micro-benchmark\n");
    std::printf("=====================================\n");
    std::printf("median userspace instructions per warm resolve; polish share is\n");
    std::printf("(on - off) / on; hit-rate is the consecutive-step identical\n");
    std::printf("active-set fraction. %d steps per regime, library-default options.\n\n",
                steps);

    run_cell(2, 10, steps);
    run_cell(2, 20, steps);
    run_cell(3, 20, steps);
    run_cell(3, 30, steps);

    return 0;
}
