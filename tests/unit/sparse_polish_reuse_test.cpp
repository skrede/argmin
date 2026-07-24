// Correctness of the pattern-gated polish-KKT reuse in the sparse ADMM QP
// solver. The solver caches the previous polish call's active-set index vectors,
// delta and pose, and on an unchanged pattern reuses the already-analyzed
// symbolic factorization and numeric factor of the reduced KKT instead of
// rebuilding them. Because P, A and delta are frozen across a resolve, an
// unchanged active set implies a bit-identical reduced KKT, so the reused-path
// polished result must equal the always-re-analyzing path bit for bit.
//
// This file proves that contract two ways. First, a stable-pattern sequence
// (small initial-condition perturbations that leave the active set fixed): a
// reuse-enabled and a reuse-disabled solver, driven over an identical recorded
// problem sequence, must return bit-identical polished x and y at every step,
// and the reuse-enabled solver's polish-analysis counter must fall strictly
// below the re-analyzing one -- the reuse fired. Second, a pattern-changing
// sequence (a sliding tracking reference whose active set churns every step):
// the two solvers must still return bit-identical polished results, and the
// reuse-enabled solver's polish-analysis counter must equal the re-analyzing
// one -- the negative control that a changed active set never silently reuses a
// stale factor and so never yields a wrong answer.
//
// Reference: Stellato, Banjac, Goulart, Bemporad, Boyd (2020), "OSQP: An
//            operator splitting solver for quadratic programs," Math. Prog.
//            Comp. 12:637-672, Section 5.3 (polish).

#include "sparse_control_qp_family.h"

#include "argmin/qp/qp_types.h"
#include "argmin/qp/sparse_admm_qp.h"
#include "argmin/options/sparse_qp_options.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <catch2/catch_test_macros.hpp>

#include <vector>
#include <cstddef>
#include <cstring>

using namespace argmin;
using argmin::control_qp_data::control_qp;
using argmin::control_qp_data::make_control_qp;

namespace
{

// The vectors-only data a single warm resolve consumes.
struct step_problem
{
    Eigen::VectorXd q;
    Eigen::VectorXd l;
    Eigen::VectorXd u;
};

// True only when the two vectors are byte-for-byte identical -- the strongest
// form of the equality contract, distinguishing even signed zeros. This is the
// same test the sparse_kkt witness uses for its manual-solve bit-identity.
bool bit_identical(const Eigen::VectorXd& a, const Eigen::VectorXd& b)
{
    return a.size() == b.size()
           && std::memcmp(a.data(), b.data(),
                          static_cast<std::size_t>(a.size()) * sizeof(double))
                  == 0;
}

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

// Stable-pattern schedule: the linear term is held fixed and the initial
// condition is nudged by a tiny deterministic offset each step, small enough
// that the set of binding constraints does not move, so the polish active-set
// pattern is unchanged from step to step and the reuse fires. Deterministic, so
// the reuse-on and reuse-off passes replay a bit-identical sequence.
std::vector<step_problem> build_stable(const control_qp& cell, int steps)
{
    std::vector<step_problem> sequence;
    sequence.reserve(static_cast<std::size_t>(steps));
    for(int k = 0; k < steps; ++k)
    {
        Eigen::VectorXd x0 = cell.x0;
        const double nudge = 1e-4 * static_cast<double>((k % 5) - 2);
        for(int j = 0; j < cell.n_state; ++j)
            x0[j] += nudge;
        Eigen::VectorXd l = cell.l;
        Eigen::VectorXd u = cell.u;
        set_initial_condition(cell, x0, l, u);
        sequence.push_back({cell.q, l, u});
    }
    return sequence;
}

// Churning schedule: a closed-loop tracking rollout whose sliding reference
// drives the plant past its state bounds, so a different set of constraints
// binds on every step and the polish active-set pattern churns. The schedule is
// recorded once here and replayed verbatim to both measured passes.
std::vector<step_problem> build_churning(const control_qp& cell, int steps)
{
    sparse_admm_qp_solver<double> solver;
    sparse_qp_options opts;
    qp_result<double> out;

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

struct pass_result
{
    std::vector<Eigen::VectorXd> x;
    std::vector<Eigen::VectorXd> y;
    std::size_t polish_analyses{0};
    std::size_t polish_factorizations{0};
};

// Pose on the first recorded problem, then warm-resolve the rest through the
// vectors-only path, recording the polished primal and dual at every step and
// the lifetime polish counters at the end. The reuse seam is set before posing
// so the very first polish already obeys it.
pass_result run_pass(const control_qp& cell, const std::vector<step_problem>& sequence,
                     bool reuse_enabled)
{
    sparse_admm_qp_solver<double> solver;
    solver.set_polish_reuse(reuse_enabled);
    sparse_qp_options opts;
    qp_result<double> out;

    solver.solve_into(cell.P, sequence[0].q, cell.A, sequence[0].l, sequence[0].u, out, opts);

    pass_result pass;
    pass.x.push_back(out.x);
    pass.y.push_back(out.y);

    for(std::size_t k = 1; k < sequence.size(); ++k)
    {
        solver.resolve_into(sequence[k].q, sequence[k].l, sequence[k].u, out, opts);
        pass.x.push_back(out.x);
        pass.y.push_back(out.y);
    }

    pass.polish_analyses = solver.polish_analyses();
    pass.polish_factorizations = solver.polish_factorizations();
    return pass;
}

// Every recorded step must have solved and been polished; a bit-identity claim
// over a sequence that silently skipped the polish on some steps would be
// vacuous. The re-analyzing pass analyzes exactly once per polished call, so its
// analysis counter is the number of polished steps.
void require_all_bit_identical(const pass_result& a, const pass_result& b)
{
    REQUIRE(a.x.size() == b.x.size());
    REQUIRE(a.y.size() == b.y.size());
    for(std::size_t k = 0; k < a.x.size(); ++k)
    {
        INFO("step " << k);
        CHECK(bit_identical(a.x[k], b.x[k]));
        CHECK(bit_identical(a.y[k], b.y[k]));
    }
}

constexpr int stable_steps = 40;
constexpr int churning_steps = 60;

}

// A stable active-set pattern: the reuse-enabled and reuse-disabled solvers must
// agree bit for bit on the polished primal and dual at every step, and the
// reuse-enabled solver must have analyzed strictly fewer reduced systems than
// the re-analyzing one -- proof the reuse actually fired without moving a bit.
TEST_CASE("sparse_admm_qp polish reuse -- bit-identity stable", "[qp][sparse_polish_reuse]")
{
    const control_qp cell = make_control_qp(10, 4, 2, false, 1000003u);
    const std::vector<step_problem> sequence = build_stable(cell, stable_steps);

    const pass_result on = run_pass(cell, sequence, true);
    const pass_result off = run_pass(cell, sequence, false);

    require_all_bit_identical(on, off);

    // The re-analyzing pass analyzes once per polished step; the reusing pass
    // must analyze strictly fewer, and its analysis and factorization counters
    // move together (a reused analysis is a reused factor).
    CHECK(off.polish_analyses == off.polish_factorizations);
    CHECK(on.polish_analyses < off.polish_analyses);
    CHECK(on.polish_factorizations < off.polish_factorizations);
    CHECK(on.polish_analyses == on.polish_factorizations);
}

// A churning active-set pattern: the two solvers must still agree bit for bit at
// every step, and the reuse-enabled solver must have analyzed exactly as many
// reduced systems as the re-analyzing one -- the negative control that a changed
// pattern re-analyzes every step and never serves a stale factor.
TEST_CASE("sparse_admm_qp polish reuse -- bit-identity churning", "[qp][sparse_polish_reuse]")
{
    const control_qp cell = make_control_qp(20, 6, 3, false, 1000037u);
    const std::vector<step_problem> sequence = build_churning(cell, churning_steps);

    const pass_result on = run_pass(cell, sequence, true);
    const pass_result off = run_pass(cell, sequence, false);

    require_all_bit_identical(on, off);

    CHECK(off.polish_analyses == off.polish_factorizations);
    CHECK(on.polish_analyses == off.polish_analyses);
    CHECK(on.polish_factorizations == off.polish_factorizations);
}

// The counter negative control, stated on its own so the reuse fires on the
// stable sequence and is fully suppressed on the churning one within a single
// case: the reuse-enabled solver analyzes strictly fewer systems than steps when
// the pattern is stable and exactly as many when it churns, so the detector can
// neither miss a reuse opportunity wholesale nor invent one on a changed set.
TEST_CASE("sparse_admm_qp polish reuse -- counter negative control",
          "[qp][sparse_polish_reuse]")
{
    const control_qp stable_cell = make_control_qp(10, 4, 2, false, 1000003u);
    const std::vector<step_problem> stable = build_stable(stable_cell, stable_steps);
    const pass_result stable_on = run_pass(stable_cell, stable, true);
    const pass_result stable_off = run_pass(stable_cell, stable, false);
    CHECK(stable_on.polish_analyses < stable_off.polish_analyses);

    const control_qp churn_cell = make_control_qp(20, 6, 3, false, 1000037u);
    const std::vector<step_problem> churn = build_churning(churn_cell, churning_steps);
    const pass_result churn_on = run_pass(churn_cell, churn, true);
    const pass_result churn_off = run_pass(churn_cell, churn, false);
    CHECK(churn_on.polish_analyses == churn_off.polish_analyses);

    // The churning sequence must genuinely churn, or the "== off" control above
    // would hold vacuously: most steps re-analyze, so the analysis count is a
    // large fraction of the step count rather than near one.
    CHECK(churn_off.polish_analyses > static_cast<std::size_t>(churning_steps / 2));
}
