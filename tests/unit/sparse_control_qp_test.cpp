#include "argmin/qp/qp_types.h"
#include "argmin/qp/sparse_admm_qp.h"

#include "sparse_control_qp_family.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
#include <algorithm>

using namespace argmin;
using argmin::control_qp_data::control_qp;

namespace
{

constexpr double inf = std::numeric_limits<double>::infinity();
constexpr double optimality_tol = 1e-6;
constexpr double equality_width = 1e-9;

// First-order optimality measured from the returned iterate and the ORIGINAL
// unscaled problem data alone. Nothing here reads the solver's own residual
// fields, its scaled quantities, or a second solver: an oracle that consulted
// the instrument under measurement would certify nothing.
struct optimality
{
    double stationarity{0.0};
    double feasibility{0.0};
    double complementarity{0.0};
    int complementarity_violations{0};

    bool within(double tol) const
    {
        return stationarity <= tol && feasibility <= tol && complementarity_violations == 0;
    }
};

optimality check_optimality(const control_qp& p, const Eigen::VectorXd& q,
                            const Eigen::VectorXd& l, const Eigen::VectorXd& u,
                            const Eigen::VectorXd& x, const Eigen::VectorXd& y, double tol)
{
    const Eigen::VectorXd px = p.P * x;
    const Eigen::VectorXd aty = p.A.transpose() * y;
    const Eigen::VectorXd ax = p.A * x;

    optimality o;

    const double dual_scale = std::max({px.cwiseAbs().maxCoeff(), aty.cwiseAbs().maxCoeff(),
                                        q.cwiseAbs().maxCoeff(), 1.0});
    o.stationarity = (px + q + aty).cwiseAbs().maxCoeff() / dual_scale;

    const double primal_scale = std::max(1.0, ax.cwiseAbs().maxCoeff());
    o.feasibility = (l - ax).cwiseMax(ax - u).cwiseMax(0.0).maxCoeff() / primal_scale;

    // Complementarity as a sign-and-support statement rather than the product
    // y_i * (a_i^T x - b_i): the product hides a large multiplier on a row that
    // merely happens to sit near its bound, which is exactly the failure mode a
    // polish step can introduce.
    const double dual_scale_y = std::max(1.0, y.cwiseAbs().maxCoeff());
    for(int i = 0; i < static_cast<int>(p.A.rows()); ++i)
    {
        const bool equality = std::isfinite(l[i]) && std::isfinite(u[i])
                              && (u[i] - l[i]) <= equality_width;
        if(equality)
            continue;
        const double to_lower = std::isfinite(l[i]) ? ax[i] - l[i] : inf;
        const double to_upper = std::isfinite(u[i]) ? u[i] - ax[i] : inf;
        double worst = 0.0;
        if(y[i] > tol * dual_scale_y)
            worst = to_upper / primal_scale;
        else if(y[i] < -tol * dual_scale_y)
            worst = to_lower / primal_scale;
        else if(to_lower > tol * primal_scale && to_upper > tol * primal_scale)
            worst = std::abs(y[i]) / dual_scale_y;
        o.complementarity = std::max(o.complementarity, worst);
        if(worst > tol)
            ++o.complementarity_violations;
    }
    return o;
}

double density(const control_qp::sparse_type& m)
{
    return static_cast<double>(m.nonZeros())
           / (static_cast<double>(m.rows()) * static_cast<double>(m.cols()));
}

}

// ---------------------------------------------------------------------------
// The family is a workload claim, and a claim gets verified before it is
// trusted. A generator that quietly stopped emitting banded dynamics, or that
// shrank below the scale at which sparsity is load-bearing, would leave every
// downstream case passing while measuring nothing -- so the shape arithmetic,
// the equality predicate, the row-by-row dynamics recursion re-derived from the
// exposed plant matrices, and the scale and density guards are all assertions
// here rather than prose in a plan.
//
// Measured on the largest member: 906 variables, 1626 rows, 3666 stored entries
// in A, density 2.49e-3 against the committed 2e-2 ceiling -- roughly an order
// of headroom on the density guard and 1.8x on the variable count.
// ---------------------------------------------------------------------------
TEST_CASE("sparse control-shaped family is structurally what it claims to be",
          "[qp][sparse_control]")
{
    const auto family = control_qp_data::control_qp_family();
    REQUIRE(family.size() >= 3u);

    for(const auto& p : family)
    {
        INFO("horizon " << p.horizon);
        const int n = p.n_variables();
        const int m = p.n_rows();
        REQUIRE(p.A.cols() == n);
        REQUIRE(p.A.rows() == m);
        REQUIRE(p.P.rows() == n);
        REQUIRE(p.P.cols() == n);
        REQUIRE(p.q.size() == n);
        REQUIRE(p.l.size() == m);
        REQUIRE(p.u.size() == m);
        REQUIRE(n == p.n_state * (p.horizon + 1) + p.n_input * p.horizon
                         + p.n_state * p.n_slack_blocks());

        // The solver's stated input convention is a full symmetric P; a
        // triangle would silently halve the state coupling term.
        const control_qp::sparse_type sym = control_qp::sparse_type(p.P.transpose()) - p.P;
        CHECK((sym.coeffs().size() == 0 || sym.coeffs().cwiseAbs().maxCoeff() == 0.0));

        for(int j = 0; j < p.n_state; ++j)
        {
            const int r = p.initial_row(j);
            CHECK(p.l[r] == p.u[r]);
            CHECK(p.l[r] == p.x0[j]);
        }

        Eigen::SparseMatrix<double, Eigen::RowMajor> by_row = p.A;
        Eigen::VectorXd expected = Eigen::VectorXd::Zero(n);
        std::vector<int> touched;
        double worst_row_error = 0.0;
        for(int k = 0; k < p.horizon; ++k)
            for(int i = 0; i < p.n_state; ++i)
            {
                const int r = p.dynamics_row(k, i);
                REQUIRE(p.l[r] == p.u[r]);
                REQUIRE(p.u[r] - p.l[r] <= equality_width);

                touched.clear();
                const int self = p.state_index(k + 1, i);
                expected[self] = 1.0;
                touched.push_back(self);
                for(int j = 0; j < p.n_state; ++j)
                    if(p.Ad(i, j) != 0.0)
                    {
                        const int c = p.state_index(k, j);
                        expected[c] -= p.Ad(i, j);
                        touched.push_back(c);
                    }
                for(int j = 0; j < p.n_input; ++j)
                    if(p.Bd(i, j) != 0.0)
                    {
                        const int c = p.input_index(k, j);
                        expected[c] -= p.Bd(i, j);
                        touched.push_back(c);
                    }

                int stored = 0;
                for(Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(by_row, r); it;
                    ++it)
                {
                    ++stored;
                    worst_row_error =
                        std::max(worst_row_error,
                                 std::abs(it.value() - expected[static_cast<int>(it.col())]));
                }
                CHECK(stored == static_cast<int>(touched.size()));
                for(int c : touched)
                    expected[c] = 0.0;
            }
        CHECK(worst_row_error == 0.0);
    }

    const auto& big = family.back();
    REQUIRE(big.A.cols() >= 500);
    REQUIRE(density(big.A) < 0.02);

    std::printf("[control-shaped family] largest member n=%d m=%d nnz(A)=%d density=%.3e\n",
                static_cast<int>(big.A.cols()), static_cast<int>(big.A.rows()),
                static_cast<int>(big.A.nonZeros()), density(big.A));
}

// ---------------------------------------------------------------------------
// The solve leg. The dense solver is not usable as an oracle at these sizes, and
// asking the sparse solver to grade itself would be worthless, so optimality is
// established from the returned (x, y) against the original unscaled data:
// stationarity, primal feasibility, and a sign-and-support complementarity
// statement. Reaching the solved status on every member is simultaneously the
// end-to-end witness that the no-pivot quasi-definite LDL^T factorization holds
// at real control scale -- a factorization failure surfaces as a non-solved
// status, which this case does not tolerate.
//
// Committed bound 1e-6 on all three quantities. Measured worst cases over the
// family: stationarity 1.20e-15, feasibility 6.71e-16, complementarity 2.22e-16
// with zero violating rows -- nine orders of headroom, so the bound is an
// accuracy gate rather than a bound fitted to an observation.
// ---------------------------------------------------------------------------
TEST_CASE("sparse_admm_qp solves the control-shaped family against an independent optimality check",
          "[qp][sparse_control]")
{
    double worst_stationarity = 0.0;
    double worst_feasibility = 0.0;
    double worst_complementarity = 0.0;

    for(const auto& p : control_qp_data::control_qp_family())
    {
        INFO("horizon " << p.horizon);
        sparse_admm_qp_solver<double> solver;
        auto r = solver.solve(p.P, p.q, p.A, p.l, p.u);
        REQUIRE(r);
        CHECK(r->status == qp_solve_status::solved);

        const optimality o = check_optimality(p, p.q, p.l, p.u, r->x, r->y, optimality_tol);
        CHECK(o.stationarity <= optimality_tol);
        CHECK(o.feasibility <= optimality_tol);
        CHECK(o.complementarity_violations == 0);

        worst_stationarity = std::max(worst_stationarity, o.stationarity);
        worst_feasibility = std::max(worst_feasibility, o.feasibility);
        worst_complementarity = std::max(worst_complementarity, o.complementarity);

        // A check that cannot fail certifies nothing, so the instrument is
        // shown to have teeth on the very iterate it just passed.
        Eigen::VectorXd nudged = r->x;
        nudged[p.state_index(p.horizon / 2, 0)] += 1e-3;
        CHECK_FALSE(check_optimality(p, p.q, p.l, p.u, nudged, r->y, optimality_tol)
                        .within(optimality_tol));
    }

    std::printf("[control-shaped solve] worst stationarity %.3e, feasibility %.3e, "
                "complementarity %.3e\n",
                worst_stationarity, worst_feasibility, worst_complementarity);
}

// ---------------------------------------------------------------------------
// The receding-horizon rollout, on the mid-sized member so ten warm steps plus
// ten cold comparisons stay inside the suite's time budget. Each step updates
// ONLY the vectors -- the initial-condition equality slides onto the state the
// previous solution implies, and the tracking target advances one sample -- and
// enters through the vectors-only resolve, which is precisely the per-step
// update a linear-MPC consumer performs.
//
// This case asserts nothing about heap traffic or elapsed time, and that is
// deliberate: the solver makes no allocation claim and reads no clock, so a
// control-shaped rollout implying a real-time property would be dishonest.
//
// Measured over ten steps: 1625 warm iterations against 2025 for the same ten
// problems posed independently on fresh instances, with worst stationarity
// 1.16e-15, feasibility 8.61e-16 and complementarity 5.55e-16 against the same
// committed 1e-6 bound. The warm and cold answers agree to 0.0 at every step,
// which is a stronger statement than the iteration count and comes for free
// from the polish resolving the same active set.
// ---------------------------------------------------------------------------
TEST_CASE("sparse_admm_qp warm-starts a control-shaped receding-horizon rollout",
          "[qp][sparse_control]")
{
    constexpr int steps = 10;

    const auto family = control_qp_data::control_qp_family();
    REQUIRE(family.size() >= 2u);
    const control_qp& p = family[1];

    Eigen::VectorXd q = p.q;
    Eigen::VectorXd l = p.l;
    Eigen::VectorXd u = p.u;

    sparse_admm_qp_solver<double> solver;
    auto posed = solver.solve(p.P, q, p.A, l, u);
    REQUIRE(posed);
    REQUIRE(posed->status == qp_solve_status::solved);

    const double scaling_cost = solver.scaling_cost();
    const Eigen::VectorXd scaling_primal = solver.scaling_primal();
    const Eigen::VectorXd scaling_dual = solver.scaling_dual();

    Eigen::VectorXd applied_state = posed->x.segment(p.n_state, p.n_state);
    int warm_iterations = 0;
    int cold_iterations = 0;
    double worst_stationarity = 0.0;
    double worst_feasibility = 0.0;
    double worst_complementarity = 0.0;

    for(int step = 1; step <= steps; ++step)
    {
        INFO("rollout step " << step);
        for(int j = 0; j < p.n_state; ++j)
        {
            l[p.initial_row(j)] = applied_state[j];
            u[p.initial_row(j)] = applied_state[j];
        }
        q = p.gradient_for(p.reference_at(step));

        auto warm = solver.resolve(q, l, u);
        REQUIRE(warm);
        CHECK(warm->status == qp_solve_status::solved);

        const optimality o = check_optimality(p, q, l, u, warm->x, warm->y, optimality_tol);
        CHECK(o.stationarity <= optimality_tol);
        CHECK(o.feasibility <= optimality_tol);
        CHECK(o.complementarity_violations == 0);
        worst_stationarity = std::max(worst_stationarity, o.stationarity);
        worst_feasibility = std::max(worst_feasibility, o.feasibility);
        worst_complementarity = std::max(worst_complementarity, o.complementarity);

        // Exact equality, not a tolerance: the claim is that no step
        // re-equilibrated or re-posed, and an approximate match would not say
        // that.
        CHECK(solver.scaling_cost() == scaling_cost);
        CHECK((solver.scaling_primal() - scaling_primal).cwiseAbs().maxCoeff() == 0.0);
        CHECK((solver.scaling_dual() - scaling_dual).cwiseAbs().maxCoeff() == 0.0);

        sparse_admm_qp_solver<double> fresh;
        auto cold = fresh.solve(p.P, q, p.A, l, u);
        REQUIRE(cold);
        CHECK(cold->status == qp_solve_status::solved);

        warm_iterations += warm->iterations;
        cold_iterations += cold->iterations;
        CHECK((warm->x - cold->x).cwiseAbs().maxCoeff() <= 1e-6);

        applied_state = warm->x.segment(p.n_state, p.n_state);
    }

    CHECK(warm_iterations < cold_iterations);

    std::printf("[control-shaped rollout] %d steps, warm %d iterations vs cold %d; worst "
                "stationarity %.3e, feasibility %.3e, complementarity %.3e\n",
                steps, warm_iterations, cold_iterations, worst_stationarity, worst_feasibility,
                worst_complementarity);
}
