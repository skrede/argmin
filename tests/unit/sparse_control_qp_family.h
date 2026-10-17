#ifndef HPP_GUARD_ARGMIN_TESTS_UNIT_SPARSE_CONTROL_QP_FAMILY_H
#define HPP_GUARD_ARGMIN_TESTS_UNIT_SPARSE_CONTROL_QP_FAMILY_H

// A structured linear-MPC-shaped QP generator: a chain of discrete double
// integrators coupled by springs, condensed into nothing -- the states stay in
// the decision vector and the plant recursion stays in the constraint matrix as
// banded equality rows, which is the sparsity this data exists to exercise.
//
// Decision vector layout, and this ordering IS the banded structure, so it is a
// contract rather than an implementation detail:
//
//     [ x_0 ... x_H | u_0 ... u_{H-1} | s_1 ... s_H ]
//
// states first over the whole horizon, then inputs, then -- when slacks are
// enabled -- one slack per bounded state entry of steps 1 through H. With this
// ordering the dynamics rows touch three consecutive blocks and nothing else.
//
// Constraint rows, in order: the initial-condition equalities fixing x_0; the
// per-step dynamics equalities x_{k+1} - A_d x_k - B_d u_k = 0; two-sided input
// boxes; the state boxes, split into an upper and a lower one-sided row sharing
// one slack column when slacks are enabled; and the slack non-negativity rows.
//
// The plant is a chain of n_input axes, each a double integrator in (p, v) with
// a spring to ground and to its neighbors, discretized by forward Euler on the
// velocity row and the exact double-integrator zero-order-hold term on the
// position row. A_d and B_d are returned alongside the assembled data so a
// consumer can re-derive the equality rows without trusting the assembly.
//
// Reference: Borrelli, Bemporad and Morari, Predictive Control for Linear and
//            Hybrid Systems, Cambridge 2017, Chapter 11 (the non-condensed
//            stacked formulation whose equality block is the plant recursion);
//            Maciejowski, Predictive Control with Constraints, Prentice Hall
//            2002, Section 3.4 (soft state constraints through slack variables
//            penalized linearly and quadratically).

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <cmath>
#include <limits>
#include <random>
#include <vector>
#include <cstdint>

namespace argmin::control_qp_data
{

struct control_qp
{
    using sparse_type = Eigen::SparseMatrix<double, Eigen::ColMajor>;

    int horizon{0};
    int n_state{0};
    int n_input{0};
    bool with_slacks{false};

    Eigen::MatrixXd Ad;
    Eigen::MatrixXd Bd;
    Eigen::VectorXd x0;

    Eigen::VectorXd state_weight_diagonal;
    double state_weight_coupling{0.0};
    double input_weight{0.0};
    double slack_quadratic{0.0};
    double slack_linear{0.0};

    Eigen::VectorXd ref_amplitude;
    Eigen::VectorXd ref_omega;
    Eigen::VectorXd ref_phase;

    sparse_type P;
    sparse_type A;
    Eigen::VectorXd q;
    Eigen::VectorXd l;
    Eigen::VectorXd u;

    int n_slack_blocks() const { return with_slacks ? horizon : 0; }
    int n_variables() const
    {
        return n_state * (horizon + 1) + n_input * horizon + n_state * n_slack_blocks();
    }
    int n_rows() const
    {
        return n_state + n_state * horizon + n_input * horizon + 2 * n_state * horizon
               + n_state * n_slack_blocks();
    }

    int state_index(int k, int j) const { return k * n_state + j; }
    int input_index(int k, int j) const { return n_state * (horizon + 1) + k * n_input + j; }
    int slack_index(int k, int j) const
    {
        return n_state * (horizon + 1) + n_input * horizon + (k - 1) * n_state + j;
    }

    int initial_row(int j) const { return j; }
    int dynamics_row(int k, int j) const { return n_state + k * n_state + j; }
    int input_box_row(int k, int j) const
    {
        return n_state + n_state * horizon + k * n_input + j;
    }
    int state_upper_row(int k, int j) const
    {
        return n_state + n_state * horizon + n_input * horizon + 2 * ((k - 1) * n_state + j);
    }
    int state_lower_row(int k, int j) const { return state_upper_row(k, j) + 1; }
    int slack_row(int k, int j) const
    {
        return n_state + n_state * horizon + n_input * horizon + 2 * n_state * horizon
               + (k - 1) * n_state + j;
    }

    // The tracking target at absolute time index t: a per-axis sinusoid on
    // position with its analytic derivative on velocity, so a rollout can slide
    // the window forward without re-posing anything.
    Eigen::VectorXd reference_at(int time_origin) const
    {
        Eigen::VectorXd r(n_state * (horizon + 1));
        for(int k = 0; k <= horizon; ++k)
        {
            const double t = static_cast<double>(time_origin + k);
            for(int axis = 0; axis < n_input; ++axis)
            {
                const double w = ref_omega[axis];
                const double a = ref_amplitude[axis];
                const double ph = ref_phase[axis];
                r[k * n_state + 2 * axis] = a * std::sin(w * t + ph);
                r[k * n_state + 2 * axis + 1] = a * w * std::cos(w * t + ph);
            }
        }
        return r;
    }

    // -Q * reference on the state blocks, the linear slack penalty on the slack
    // blocks, zero on the inputs: the gradient of the stage cost written around
    // the tracking target.
    Eigen::VectorXd gradient_for(const Eigen::VectorXd& reference) const
    {
        Eigen::VectorXd g = Eigen::VectorXd::Zero(n_variables());
        for(int k = 0; k <= horizon; ++k)
            for(int axis = 0; axis < n_input; ++axis)
            {
                const int p = 2 * axis;
                const int v = p + 1;
                const double rp = reference[k * n_state + p];
                const double rv = reference[k * n_state + v];
                g[state_index(k, p)] =
                    -(state_weight_diagonal[p] * rp + state_weight_coupling * rv);
                g[state_index(k, v)] =
                    -(state_weight_diagonal[v] * rv + state_weight_coupling * rp);
            }
        for(int k = 1; k <= n_slack_blocks(); ++k)
            for(int j = 0; j < n_state; ++j)
                g[slack_index(k, j)] = slack_linear;
        return g;
    }
};

namespace detail
{

inline Eigen::MatrixXd plant_state_matrix(int n_input, double dt, double k_ground, double k_couple)
{
    const int n_state = 2 * n_input;
    Eigen::MatrixXd Ad = Eigen::MatrixXd::Zero(n_state, n_state);
    for(int axis = 0; axis < n_input; ++axis)
    {
        const int p = 2 * axis;
        const int v = p + 1;
        Ad(p, p) = 1.0;
        Ad(p, v) = dt;
        Ad(v, v) = 1.0;
        double self = k_ground;
        if(axis > 0)
        {
            Ad(v, p - 2) = dt * k_couple;
            self += k_couple;
        }
        if(axis + 1 < n_input)
        {
            Ad(v, p + 2) = dt * k_couple;
            self += k_couple;
        }
        Ad(v, p) = -dt * self;
    }
    return Ad;
}

inline Eigen::MatrixXd plant_input_matrix(int n_input, double dt)
{
    Eigen::MatrixXd Bd = Eigen::MatrixXd::Zero(2 * n_input, n_input);
    for(int axis = 0; axis < n_input; ++axis)
    {
        Bd(2 * axis, axis) = 0.5 * dt * dt;
        Bd(2 * axis + 1, axis) = dt;
    }
    return Bd;
}

}

// n_state must be twice n_input: every axis contributes a position and a
// velocity, and the spring coupling is defined between adjacent axes.
inline control_qp make_control_qp(int horizon, int n_state, int n_input, bool with_slacks,
                                  std::uint32_t seed)
{
    constexpr double inf = std::numeric_limits<double>::infinity();
    constexpr double dt = 0.1;
    constexpr double k_ground = 1.0;
    constexpr double k_couple = 0.3;
    constexpr double position_bound = 2.0;
    constexpr double velocity_bound = 1.5;
    constexpr double input_bound = 2.0;

    control_qp p;
    p.horizon = horizon;
    p.n_state = n_state;
    p.n_input = n_input;
    p.with_slacks = with_slacks;
    p.Ad = detail::plant_state_matrix(n_input, dt, k_ground, k_couple);
    p.Bd = detail::plant_input_matrix(n_input, dt);

    p.state_weight_diagonal.resize(n_state);
    for(int axis = 0; axis < n_input; ++axis)
    {
        p.state_weight_diagonal[2 * axis] = 2.0;
        p.state_weight_diagonal[2 * axis + 1] = 1.0;
    }
    p.state_weight_coupling = 0.1;
    p.input_weight = 0.5;
    p.slack_quadratic = 20.0;
    p.slack_linear = 5.0;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> amp(2.2, 3.4);
    std::uniform_real_distribution<double> omg(0.18, 0.32);
    std::uniform_real_distribution<double> pha(0.0, 3.0);
    std::uniform_real_distribution<double> x0d(-0.8, 0.8);
    p.ref_amplitude.resize(n_input);
    p.ref_omega.resize(n_input);
    p.ref_phase.resize(n_input);
    for(int axis = 0; axis < n_input; ++axis)
    {
        p.ref_amplitude[axis] = amp(rng);
        p.ref_omega[axis] = omg(rng);
        p.ref_phase[axis] = pha(rng);
    }
    p.x0.resize(n_state);
    for(int j = 0; j < n_state; ++j)
        p.x0[j] = x0d(rng);

    const int n = p.n_variables();
    const int m = p.n_rows();
    const int slack_blocks = p.n_slack_blocks();

    std::vector<Eigen::Triplet<double>> tp;
    for(int k = 0; k <= horizon; ++k)
        for(int axis = 0; axis < n_input; ++axis)
        {
            const int pi = p.state_index(k, 2 * axis);
            const int vi = p.state_index(k, 2 * axis + 1);
            tp.emplace_back(pi, pi, p.state_weight_diagonal[2 * axis]);
            tp.emplace_back(vi, vi, p.state_weight_diagonal[2 * axis + 1]);
            tp.emplace_back(pi, vi, p.state_weight_coupling);
            tp.emplace_back(vi, pi, p.state_weight_coupling);
        }
    for(int k = 0; k < horizon; ++k)
        for(int j = 0; j < n_input; ++j)
        {
            const int ui = p.input_index(k, j);
            tp.emplace_back(ui, ui, p.input_weight);
        }
    for(int k = 1; k <= slack_blocks; ++k)
        for(int j = 0; j < n_state; ++j)
        {
            const int si = p.slack_index(k, j);
            tp.emplace_back(si, si, p.slack_quadratic);
        }
    p.P = control_qp::sparse_type(n, n);
    p.P.setFromTriplets(tp.begin(), tp.end());
    p.P.makeCompressed();

    p.l = Eigen::VectorXd::Constant(m, -inf);
    p.u = Eigen::VectorXd::Constant(m, inf);
    tp.clear();

    for(int j = 0; j < n_state; ++j)
    {
        const int r = p.initial_row(j);
        tp.emplace_back(r, p.state_index(0, j), 1.0);
        p.l[r] = p.x0[j];
        p.u[r] = p.x0[j];
    }

    for(int k = 0; k < horizon; ++k)
        for(int i = 0; i < n_state; ++i)
        {
            const int r = p.dynamics_row(k, i);
            tp.emplace_back(r, p.state_index(k + 1, i), 1.0);
            for(int j = 0; j < n_state; ++j)
                if(p.Ad(i, j) != 0.0)
                    tp.emplace_back(r, p.state_index(k, j), -p.Ad(i, j));
            for(int j = 0; j < n_input; ++j)
                if(p.Bd(i, j) != 0.0)
                    tp.emplace_back(r, p.input_index(k, j), -p.Bd(i, j));
            p.l[r] = 0.0;
            p.u[r] = 0.0;
        }

    for(int k = 0; k < horizon; ++k)
        for(int j = 0; j < n_input; ++j)
        {
            const int r = p.input_box_row(k, j);
            tp.emplace_back(r, p.input_index(k, j), 1.0);
            p.l[r] = -input_bound;
            p.u[r] = input_bound;
        }

    for(int k = 1; k <= horizon; ++k)
        for(int j = 0; j < n_state; ++j)
        {
            const double bound = (j % 2 == 0) ? position_bound : velocity_bound;
            const int ru = p.state_upper_row(k, j);
            const int rl = p.state_lower_row(k, j);
            tp.emplace_back(ru, p.state_index(k, j), 1.0);
            tp.emplace_back(rl, p.state_index(k, j), 1.0);
            if(with_slacks)
            {
                tp.emplace_back(ru, p.slack_index(k, j), -1.0);
                tp.emplace_back(rl, p.slack_index(k, j), 1.0);
            }
            p.u[ru] = bound;
            p.l[rl] = -bound;
        }

    for(int k = 1; k <= slack_blocks; ++k)
        for(int j = 0; j < n_state; ++j)
        {
            const int r = p.slack_row(k, j);
            tp.emplace_back(r, p.slack_index(k, j), 1.0);
            p.l[r] = 0.0;
        }

    p.A = control_qp::sparse_type(m, n);
    p.A.setFromTriplets(tp.begin(), tp.end());
    p.A.makeCompressed();

    p.q = p.gradient_for(p.reference_at(0));
    return p;
}

// Three horizons spanning a small, a mid and a large member. The largest is the
// one that carries the scale and density claim; the mid one is what the rollout
// runs, so it has to stay affordable across a dozen consecutive solves.
inline std::vector<control_qp> control_qp_family()
{
    std::vector<control_qp> out;
    out.push_back(make_control_qp(10, 6, 3, true, 1000003u));
    out.push_back(make_control_qp(30, 6, 3, true, 1000033u));
    out.push_back(make_control_qp(60, 6, 3, true, 1000037u));
    return out;
}

}

#endif
