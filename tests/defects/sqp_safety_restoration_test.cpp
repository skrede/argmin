#include "argmin/detail/restoration.h"
#include "argmin/detail/filter_restoration.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <catch2/catch_test_macros.hpp>

#include <limits>

using namespace argmin;

namespace
{

constexpr double inf = std::numeric_limits<double>::infinity();

struct restoration_workspace
{
    restoration_workspace() : c(2), c_trial(2), J(2, 1), JtJ(1, 1), rhs(1), dx(1), x_trial(1), ldlt(1) {}

    Eigen::VectorXd c;
    Eigen::VectorXd c_trial;
    Eigen::MatrixXd J;
    Eigen::MatrixXd JtJ;
    Eigen::VectorXd rhs;
    Eigen::VectorXd dx;
    Eigen::VectorXd x_trial;
    Eigen::LDLT<Eigen::MatrixXd> ldlt;
};

// The rows x >= 0 and x + 1 >= 0 handed over as raw values, the way filter_trsqp builds its restoration input.
detail::lm_restoration_result<double> restore_inequalities(Eigen::VectorXd& x)
{
    restoration_workspace ws;
    const auto cfn = [](const Eigen::VectorXd& v, Eigen::VectorXd& c) { c = Eigen::Vector2d(v[0], v[0] + 1.0); };
    const auto jfn = [](const Eigen::VectorXd&, Eigen::MatrixXd& J) { J = Eigen::Vector2d::Ones(); };
    const Eigen::VectorXd lower = Eigen::VectorXd::Constant(1, -inf);
    const Eigen::VectorXd upper = Eigen::VectorXd::Constant(1, inf);
    return detail::feasibility_restoration<double, Eigen::Dynamic>(x, cfn, jfn, lower, upper, 50, 1e-3, 1e-6, ws.c, ws.c_trial, ws.J, ws.JtJ, ws.rhs, ws.dx, ws.x_trial, ws.ldlt);
}

// The single equality row scale * (x - 1) = 0.
struct scaled_row
{
    void eval_constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c_eq, Eigen::VectorXd& c_ineq) const
    {
        c_eq = Eigen::VectorXd::Constant(1, scale * (x[0] - 1.0));
        c_ineq.resize(0);
    }

    void eval_constraint_jacobians(const Eigen::VectorXd&, Eigen::MatrixXd& J_eq, Eigen::MatrixXd& J_ineq) const
    {
        J_eq = Eigen::MatrixXd::Constant(1, 1, scale);
        J_ineq.resize(0, 1);
    }

    double scale;
};

bool l1_leaf_restores(double scale, double x0)
{
    const Eigen::VectorXd x = Eigen::VectorXd::Constant(1, x0);
    const Eigen::VectorXd none = Eigen::VectorXd::Zero(1);
    const Eigen::VectorXd lower = Eigen::VectorXd::Constant(1, -inf);
    const Eigen::VectorXd upper = Eigen::VectorXd::Constant(1, inf);
    return detail::restore_l1<double, Eigen::Dynamic>(scaled_row{scale}, x, none, lower, upper).success;
}

}

TEST_CASE("feasibility restoration reaches a point that satisfies the inequality rows", "[known-defect][!shouldfail]")
{
    Eigen::VectorXd x = Eigen::VectorXd::Constant(1, -2.0);
    const auto outcome = restore_inequalities(x);
    CHECK((outcome.status == detail::restoration_status::converged && x[0] >= 0.0));
}

TEST_CASE("the L1 restoration leaf accepts a start already inside the feasibility tolerance", "[known-defect][!shouldfail]")
{
    CHECK(l1_leaf_restores(1.0, 1.0 + 1e-9));
}

TEST_CASE("the L1 restoration leaf reaches the same verdict when its row is rescaled", "[known-defect][!shouldfail]")
{
    const bool unit = l1_leaf_restores(1.0, 0.0);
    CHECK((l1_leaf_restores(1e-6, 0.0) == unit && l1_leaf_restores(1e6, 0.0) == unit));
}
