// Equality/inequality-constrained minimization with SLSQP.
//
// Defines a small custom constrained problem (bring-your-own, satisfying the
// `constrained` concept) and solves it with Kraft's SLSQP. The problem is
//
//     minimize   x0^2 + x1^2
//     subject to x0 + x1 - 1 = 0        (one equality)
//                x0 >= 0.75             (one inequality, written g(x) >= 0)
//
// The equality-only minimum is (0.5, 0.5); the inequality x0 >= 0.75 is
// active and pushes the solution to (0.75, 0.25) with objective 0.625.

#include "example_common.h"

#include <argmin/argmin.h>
#include <argmin/solver/kraft_slsqp_policy.h>

#include <Eigen/Core>
#include <iostream>

// A problem type satisfies the solver by duck-typing -- no base class. The
// `constrained` concept requires value/gradient (it is also differentiable),
// constraints/constraint_jacobian, and num_equality/num_inequality. Equality
// rows come first, inequality rows second (inequalities in g(x) >= 0 form).
struct constrained_qp
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }
    int num_equality() const { return 1; }
    int num_inequality() const { return 1; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        return x[0] * x[0] + x[1] * x[1];
    }

    void gradient(const Eigen::Vector<double, 2>& x, Eigen::Vector<double, 2>& g) const
    {
        g = 2.0 * x;
    }

    void constraints(const Eigen::Vector<double, 2>& x, auto& c) const
    {
        c[0] = x[0] + x[1] - 1.0;   // equality:   x0 + x1 - 1 == 0
        c[1] = x[0] - 0.75;         // inequality: x0 - 0.75 >= 0  (active at optimum)
    }

    // The constraints are linear, so the Jacobian is constant in x.
    void constraint_jacobian([[maybe_unused]] const Eigen::Vector<double, 2>& x, auto& J) const
    {
        J(0, 0) = 1.0; J(0, 1) = 1.0;   // d/dx of the equality row
        J(1, 0) = 1.0; J(1, 1) = 0.0;   // d/dx of the inequality row
    }
};

int main()
{
    using namespace argmin;

    constrained_qp problem;
    Eigen::VectorXd x0{{2.0, 2.0}};

    solver_options opts;
    opts.max_iterations = 100;
    opts.set_gradient_threshold(1e-6);
    opts.set_step_threshold(1e-12);

    step_budget_solver solver{
        kraft_slsqp_policy<constrained_qp::problem_dimension>{}, problem, x0, opts};
    auto result = solver.solve(opts);

    std::cout << "status: " << example::status_name(result.status) << '\n'
              << "x:      " << result.x.transpose()                << "  (optimum: (0.75, 0.25))\n"
              << "f(x):   " << result.objective_value              << "  (optimum: 0.625)\n"
              << "cv:     " << solver.constraint_violation()       << '\n';
}
