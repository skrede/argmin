// Nonlinear least-squares with Levenberg-Marquardt.
//
// Fits the Rosenbrock residual form r(x) = [1 - x0, sqrt(5)(x1 - x0^2)] by
// minimizing 0.5 * ||r(x)||^2. A least-squares problem exposes residuals and
// their Jacobian (the `least_squares` concept) instead of a gradient.

#include "example_common.h"

#include <argmin/argmin.h>
#include <argmin/solver/lm_policy.h>

#include <Eigen/Core>
#include <cmath>
#include <iostream>

struct rosenbrock_ls
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }
    int num_residuals() const { return 2; }

    // 0.5 * sum of squared residuals -- the scalar the solver reports.
    double value(const Eigen::Vector<double, 2>& x) const
    {
        const double r0 = 1.0 - x(0);
        const double r1 = std::sqrt(5.0) * (x(1) - x(0) * x(0));
        return 0.5 * (r0 * r0 + r1 * r1);
    }

    void residuals(const Eigen::Vector<double, 2>& x, Eigen::VectorXd& r) const
    {
        r(0) = 1.0 - x(0);
        r(1) = std::sqrt(5.0) * (x(1) - x(0) * x(0));
    }

    void jacobian(const Eigen::Vector<double, 2>& x, Eigen::MatrixXd& J) const
    {
        J(0, 0) = -1.0;                          J(0, 1) = 0.0;
        J(1, 0) = -2.0 * std::sqrt(5.0) * x(0);  J(1, 1) = std::sqrt(5.0);
    }
};

int main()
{
    using namespace argmin;

    rosenbrock_ls problem;
    Eigen::VectorXd x0{{-1.0, 1.0}};

    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-12);

    step_budget_solver solver{lm_policy{}, problem, x0, opts};
    auto result = solver.solve(opts);

    std::cout << "status: " << example::status_name(result.status) << '\n'
              << "x:      " << result.x.transpose()                << "  (optimum: (1, 1))\n"
              << "f(x):   " << result.objective_value              << '\n';
}
