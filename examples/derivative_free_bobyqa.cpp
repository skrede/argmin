// Derivative-free minimization with BOBYQA.
//
// Minimizes the Booth function on a box without ever providing a gradient --
// the problem type exposes only value() and bounds. BOBYQA builds an internal
// quadratic model from samples, so it fits problems whose derivatives are
// unavailable or expensive.

#include "example_common.h"

#include <argmin/argmin.h>
#include <argmin/solver/bobyqa_policy.h>

#include <Eigen/Core>
#include <iostream>

struct booth_boxed
{
    static constexpr int problem_dimension = 2;

    Eigen::Vector<double, 2> lb;
    Eigen::Vector<double, 2> ub;

    int dimension() const { return 2; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        const double t1 = x[0] + 2.0 * x[1] - 7.0;
        const double t2 = 2.0 * x[0] + x[1] - 5.0;
        return t1 * t1 + t2 * t2;
    }

    Eigen::Vector<double, 2> lower_bounds() const { return lb; }
    Eigen::Vector<double, 2> upper_bounds() const { return ub; }
};

int main()
{
    using namespace argmin;

    booth_boxed problem{
        .lb = Eigen::Vector<double, 2>{{-10.0, -10.0}},
        .ub = Eigen::Vector<double, 2>{{10.0, 10.0}},
    };
    Eigen::VectorXd x0{{0.0, 0.0}};

    solver_options opts;
    opts.max_iterations = 300;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-12);

    step_budget_solver solver{bobyqa_policy{}, problem, x0, opts};
    auto result = solver.solve(opts);

    std::cout << "status: " << example::status_name(result.status) << '\n'
              << "x:      " << result.x.transpose()                << "  (optimum: (1, 3))\n"
              << "f(x):   " << result.objective_value              << '\n';
}
