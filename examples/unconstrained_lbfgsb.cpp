// Unconstrained smooth minimization with L-BFGS-B.
//
// Minimizes the 2-D Rosenbrock function from a hard starting point. This is
// the canonical "first solve": pick a policy, wrap it in a step_budget_solver,
// call solve(), read the result.

#include "example_common.h"

#include <argmin/argmin.h>
#include <argmin/solver/lbfgsb_policy.h>
#include <argmin/test_functions/rosenbrock.h>

#include <Eigen/Core>
#include <iostream>

int main()
{
    using namespace argmin;

    // A built-in test function; ".n = 2" selects the 2-D instance.
    rosenbrock<> problem{.n = 2};
    Eigen::VectorXd x0{{-1.2, 1.0}};

    solver_options opts;
    opts.set_gradient_threshold(1e-6);
    opts.max_iterations = 200;

    step_budget_solver solver{lbfgsb_policy<>{}, problem, x0, opts};
    auto result = solver.solve(opts);

    std::cout << "status: " << example::status_name(result.status) << '\n'
              << "x:      " << result.x.transpose()                << '\n'
              << "f(x):   " << result.objective_value              << "  (optimum: 0 at (1, 1))\n";
}
