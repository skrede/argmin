// Step-level / budgeted execution for control loops.
//
// The same algorithms run in three modes. For an RT control tick or an IK
// frame budget that cannot afford a blocking solve(), the caller owns the
// loop and advances the solver one bounded-work step at a time.
//
//   stepper              -- no internal loop; the caller's scheduler drives it
//   step_budget_solver   -- step() / step_n(budget) / solve() on one policy

#include "example_common.h"

#include <argmin/argmin.h>
#include <argmin/solver/stepper.h>
#include <argmin/solver/lbfgsb_policy.h>
#include <argmin/test_functions/rosenbrock.h>

#include <Eigen/Core>
#include <iostream>

int main()
{
    using namespace argmin;

    rosenbrock<> problem{.n = 2};
    Eigen::VectorXd x0{{-1.2, 1.0}};

    solver_options opts;
    opts.set_gradient_threshold(1e-6);
    opts.max_iterations = 1000;

    // (1) stepper: the purest primitive. The caller's own executor -- here a
    // plain while-loop, in practice a control-loop tick -- decides when to
    // call step(), so the iteration budget is external to the solver.
    stepper st{lbfgsb_policy<>{}, problem, x0, opts};

    std::uint32_t ticks = 0;
    while(st.status() == solver_status::running && ticks < opts.max_iterations)
    {
        st.step();          // exactly one bounded-work iteration per tick
        ++ticks;
    }

    std::cout << "[stepper]  converged: " << example::status_name(st.status())
              << "  ticks: " << ticks
              << "  x: " << st.state().x.transpose() << '\n';

    // (2) step_budget_solver: a loop-owning driver exposing all three modes.
    step_budget_solver driver{lbfgsb_policy<>{}, problem, x0, opts};
    driver.step();                      // a single iteration
    driver.step_n(10);                  // at most 10 more iterations
    auto result = driver.solve(opts);   // run to convergence / max_iterations

    std::cout << "[driver]   status: " << example::status_name(result.status)
              << "  x: " << result.x.transpose() << '\n';
}
