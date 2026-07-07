// Verify that the umbrella header compiles without errors
// and that all module headers are transitively included.

#include "argmin/argmin.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/test_functions/rosenbrock.h"

#include <Eigen/Core>

// Compile-only assertion that the primary documented usage snippets (README
// Quickstart and docs/choosing-a-solver.md) build verbatim. The function is
// never invoked -- instantiation alone is the test. If the documented
// spelling ever stops compiling, this translation unit fails to build,
// catching front-door contract regressions that runtime tests would miss.
[[maybe_unused]] static void documented_snippets_compile()
{
    using namespace argmin;

    // README Quickstart: CTAD from a policy value + problem + x0 + options.
    {
        rosenbrock<> problem{.n = 2};
        Eigen::VectorXd x0{{-1.2, 1.0}};

        solver_options opts;
        opts.set_gradient_threshold(1e-6);
        opts.max_iterations = 500;

        step_budget_solver solver{lbfgsb_policy<>{}, problem, x0, opts};
        auto result = solver.solve(opts);
        (void)result;
    }

    // docs/choosing-a-solver.md: compile-time vs dynamic dimensions. The
    // fully-spelled fixed-dimension form and the CTAD form must both build.
    {
        rosenbrock<double, 2> problem;
        Eigen::VectorXd x0{{-1.2, 1.0}};
        solver_options opts;

        step_budget_solver<lbfgsb_policy<2>, 2, rosenbrock<double, 2>> fast{problem, x0, opts};
        step_budget_solver generic{lbfgsb_policy<>{}, problem, x0, opts};
        (void)fast;
        (void)generic;
    }
}

int main() { return 0; }
