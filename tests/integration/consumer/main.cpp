#include "argmin/argmin.h"

#include "argmin/test_functions/rosenbrock.h"

#include <cmath>

namespace
{

bool float_lbfgsb_converges()
{
    const argmin::rosenbrock<float, 2> problem;
    argmin::solver_options<argmin::default_convergence<float>> opts{};
    opts.max_iterations = 1000;
    argmin::step_budget_solver solver{argmin::lbfgsb_policy<float>{}, problem, problem.initial_point(), opts};
    const auto result = solver.solve();
    return result.status == argmin::solver_status::converged && result.x.allFinite() && std::isfinite(result.objective_value);
}

}

int main()
{
    argmin::vector<double> v(3);
    v << 1.0, 2.0, 3.0;
    if(v.size() != 3)
        return 1;
    return float_lbfgsb_converges() ? 0 : 2;
}
