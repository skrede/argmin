#include "argmin/solver/options.h"
#include "argmin/solver/cmaes_policy.h"

#include <Eigen/Core>

#include <cstdint>

namespace
{

using vector33 = Eigen::Vector<double, 33>;

struct box_problem
{
    static constexpr std::int32_t problem_dimension = 33;

    std::int32_t dimension() const { return 33; }

    double value(const vector33& x) const { return x.squaredNorm(); }

    vector33 lower_bounds() const { return vector33::Constant(-1.0); }

    vector33 upper_bounds() const { return vector33::Constant(1.0); }
};

}

void cmaes_steps_at_thirty_three_fixed_variables()
{
    argmin::cmaes_policy<double, 33> policy;
    const box_problem problem;
    auto s = policy.init(problem, vector33::Constant(0.5), argmin::solver_options<>{});
    policy.step(s);
}
