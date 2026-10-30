#include "argmin/solver/options.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include <Eigen/Core>

#include <cstdint>

namespace
{

using vector2 = Eigen::Vector<double, 2>;

struct plane_bowl
{
    static constexpr std::int32_t problem_dimension = 2;

    std::int32_t dimension() const { return 2; }

    double value(const vector2& x) const { return x.squaredNorm(); }

    void gradient(const vector2& x, vector2& g) const { g = 2.0 * x; }
};

}

void construct_from_a_start_of_the_solver_length()
{
    const plane_bowl problem;
    argmin::step_budget_solver solver{argmin::lbfgsb_policy<double, 2>{}, problem, Eigen::Vector2d::Zero().eval(), argmin::solver_options<>{}};
    solver.step();
}
