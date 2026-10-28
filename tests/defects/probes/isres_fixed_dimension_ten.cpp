#include "argmin/solver/options.h"
#include "argmin/solver/isres_policy.h"

#include <Eigen/Core>

#include <cstdint>

namespace
{

using vector10 = Eigen::Vector<double, 10>;

struct boxed_row_problem
{
    static constexpr std::int32_t problem_dimension = 10;

    std::int32_t dimension() const { return 10; }

    std::int32_t num_equality() const { return 0; }

    std::int32_t num_inequality() const { return 1; }

    double value(const vector10& x) const { return x.squaredNorm(); }

    void constraints(const vector10& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = x.sum() - 1.0;
    }

    vector10 lower_bounds() const { return vector10::Constant(-1.0); }

    vector10 upper_bounds() const { return vector10::Constant(1.0); }
};

}

void isres_steps_at_ten_fixed_variables()
{
    argmin::isres_policy<double, 10> policy;
    const boxed_row_problem problem;
    auto s = policy.init(problem, vector10::Constant(0.5), argmin::solver_options<>{});
    policy.step(s);
}
