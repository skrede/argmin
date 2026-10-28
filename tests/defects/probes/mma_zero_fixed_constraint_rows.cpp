#include "argmin/solver/options.h"
#include "argmin/solver/mma_policy.h"

#include <Eigen/Core>

#include <cstdint>

namespace
{

using vector4 = Eigen::Vector<double, 4>;

struct bound_only_bowl
{
    static constexpr std::int32_t problem_dimension = 4;

    static constexpr std::int32_t constraint_count = 0;

    std::int32_t dimension() const { return 4; }

    std::int32_t num_equality() const { return 0; }

    std::int32_t num_inequality() const { return 0; }

    double value(const vector4& x) const { return x.squaredNorm(); }

    void gradient(const vector4& x, vector4& g) const { g = 2.0 * x; }

    void constraints(const vector4&, Eigen::VectorXd&) const {}

    void constraint_jacobian(const vector4&, Eigen::MatrixXd&) const {}

    vector4 lower_bounds() const { return vector4::Constant(-1.0); }

    vector4 upper_bounds() const { return vector4::Constant(1.0); }
};

}

void mma_steps_with_a_fixed_zero_constraint_count()
{
    argmin::mma_policy<double, 4> policy;
    const bound_only_bowl problem;
    auto s = policy.init(problem, vector4::Constant(0.5), argmin::solver_options<>{});
    policy.step(s);
}
