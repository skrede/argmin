#include "argmin/solver/options.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/result/step_result.h"

#include <Eigen/Core>

#include <cstdint>

namespace
{

struct plane_bowl
{
    static constexpr std::int32_t problem_dimension = 2;

    std::int32_t dimension() const { return 2; }

    double value(const Eigen::Vector2d& x) const { return x.squaredNorm(); }
};

struct void_reset_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = void_reset_policy;

    struct state_type
    {
        Eigen::VectorXd x;
    };

    template <typename Problem, typename Convergence>
    state_type init(const Problem&, const Eigen::VectorXd& x0, const argmin::solver_options<Convergence>&)
    {
        return state_type{.x = x0};
    }

    argmin::step_result<double> step(state_type&) { return {}; }

    void reset(state_type& s, const Eigen::VectorXd& x0) { s.x = x0; }

    void reset_clear(state_type& s, const Eigen::VectorXd& x0) { s.x = x0; }
};

}

void drive_a_policy_whose_reset_returns_nothing()
{
    const plane_bowl problem;
    argmin::step_budget_solver solver{void_reset_policy{}, problem, Eigen::Vector2d::Zero().eval(), argmin::solver_options<>{}};
    solver.step();
}
