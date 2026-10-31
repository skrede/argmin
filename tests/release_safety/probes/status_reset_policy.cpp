#include "argmin/solver/options.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/result/status.h"
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

struct status_reset_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = status_reset_policy;

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

    argmin::solver_status reset(state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = x0;
        return argmin::solver_status::running;
    }

    argmin::solver_status reset_clear(state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = x0;
        return argmin::solver_status::running;
    }
};

}

void drive_a_policy_whose_reset_returns_a_status()
{
    const plane_bowl problem;
    argmin::step_budget_solver solver{status_reset_policy{}, problem, Eigen::Vector2d::Zero().eval(), argmin::solver_options<>{}};
    solver.step();
}
