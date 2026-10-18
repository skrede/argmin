#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/filter_nw_sqp_policy.h"
#include "argmin/solver/filter_trsqp_policy.h"

#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <utility>

namespace
{

template <int Neq, int Nineq>
struct dyn_problem
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int n;

    int dimension() const { return n; }

    int num_equality() const { return Neq; }

    int num_inequality() const { return Nineq; }

    double value(const Eigen::VectorXd& x) const { return 0.5 * x.squaredNorm() + x.sum(); }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        g = x;
        g.array() += 1.0;
    }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(Neq + Nineq);
        if constexpr(Neq > 0)
            c[0] = x.sum() - 1.0;
        if constexpr(Neq > 1)
            c[1] = x[0] - x[1] - 0.5;
        if constexpr(Nineq > 0)
            c[Neq] = 10.0 - x[0];
    }

    void constraint_jacobian(const Eigen::VectorXd&, Eigen::MatrixXd& J) const
    {
        J.setZero(Neq + Nineq, n);
        if constexpr(Neq > 0)
            J.row(0).setOnes();
        if constexpr(Neq > 1)
        {
            J(1, 0) = 1.0;
            J(1, 1) = -1.0;
        }
        if constexpr(Nineq > 0)
            J(Neq, 0) = -1.0;
    }
};

// Seeds the stack a following frame occupies, so an indeterminate enum copy has an invalid value
// for UBSan to see.
void dirty_stack()
{
    std::array<volatile std::uint8_t, 65536> junk;
    for(volatile std::uint8_t& byte : junk)
        byte = 0xAB;
}

template <typename Problem>
using point = Eigen::Vector<double, Problem::problem_dimension>;

template <typename Policy, typename Problem>
bool construct_move_step(Policy policy, const Problem& problem, const point<Problem>& x0)
{
    const argmin::solver_options<> opts{};
    argmin::step_budget_solver built{std::move(policy), problem, x0, opts};
    argmin::step_budget_solver moved{std::move(built)};
    for(std::int32_t i = 0; i < 3; ++i)
        moved.step();
    return moved.state().x.allFinite();
}

template <typename Policy, typename Problem>
bool on_dirty_stack(const Policy& policy, const Problem& problem, const point<Problem>& x0)
{
    void (*volatile dirty)() = dirty_stack;
    bool (*volatile run)(Policy, const Problem&, const point<Problem>&) = construct_move_step<Policy, Problem>;
    dirty();
    return run(policy, problem, x0);
}

template <typename Policy>
void check_every_constraint_shape(const Policy& policy)
{
    const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(6, 0.3);
    CHECK(on_dirty_stack(policy, dyn_problem<2, 1>{6}, x0));
    CHECK(on_dirty_stack(policy, dyn_problem<0, 1>{6}, x0));
    CHECK(on_dirty_stack(policy, dyn_problem<0, 0>{6}, x0));
}

const argmin::hs071<> hs071_problem{};
constexpr int hs071_n = argmin::hs071<>::problem_dimension;

}

TEST_CASE("decomposition state is defined before the first factorization -- nw_sqp", "[decomposition_init]")
{
    check_every_constraint_shape(argmin::nw_sqp_policy<double>{});
    CHECK(on_dirty_stack(argmin::nw_sqp_policy<double, hs071_n>{}, hs071_problem, hs071_problem.initial_point()));
}

TEST_CASE("decomposition state is defined before the first factorization -- filter_nw_sqp", "[decomposition_init]")
{
    check_every_constraint_shape(argmin::filter_nw_sqp_policy<double>{});
    CHECK(on_dirty_stack(argmin::filter_nw_sqp_policy<double, hs071_n>{}, hs071_problem,
                         hs071_problem.initial_point()));
}

TEST_CASE("decomposition state is defined before the first factorization -- tr_sqp", "[decomposition_init]")
{
    check_every_constraint_shape(argmin::tr_sqp_policy<double>{});
    CHECK(on_dirty_stack(argmin::tr_sqp_policy<double, hs071_n>{}, hs071_problem, hs071_problem.initial_point()));
}

TEST_CASE("decomposition state is defined before the first factorization -- filter_trsqp", "[decomposition_init]")
{
    check_every_constraint_shape(argmin::filter_trsqp_policy<double>{});
    CHECK(on_dirty_stack(argmin::filter_trsqp_policy<double, hs071_n>{}, hs071_problem,
                         hs071_problem.initial_point()));

    argmin::filter_trsqp_policy<double> without_restoration{};
    without_restoration.options.restoration_max_iter = 0;
    check_every_constraint_shape(without_restoration);
}
