#include "eigen_trap.h"

#include "argmin/solver/options.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/filter_trsqp_policy.h"
#include "argmin/solver/filter_nw_sqp_policy.h"

#include "argmin/result/status.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <cstdint>

using namespace argmin;

namespace
{

using vector2 = Eigen::Vector2d;

// f = (x0 + 1)^2 + (x1 + 1)^2 subject to x0 >= 0, x1 >= 0 and x0 + x1 >= 0. All three rows are active at the
// minimizer (0, 0), so the active set holds more rows than there are variables.
struct vertex_rows
{
    static constexpr std::int32_t problem_dimension = 2;

    std::int32_t dimension() const { return 2; }

    std::int32_t num_equality() const { return 0; }

    std::int32_t num_inequality() const { return 3; }

    double value(const vector2& x) const { return (x + vector2::Ones()).squaredNorm(); }

    void gradient(const vector2& x, vector2& g) const { g = 2.0 * (x + vector2::Ones()); }

    template <typename Values>
    void constraints(const vector2& x, Values& c) const
    {
        c = normals() * x;
    }

    template <typename Jacobian>
    void constraint_jacobian(const vector2&, Jacobian& J) const
    {
        J = normals();
    }

    static Eigen::Matrix<double, 3, 2> normals() { return (Eigen::Matrix<double, 3, 2>() << 1.0, 0.0, 0.0, 1.0, 1.0, 1.0).finished(); }
};

struct declared_vertex_rows : vertex_rows
{
    static constexpr std::int32_t constraint_count = 3;
};

// The same objective inside the box [-5, 5]^2 with no constraint rows at all, minimized at (-1, -1).
struct boxed_bowl
{
    static constexpr std::int32_t problem_dimension = 2;
    static constexpr std::int32_t constraint_count = 0;

    std::int32_t dimension() const { return 2; }

    std::int32_t num_equality() const { return 0; }

    std::int32_t num_inequality() const { return 0; }

    double value(const vector2& x) const { return (x + vector2::Ones()).squaredNorm(); }

    void gradient(const vector2& x, vector2& g) const { g = 2.0 * (x + vector2::Ones()); }

    template <typename Values>
    void constraints(const vector2&, Values& c) const
    {
        c.resize(0);
    }

    template <typename Jacobian>
    void constraint_jacobian(const vector2&, Jacobian& J) const
    {
        J.resize(0, 2);
    }

    vector2 lower_bounds() const { return vector2::Constant(-5.0); }

    vector2 upper_bounds() const { return vector2::Constant(5.0); }
};

template <typename Policy, typename Problem>
void check_converges(const Problem& problem, const vector2& minimizer)
{
    step_budget_solver solver{Policy{}, problem, vector2(0.5, 0.5), solver_options<>{}};
    const auto result = solver.solve(solver_options<>{});
    CHECK(result.status == solver_status::converged);
    CHECK((result.x - minimizer).norm() < 1e-8);
}

}

TEMPLATE_TEST_CASE("an SQP policy recovers multipliers at a vertex with more active rows than variables", "[release-safety][kkt-vertex]", (nw_sqp_policy<double, 2>), (filter_nw_sqp_policy<double, 2>), (filter_slsqp_policy<double, 2>), (tr_sqp_policy<double, 2>), (filter_trsqp_policy<double, 2>))
{
    SECTION("with the constraint count declared")
    {
        check_converges<TestType>(declared_vertex_rows{}, vector2::Zero());
    }
    SECTION("without a declared constraint count")
    {
        check_converges<TestType>(vertex_rows{}, vector2::Zero());
    }
}

TEST_CASE("an SQP policy solves a fixed-dimension problem that declares no constraint rows", "[release-safety][kkt-vertex]")
{
    check_converges<nw_sqp_policy<double, 2>>(boxed_bowl{}, -vector2::Ones());
}
