// Concept satisfaction compile-time tests.
//
// These verify that the concept hierarchy is correctly defined and that
// test function types satisfy the expected concepts.
//
// If this file compiles, all assertions pass. If any assertion fails,
// the build fails with a clear concept-violation error.

#include "nablapp/formulation/concepts.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/solver/options.h"
#include "nablapp/solver/nw_sqp_policy.h"
#include "nablapp/solver/kraft_slsqp_policy.h"
#include "nablapp/solver/lbfgsb_policy.h"
#include "nablapp/solver/bobyqa_policy.h"
#include "nablapp/solver/cmaes_policy.h"
#include "nablapp/solver/gcmma_policy.h"
#include "nablapp/solver/lm_policy.h"
#include "nablapp/solver/mma_policy.h"
#include "nablapp/solver/cobyla_policy.h"
#include "nablapp/solver/isres_policy.h"
#include "nablapp/solver/augmented_lagrangian_policy.h"
#include "nablapp/test_functions/booth.h"
#include "nablapp/test_functions/beale.h"
#include "nablapp/test_functions/ackley.h"
#include "nablapp/test_functions/schwefel.h"
#include "nablapp/test_functions/griewank.h"
#include "nablapp/test_functions/rastrigin.h"
#include "nablapp/test_functions/rosenbrock.h"
#include "nablapp/test_functions/himmelblau.h"
#include "nablapp/test_functions/hock_schittkowski.h"
#include "nablapp/test_functions/more_garbow_hillstrom.h"

// ---------------------------------------------------------------------------
// Positive tests: all six test functions satisfy objective and differentiable
// ---------------------------------------------------------------------------

static_assert(nablapp::objective<nablapp::rosenbrock<>>);
static_assert(nablapp::differentiable<nablapp::rosenbrock<>>);

static_assert(nablapp::objective<nablapp::booth<>>);
static_assert(nablapp::differentiable<nablapp::booth<>>);

static_assert(nablapp::objective<nablapp::beale<>>);
static_assert(nablapp::differentiable<nablapp::beale<>>);

static_assert(nablapp::objective<nablapp::himmelblau<>>);
static_assert(nablapp::differentiable<nablapp::himmelblau<>>);

static_assert(nablapp::objective<nablapp::rastrigin<>>);
static_assert(nablapp::differentiable<nablapp::rastrigin<>>);

static_assert(nablapp::objective<nablapp::ackley<>>);
static_assert(nablapp::differentiable<nablapp::ackley<>>);

// ---------------------------------------------------------------------------
// Float scalar support (D-07)
// ---------------------------------------------------------------------------

static_assert(nablapp::objective<nablapp::rosenbrock<float>, float>);
static_assert(nablapp::differentiable<nablapp::rosenbrock<float>, float>);

static_assert(nablapp::objective<nablapp::booth<float>, float>);
static_assert(nablapp::differentiable<nablapp::booth<float>, float>);

static_assert(nablapp::objective<nablapp::beale<float>, float>);
static_assert(nablapp::differentiable<nablapp::beale<float>, float>);

static_assert(nablapp::objective<nablapp::himmelblau<float>, float>);
static_assert(nablapp::differentiable<nablapp::himmelblau<float>, float>);

static_assert(nablapp::objective<nablapp::rastrigin<float>, float>);
static_assert(nablapp::differentiable<nablapp::rastrigin<float>, float>);

static_assert(nablapp::objective<nablapp::ackley<float>, float>);
static_assert(nablapp::differentiable<nablapp::ackley<float>, float>);

// ---------------------------------------------------------------------------
// Test functions should NOT satisfy higher-order concepts
// ---------------------------------------------------------------------------

static_assert(!nablapp::second_order<nablapp::rosenbrock<>>);
static_assert(!nablapp::bound_constrained<nablapp::rosenbrock<>>);
static_assert(!nablapp::constrained<nablapp::rosenbrock<>>);
static_assert(!nablapp::least_squares<nablapp::rosenbrock<>>);

// ---------------------------------------------------------------------------
// Negative tests: mock types that intentionally lack required methods
// ---------------------------------------------------------------------------

namespace
{

// A type with only value() -- satisfies objective but NOT differentiable
struct objective_only
{
    double value(const Eigen::VectorXd&) const { return 0.0; }
    static constexpr int problem_dimension = nablapp::dynamic_dimension;

    int dimension() const { return 1; }
};

static_assert(nablapp::objective<objective_only>);
static_assert(!nablapp::differentiable<objective_only>);
static_assert(!nablapp::second_order<objective_only>);
static_assert(!nablapp::bound_constrained<objective_only>);
static_assert(!nablapp::constrained<objective_only>);
static_assert(!nablapp::least_squares<objective_only>);

// A type with value() + gradient() -- differentiable but NOT second_order
struct differentiable_only
{
    double value(const Eigen::VectorXd&) const { return 0.0; }
    void gradient(const Eigen::VectorXd&, Eigen::VectorXd&) const {}
    static constexpr int problem_dimension = nablapp::dynamic_dimension;

    int dimension() const { return 1; }
};

static_assert(nablapp::differentiable<differentiable_only>);
static_assert(!nablapp::second_order<differentiable_only>);

// An empty struct -- satisfies nothing
struct empty_type {};

static_assert(!nablapp::objective<empty_type>);
static_assert(!nablapp::differentiable<empty_type>);
static_assert(!nablapp::second_order<empty_type>);
static_assert(!nablapp::bound_constrained<empty_type>);
static_assert(!nablapp::constrained<empty_type>);
static_assert(!nablapp::least_squares<empty_type>);

// ---------------------------------------------------------------------------
// Concept refinement: a type satisfying all concepts simultaneously
// ---------------------------------------------------------------------------

struct full_problem
{
    double value(const Eigen::VectorXd&) const { return 0.0; }
    void gradient(const Eigen::VectorXd&, Eigen::VectorXd&) const {}
    void hessian(const Eigen::VectorXd&, Eigen::MatrixXd&) const {}
    Eigen::VectorXd lower_bounds() const { return {}; }
    Eigen::VectorXd upper_bounds() const { return {}; }
    void constraints(const Eigen::VectorXd&, Eigen::VectorXd&) const {}
    void constraint_jacobian(const Eigen::VectorXd&, Eigen::MatrixXd&) const {}
    int num_equality() const { return 0; }
    int num_inequality() const { return 0; }
    void residuals(const Eigen::VectorXd&, Eigen::VectorXd&) const {}
    void jacobian(const Eigen::VectorXd&, Eigen::MatrixXd&) const {}
    int num_residuals() const { return 0; }
    static constexpr int problem_dimension = nablapp::dynamic_dimension;

    int dimension() const { return 1; }
};

static_assert(nablapp::objective<full_problem>);
static_assert(nablapp::differentiable<full_problem>);
static_assert(nablapp::second_order<full_problem>);
static_assert(nablapp::bound_constrained<full_problem>);
static_assert(nablapp::constrained<full_problem>);
static_assert(nablapp::least_squares<full_problem>);

// ---------------------------------------------------------------------------
// Fixed-dimension concept satisfaction
// ---------------------------------------------------------------------------

struct fixed_2d_objective
{
    static constexpr int problem_dimension = 2;
    double value(const Eigen::Vector<double, 2>&) const { return 0.0; }
    int dimension() const { return 2; }
};

static_assert(nablapp::objective<fixed_2d_objective>);
static_assert(!nablapp::differentiable<fixed_2d_objective>);

struct fixed_2d_differentiable
{
    static constexpr int problem_dimension = 2;
    double value(const Eigen::Vector<double, 2>&) const { return 0.0; }
    void gradient(const Eigen::Vector<double, 2>&, Eigen::Vector<double, 2>&) const {}
    int dimension() const { return 2; }
};

static_assert(nablapp::differentiable<fixed_2d_differentiable>);

struct fixed_2d_full
{
    static constexpr int problem_dimension = 2;
    double value(const Eigen::Vector<double, 2>&) const { return 0.0; }
    void gradient(const Eigen::Vector<double, 2>&, Eigen::Vector<double, 2>&) const {}
    void hessian(const Eigen::Vector<double, 2>&, Eigen::MatrixXd&) const {}
    Eigen::Vector<double, 2> lower_bounds() const { return {}; }
    Eigen::Vector<double, 2> upper_bounds() const { return {}; }
    void constraints(const Eigen::Vector<double, 2>&, Eigen::VectorXd&) const {}
    void constraint_jacobian(const Eigen::Vector<double, 2>&, Eigen::MatrixXd&) const {}
    int num_equality() const { return 0; }
    int num_inequality() const { return 0; }
    void residuals(const Eigen::Vector<double, 2>&, Eigen::VectorXd&) const {}
    void jacobian(const Eigen::Vector<double, 2>&, Eigen::MatrixXd&) const {}
    int num_residuals() const { return 0; }
    int dimension() const { return 2; }
};

static_assert(nablapp::objective<fixed_2d_full>);
static_assert(nablapp::differentiable<fixed_2d_full>);
static_assert(nablapp::second_order<fixed_2d_full>);
static_assert(nablapp::bound_constrained<fixed_2d_full>);
static_assert(nablapp::constrained<fixed_2d_full>);
static_assert(nablapp::least_squares<fixed_2d_full>);

// A mock solver type missing constraint_violation -- must NOT satisfy nlp_solver
struct mock_solver_no_cv
{
    using scalar_type = double;
    using state_type = int;
    nablapp::step_result<double> step() { return {}; }
    nablapp::solve_result<double> solve() { return {}; }
    nablapp::solve_result<double> step_n(int) { return {}; }
    const state_type& state() const { static state_type s{}; return s; }
    nablapp::solver_status status() const { return {}; }
    void reset(const Eigen::VectorXd&) {}
    void reset_clear(const Eigen::VectorXd&) {}
    void abort() {}
    // deliberately omitting constraint_violation()
};

static_assert(!nablapp::nlp_solver<mock_solver_no_cv>);

}

// ---------------------------------------------------------------------------
// HS test functions satisfy constrained + differentiable + bound_constrained
// ---------------------------------------------------------------------------

static_assert(nablapp::constrained<nablapp::hs071<>>);
static_assert(nablapp::differentiable<nablapp::hs071<>>);
static_assert(nablapp::bound_constrained<nablapp::hs071<>>);

static_assert(nablapp::constrained<nablapp::hs076<>>);
static_assert(nablapp::differentiable<nablapp::hs076<>>);
static_assert(nablapp::bound_constrained<nablapp::hs076<>>);

static_assert(nablapp::constrained<nablapp::hs024<>>);
static_assert(nablapp::differentiable<nablapp::hs024<>>);
static_assert(nablapp::bound_constrained<nablapp::hs024<>>);

static_assert(nablapp::constrained<nablapp::hs035<>>);
static_assert(nablapp::differentiable<nablapp::hs035<>>);
static_assert(nablapp::bound_constrained<nablapp::hs035<>>);

// ---------------------------------------------------------------------------
// nlp_solver concept satisfaction (INTG-02)
// basic_solver<Policy> satisfies nlp_solver for all solver policies
// ---------------------------------------------------------------------------

static_assert(nablapp::nlp_solver<nablapp::basic_solver<nablapp::nw_sqp_policy<>>>);
static_assert(nablapp::nlp_solver<nablapp::basic_solver<nablapp::kraft_slsqp_policy<>>>);
static_assert(nablapp::nlp_solver<nablapp::basic_solver<nablapp::lbfgsb_policy<>>>);
static_assert(nablapp::nlp_solver<nablapp::basic_solver<nablapp::bobyqa_policy<>>>);

static_assert(nablapp::nlp_solver<nablapp::basic_solver<nablapp::augmented_lagrangian_policy<>>>);
static_assert(nablapp::nlp_solver<nablapp::basic_solver<nablapp::mma_policy<>>>);
static_assert(nablapp::nlp_solver<nablapp::basic_solver<nablapp::gcmma_policy<>>>);

static_assert(nablapp::nlp_solver<nablapp::basic_solver<nablapp::cmaes_policy<>>>);
static_assert(nablapp::nlp_solver<nablapp::basic_solver<nablapp::lm_policy<>>>);

static_assert(nablapp::nlp_solver<nablapp::basic_solver<nablapp::cobyla_policy>>);
static_assert(nablapp::nlp_solver<nablapp::basic_solver<nablapp::isres_policy<>>>);

// problem_dimension concept verification
static_assert(nablapp::has_problem_dimension<nablapp::himmelblau<>>);
static_assert(nablapp::problem_dimension_v<nablapp::himmelblau<>> == 2);
static_assert(nablapp::problem_dimension_v<nablapp::rosenbrock<>> == nablapp::dynamic_dimension);
static_assert(nablapp::problem_dimension_v<nablapp::beale<>> == 2);
static_assert(nablapp::problem_dimension_v<nablapp::booth<>> == 2);
static_assert(nablapp::problem_dimension_v<nablapp::hs001<>> == 2);
static_assert(nablapp::problem_dimension_v<nablapp::powell_singular<>> == 4);
static_assert(nablapp::problem_dimension_v<nablapp::helical_valley<>> == 3);

// Negative: type without problem_dimension does NOT satisfy objective
static_assert(!nablapp::objective<empty_type>);

int main() { return 0; }
