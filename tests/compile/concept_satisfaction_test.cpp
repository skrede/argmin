// Concept satisfaction compile-time tests.
//
// These verify that the concept hierarchy is correctly defined and that
// test function types satisfy the expected concepts.
//
// If this file compiles, all assertions pass. If any assertion fails,
// the build fails with a clear concept-violation error.

#include "nablapp/formulation/concepts.h"
#include "nablapp/test_functions/booth.h"
#include "nablapp/test_functions/beale.h"
#include "nablapp/test_functions/ackley.h"
#include "nablapp/test_functions/rastrigin.h"
#include "nablapp/test_functions/rosenbrock.h"
#include "nablapp/test_functions/himmelblau.h"

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
    int dimension() const { return 1; }
};

static_assert(nablapp::objective<full_problem>);
static_assert(nablapp::differentiable<full_problem>);
static_assert(nablapp::second_order<full_problem>);
static_assert(nablapp::bound_constrained<full_problem>);
static_assert(nablapp::constrained<full_problem>);
static_assert(nablapp::least_squares<full_problem>);

}

int main() { return 0; }
