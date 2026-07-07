// Concept satisfaction compile-time tests.
//
// These verify that the concept hierarchy is correctly defined and that
// test function types satisfy the expected concepts.
//
// If this file compiles, all assertions pass. If any assertion fails,
// the build fails with a clear concept-violation error.

#include "argmin/formulation/concepts.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/options.h"
#include "argmin/schedule/round_robin_schedule.h"
#include "argmin/schedule/fallback_schedule.h"
#include "argmin/schedule/time_boxed_schedule.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/filter_nw_sqp_policy.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/byrd_lbfgsb_policy.h"
#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/cmaes_policy.h"
#include "argmin/solver/lm_policy.h"
#include "argmin/solver/ccsa_quadratic_policy.h"
#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/isres_policy.h"
#include "argmin/solver/multistart_policy.h"
#include "argmin/solver/restarting_policy.h"
#include "argmin/solver/projected_gn_policy.h"
#include "argmin/solver/projected_gradient_gn_policy.h"
#include "argmin/solver/augmented_lagrangian_policy.h"
#include "argmin/test_functions/booth.h"
#include "argmin/test_functions/beale.h"
#include "argmin/test_functions/ackley.h"
#include "argmin/test_functions/schwefel.h"
#include "argmin/test_functions/griewank.h"
#include "argmin/test_functions/rastrigin.h"
#include "argmin/test_functions/rosenbrock.h"
#include "argmin/test_functions/himmelblau.h"
#include "argmin/test_functions/hock_schittkowski.h"
#include "argmin/test_functions/more_garbow_hillstrom.h"

// ---------------------------------------------------------------------------
// Positive tests: all six test functions satisfy objective and differentiable
// ---------------------------------------------------------------------------

static_assert(argmin::objective<argmin::rosenbrock<>>);
static_assert(argmin::differentiable<argmin::rosenbrock<>>);

static_assert(argmin::objective<argmin::booth<>>);
static_assert(argmin::differentiable<argmin::booth<>>);

static_assert(argmin::objective<argmin::beale<>>);
static_assert(argmin::differentiable<argmin::beale<>>);

static_assert(argmin::objective<argmin::himmelblau<>>);
static_assert(argmin::differentiable<argmin::himmelblau<>>);

static_assert(argmin::objective<argmin::rastrigin<>>);
static_assert(argmin::differentiable<argmin::rastrigin<>>);

static_assert(argmin::objective<argmin::ackley<>>);
static_assert(argmin::differentiable<argmin::ackley<>>);

// ---------------------------------------------------------------------------
// Float scalar support (D-07)
// ---------------------------------------------------------------------------

static_assert(argmin::objective<argmin::rosenbrock<float>, float>);
static_assert(argmin::differentiable<argmin::rosenbrock<float>, float>);

static_assert(argmin::objective<argmin::booth<float>, float>);
static_assert(argmin::differentiable<argmin::booth<float>, float>);

static_assert(argmin::objective<argmin::beale<float>, float>);
static_assert(argmin::differentiable<argmin::beale<float>, float>);

static_assert(argmin::objective<argmin::himmelblau<float>, float>);
static_assert(argmin::differentiable<argmin::himmelblau<float>, float>);

static_assert(argmin::objective<argmin::rastrigin<float>, float>);
static_assert(argmin::differentiable<argmin::rastrigin<float>, float>);

static_assert(argmin::objective<argmin::ackley<float>, float>);
static_assert(argmin::differentiable<argmin::ackley<float>, float>);

// ---------------------------------------------------------------------------
// Test functions should NOT satisfy higher-order concepts
// ---------------------------------------------------------------------------

static_assert(!argmin::second_order<argmin::rosenbrock<>>);
static_assert(!argmin::bound_constrained<argmin::rosenbrock<>>);
static_assert(!argmin::constrained<argmin::rosenbrock<>>);
static_assert(!argmin::least_squares<argmin::rosenbrock<>>);

// ---------------------------------------------------------------------------
// Negative tests: mock types that intentionally lack required methods
// ---------------------------------------------------------------------------

namespace
{

// A type with only value() -- satisfies objective but NOT differentiable
struct objective_only
{
    double value(const Eigen::VectorXd&) const { return 0.0; }
    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int dimension() const { return 1; }
};

static_assert(argmin::objective<objective_only>);
static_assert(!argmin::differentiable<objective_only>);
static_assert(!argmin::second_order<objective_only>);
static_assert(!argmin::bound_constrained<objective_only>);
static_assert(!argmin::constrained<objective_only>);
static_assert(!argmin::least_squares<objective_only>);

// A type with value() + gradient() -- differentiable but NOT second_order
struct differentiable_only
{
    double value(const Eigen::VectorXd&) const { return 0.0; }
    void gradient(const Eigen::VectorXd&, Eigen::VectorXd&) const {}
    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int dimension() const { return 1; }
};

static_assert(argmin::differentiable<differentiable_only>);
static_assert(!argmin::second_order<differentiable_only>);

// An empty struct -- satisfies nothing
struct empty_type {};

static_assert(!argmin::objective<empty_type>);
static_assert(!argmin::differentiable<empty_type>);
static_assert(!argmin::second_order<empty_type>);
static_assert(!argmin::bound_constrained<empty_type>);
static_assert(!argmin::constrained<empty_type>);
static_assert(!argmin::least_squares<empty_type>);

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
    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int dimension() const { return 1; }
};

static_assert(argmin::objective<full_problem>);
static_assert(argmin::differentiable<full_problem>);
static_assert(argmin::second_order<full_problem>);
static_assert(argmin::bound_constrained<full_problem>);
static_assert(argmin::constrained<full_problem>);
static_assert(argmin::least_squares<full_problem>);

// ---------------------------------------------------------------------------
// Fixed-dimension concept satisfaction
// ---------------------------------------------------------------------------

struct fixed_2d_objective
{
    static constexpr int problem_dimension = 2;
    double value(const Eigen::Vector<double, 2>&) const { return 0.0; }
    int dimension() const { return 2; }
};

static_assert(argmin::objective<fixed_2d_objective>);
static_assert(!argmin::differentiable<fixed_2d_objective>);

struct fixed_2d_differentiable
{
    static constexpr int problem_dimension = 2;
    double value(const Eigen::Vector<double, 2>&) const { return 0.0; }
    void gradient(const Eigen::Vector<double, 2>&, Eigen::Vector<double, 2>&) const {}
    int dimension() const { return 2; }
};

static_assert(argmin::differentiable<fixed_2d_differentiable>);

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

static_assert(argmin::objective<fixed_2d_full>);
static_assert(argmin::differentiable<fixed_2d_full>);
static_assert(argmin::second_order<fixed_2d_full>);
static_assert(argmin::bound_constrained<fixed_2d_full>);
static_assert(argmin::constrained<fixed_2d_full>);
static_assert(argmin::least_squares<fixed_2d_full>);

// A mock solver type missing constraint_violation -- must NOT satisfy nlp_solver
struct mock_solver_no_cv
{
    using scalar_type = double;
    using state_type = int;
    argmin::step_result<double> step() { return {}; }
    argmin::solve_result<double> solve() { return {}; }
    argmin::solve_result<double> step_n(int) { return {}; }
    const state_type& state() const { static state_type s{}; return s; }
    argmin::solver_status status() const { return {}; }
    void reset(const Eigen::VectorXd&) {}
    void reset_clear(const Eigen::VectorXd&) {}
    void abort() {}
    // deliberately omitting constraint_violation()
};

static_assert(!argmin::nlp_solver<mock_solver_no_cv>);

// The same mock also fails the core steppable contract: it omits
// constraint_violation(), which steppable requires.
static_assert(!argmin::steppable<mock_solver_no_cv>);

// ---------------------------------------------------------------------------
// Harness contract: solver_policy / solver_state / schedule concepts.
//
// These lock the enforcement introduced at the harness (step_budget_solver static-
// asserts solver_policy; basic_solver_group static-asserts schedule). The
// negative cases prove the concepts are non-vacuous: a policy missing
// reset_clear, or a state without x, or a schedule missing reset, is rejected.
// ---------------------------------------------------------------------------

// A well-formed minimal state and policy that satisfy the contract.
struct good_state
{
    Eigen::VectorXd x;
};

struct good_policy
{
    using scalar_type = double;
    argmin::step_result<double> step(good_state&) { return {}; }
    void reset(good_state&, const Eigen::VectorXd&) {}
    void reset_clear(good_state&, const Eigen::VectorXd&) {}
};

static_assert(argmin::solver_state<good_state>);
static_assert(argmin::solver_policy<good_policy, good_state>);

// A state without x is not a solver_state (and cannot back a solver_policy).
struct state_no_x
{
    double objective_value{};
};

static_assert(!argmin::solver_state<state_no_x>);
static_assert(!argmin::solver_policy<good_policy, state_no_x>);

// A policy missing reset_clear is rejected by solver_policy.
struct policy_no_reset_clear
{
    using scalar_type = double;
    argmin::step_result<double> step(good_state&) { return {}; }
    void reset(good_state&, const Eigen::VectorXd&) {}
    // deliberately omitting reset_clear()
};

static_assert(!argmin::solver_policy<policy_no_reset_clear, good_state>);

// A policy missing scalar_type is rejected by solver_policy.
struct policy_no_scalar_type
{
    argmin::step_result<double> step(good_state&) { return {}; }
    void reset(good_state&, const Eigen::VectorXd&) {}
    void reset_clear(good_state&, const Eigen::VectorXd&) {}
};

static_assert(!argmin::solver_policy<policy_no_scalar_type, good_state>);

// Opt-in constrained-state refinement.
struct constrained_state
{
    Eigen::VectorXd x;
    Eigen::VectorXd c_eq;
    Eigen::VectorXd c_ineq;
};

static_assert(argmin::constrained_policy_state<constrained_state>);
static_assert(!argmin::constrained_policy_state<good_state>);

// A well-formed minimal schedule satisfies the schedule contract.
struct good_schedule
{
    std::size_t select(std::size_t) { return 0; }
    void reset() {}
    template <typename Scalar>
    void notify(const argmin::step_result<Scalar>&) {}
};

static_assert(argmin::schedule<good_schedule>);

// A schedule missing reset() is rejected by the schedule concept.
struct schedule_no_reset
{
    std::size_t select(std::size_t) { return 0; }
    template <typename Scalar>
    void notify(const argmin::step_result<Scalar>&) {}
    // deliberately omitting reset()
};

static_assert(!argmin::schedule<schedule_no_reset>);

// A schedule missing notify() is rejected by the schedule concept.
struct schedule_no_notify
{
    std::size_t select(std::size_t) { return 0; }
    void reset() {}
    // deliberately omitting notify()
};

static_assert(!argmin::schedule<schedule_no_notify>);

}

// The three shipped schedules satisfy the schedule contract.
static_assert(argmin::schedule<argmin::round_robin_schedule>);
static_assert(argmin::schedule<argmin::fallback_schedule>);
static_assert(argmin::schedule<argmin::time_boxed_schedule>);

// ---------------------------------------------------------------------------
// HS test functions satisfy constrained + differentiable + bound_constrained
// ---------------------------------------------------------------------------

static_assert(argmin::constrained<argmin::hs071<>>);
static_assert(argmin::differentiable<argmin::hs071<>>);
static_assert(argmin::bound_constrained<argmin::hs071<>>);

static_assert(argmin::constrained<argmin::hs076<>>);
static_assert(argmin::differentiable<argmin::hs076<>>);
static_assert(argmin::bound_constrained<argmin::hs076<>>);

static_assert(argmin::constrained<argmin::hs024<>>);
static_assert(argmin::differentiable<argmin::hs024<>>);
static_assert(argmin::bound_constrained<argmin::hs024<>>);

static_assert(argmin::constrained<argmin::hs035<>>);
static_assert(argmin::differentiable<argmin::hs035<>>);
static_assert(argmin::bound_constrained<argmin::hs035<>>);

// ---------------------------------------------------------------------------
// nlp_solver concept satisfaction (INTG-02)
// step_budget_solver<Policy> satisfies nlp_solver for all solver policies
// ---------------------------------------------------------------------------

static_assert(argmin::nlp_solver<argmin::step_budget_solver<argmin::nw_sqp_policy<>>>);
static_assert(argmin::nlp_solver<argmin::step_budget_solver<argmin::filter_nw_sqp_policy<>>>);
static_assert(argmin::nlp_solver<argmin::step_budget_solver<argmin::kraft_slsqp_policy<>>>);
static_assert(argmin::nlp_solver<argmin::step_budget_solver<argmin::filter_slsqp_policy<>>>);
static_assert(argmin::nlp_solver<argmin::step_budget_solver<argmin::lbfgsb_policy<>>>);
static_assert(argmin::nlp_solver<argmin::step_budget_solver<argmin::byrd_lbfgsb_policy<>>>);
static_assert(argmin::nlp_solver<argmin::step_budget_solver<argmin::bobyqa_policy<>>>);

static_assert(argmin::nlp_solver<argmin::step_budget_solver<argmin::augmented_lagrangian_policy<>>>);
static_assert(argmin::nlp_solver<argmin::step_budget_solver<argmin::ccsa_quadratic_policy<>>>);

static_assert(argmin::nlp_solver<argmin::step_budget_solver<argmin::cmaes_policy<>>>);
static_assert(argmin::nlp_solver<argmin::step_budget_solver<argmin::lm_policy<>>>);

static_assert(argmin::nlp_solver<argmin::step_budget_solver<argmin::cobyla_policy>>);
static_assert(argmin::nlp_solver<argmin::step_budget_solver<argmin::isres_policy<>>>);

static_assert(argmin::nlp_solver<argmin::step_budget_solver<argmin::projected_gn_policy>>);
static_assert(argmin::nlp_solver<argmin::step_budget_solver<argmin::projected_gradient_gn_policy>>);

static_assert(argmin::nlp_solver<argmin::step_budget_solver<argmin::multistart_policy<argmin::bobyqa_policy<>>>>);
static_assert(argmin::nlp_solver<argmin::step_budget_solver<argmin::restarting_policy<argmin::cmaes_policy<>>>>);

// ---------------------------------------------------------------------------
// steppable core concept satisfaction
//
// step_budget_solver<Policy> exposes the full loop-owning surface, so it satisfies
// both the core steppable contract and its nlp_solver refinement. These pin
// that the split is additive: every nlp_solver is a steppable, across a
// representative constrained policy, an unconstrained policy, and a
// derivative-free policy.
// ---------------------------------------------------------------------------

static_assert(argmin::steppable<argmin::step_budget_solver<argmin::nw_sqp_policy<>>>);
static_assert(argmin::steppable<argmin::step_budget_solver<argmin::lbfgsb_policy<>>>);
static_assert(argmin::steppable<argmin::step_budget_solver<argmin::cmaes_policy<>>>);

// nlp_solver refines steppable: satisfying the loop-owning concept implies
// the core stepping surface holds too.
static_assert(argmin::steppable<argmin::step_budget_solver<argmin::nw_sqp_policy<>>>
              && argmin::nlp_solver<argmin::step_budget_solver<argmin::nw_sqp_policy<>>>);

// problem_dimension concept verification
static_assert(argmin::has_problem_dimension<argmin::himmelblau<>>);
static_assert(argmin::problem_dimension_v<argmin::himmelblau<>> == 2);
static_assert(argmin::problem_dimension_v<argmin::rosenbrock<>> == argmin::dynamic_dimension);
static_assert(argmin::problem_dimension_v<argmin::beale<>> == 2);
static_assert(argmin::problem_dimension_v<argmin::booth<>> == 2);
static_assert(argmin::problem_dimension_v<argmin::hs001<>> == 2);
static_assert(argmin::problem_dimension_v<argmin::powell_singular<>> == 4);
static_assert(argmin::problem_dimension_v<argmin::helical_valley<>> == 3);

// Negative: type without problem_dimension does NOT satisfy objective
static_assert(!argmin::objective<empty_type>);

int main() { return 0; }
