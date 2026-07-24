"""A compiled objective against the same objective defined in the interpreter.

The bound test problems exist so this comparison is possible at all: checking an
interpreter-defined objective against a hand-written expected value tests
arithmetic, whereas checking it against the identical objective compiled into
the extension tests the callback boundary itself.

Two families run here. The delegating family hands the wrapped solver a Python
function that forwards straight back to the compiled evaluator, so the floating
point arithmetic is bit-for-bit the same on both paths and any disagreement is
the boundary and nothing else. The re-expressed family writes the same
mathematics out again in numpy, where operation ordering may legitimately
differ, and holds it to a stated tolerance.

On cost: an objective defined in the interpreter pays at least two interpreter
round trips per iteration -- one to enter Python, one to convert and return --
against native iterations measured in fractions of a microsecond. That
slowdown is structural. It is a property of putting an interpreter inside the
loop, not a defect in this binding and not something this phase set out to fix.
This module measures the ratio and prints it. It asserts nothing about it, and
no tolerance or iteration budget anywhere in this suite has been chosen to make
the interpreter path finish sooner.
"""

import time

import numpy as np
import pytest

import argmin

# The objective tolerance is deliberately the tighter of the two: a valley
# floor this flat admits many nearly-optimal points, so agreeing on the value
# is the stronger statement and agreeing on the point is the weaker one.
OBJECTIVE_TOLERANCE = 1e-12
POINT_TOLERANCE = 1e-6

# Distance to the published optimum is a different claim from parity and is
# governed by the driver's default gradient threshold, which stops a solve while
# the gradient still measures 1e-5. On this valley that admits a point offset of
# the same order, so the parity tolerance above would be the wrong constant here.
OPTIMUM_POINT_TOLERANCE = 1e-4

BUDGET = 500
BOX = (np.full(2, -5.0), np.full(2, 5.0))

TIMING_REPEATS = 20


def rosenbrock_problem():
    lower, upper = BOX
    return argmin.Rosenbrock(2, lower_bounds=lower, upper_bounds=upper)


def reexpressed_rosenbrock(x):
    return float((1.0 - x[0]) ** 2 + 5.0 * (x[1] - x[0] ** 2) ** 2)


def reexpressed_rosenbrock_gradient(x):
    return np.array(
        [
            -2.0 * (1.0 - x[0]) - 20.0 * x[0] * (x[1] - x[0] ** 2),
            10.0 * (x[1] - x[0] ** 2),
        ]
    )


def reexpressed_constrained(x):
    return float((1.0 - x[0]) ** 2)


def reexpressed_constrained_gradient(x):
    return np.array([-2.0 * (1.0 - x[0]), 0.0])


def reexpressed_constraints(x):
    return np.array([10.0 * (x[1] - x[0] ** 2)])


def reexpressed_constraint_jacobian(x):
    return np.array([[-20.0 * x[0], 10.0]])


def wrapped_lbfgsb(problem, objective, gradient):
    lower, upper = BOX
    return argmin.LbfgsbSolver(
        objective,
        problem.initial_point(),
        gradient=gradient,
        lower_bounds=lower,
        upper_bounds=upper,
        max_iterations=BUDGET,
    )


def wrapped_bobyqa(problem, objective, gradient):
    lower, upper = BOX
    return argmin.BobyqaSolver(
        objective,
        problem.initial_point(),
        lower_bounds=lower,
        upper_bounds=upper,
        max_iterations=BUDGET,
    )


def wrapped_slsqp(problem, objective, gradient, constraints, jacobian):
    return argmin.SlsqpSolver(
        objective,
        problem.initial_point(),
        gradient=gradient,
        constraints=constraints,
        constraint_jacobian=jacobian,
        num_equality=problem.num_equality(),
        num_inequality=problem.num_inequality(),
        max_iterations=BUDGET,
    )


def wrapped_cobyla(problem, objective, gradient, constraints, jacobian):
    return argmin.CobylaSolver(
        objective,
        problem.initial_point(),
        constraints=constraints,
        num_equality=problem.num_equality(),
        num_inequality=problem.num_inequality(),
        max_iterations=BUDGET,
    )


UNCONSTRAINED_METHODS = {"lbfgsb": wrapped_lbfgsb, "bobyqa": wrapped_bobyqa}
CONSTRAINED_METHODS = {"slsqp": wrapped_slsqp, "cobyla": wrapped_cobyla}


def assert_agree(native, interpreted):
    assert abs(native.objective_value - interpreted.objective_value) <= (
        OBJECTIVE_TOLERANCE
    )
    assert np.linalg.norm(native.x - interpreted.x) <= POINT_TOLERANCE


@pytest.mark.parametrize("method", sorted(UNCONSTRAINED_METHODS))
def test_a_delegating_objective_matches_the_compiled_one(method):
    problem = rosenbrock_problem()
    native = problem.solve_native(method, max_iterations=BUDGET)
    interpreted = UNCONSTRAINED_METHODS[method](
        problem, problem.value, problem.gradient
    ).solve()

    assert native.status == interpreted.status
    assert native.iterations == interpreted.iterations
    # Identical arithmetic on both paths, so anything but exact equality here is
    # the boundary corrupting a value rather than a floating point reordering.
    assert native.objective_value == interpreted.objective_value
    assert np.array_equal(native.x, interpreted.x)


@pytest.mark.parametrize("method", sorted(UNCONSTRAINED_METHODS))
def test_a_reexpressed_objective_matches_the_compiled_one(method):
    problem = rosenbrock_problem()
    native = problem.solve_native(method, max_iterations=BUDGET)
    interpreted = UNCONSTRAINED_METHODS[method](
        problem, reexpressed_rosenbrock, reexpressed_rosenbrock_gradient
    ).solve()

    assert_agree(native, interpreted)


@pytest.mark.parametrize("method", sorted(CONSTRAINED_METHODS))
def test_a_delegating_constrained_objective_matches_the_compiled_one(method):
    problem = argmin.ConstrainedTestProblem()
    native = problem.solve_native(method, max_iterations=BUDGET)
    interpreted = CONSTRAINED_METHODS[method](
        problem,
        problem.value,
        problem.gradient,
        problem.constraints,
        problem.constraint_jacobian,
    ).solve()

    assert native.status == interpreted.status
    assert native.iterations == interpreted.iterations
    assert native.objective_value == interpreted.objective_value
    assert np.array_equal(native.x, interpreted.x)


@pytest.mark.parametrize("method", sorted(CONSTRAINED_METHODS))
def test_a_reexpressed_constrained_objective_matches_the_compiled_one(method):
    problem = argmin.ConstrainedTestProblem()
    native = problem.solve_native(method, max_iterations=BUDGET)
    interpreted = CONSTRAINED_METHODS[method](
        problem,
        reexpressed_constrained,
        reexpressed_constrained_gradient,
        reexpressed_constraints,
        reexpressed_constraint_jacobian,
    ).solve()

    assert_agree(native, interpreted)
    assert abs(native.objective_value - problem.optimal_value()) <= OBJECTIVE_TOLERANCE


def test_both_paths_reach_the_published_optimum():
    problem = rosenbrock_problem()
    native = problem.solve_native("lbfgsb", max_iterations=BUDGET)
    interpreted = wrapped_lbfgsb(
        problem, reexpressed_rosenbrock, reexpressed_rosenbrock_gradient
    ).solve()

    assert abs(native.objective_value - problem.optimal_value()) <= OBJECTIVE_TOLERANCE
    assert np.linalg.norm(interpreted.x - np.ones(2)) <= OPTIMUM_POINT_TOLERANCE


def _average_seconds(run):
    run()
    started = time.perf_counter()
    for _ in range(TIMING_REPEATS):
        run()
    return (time.perf_counter() - started) / TIMING_REPEATS


@pytest.mark.parametrize("method", sorted(UNCONSTRAINED_METHODS))
def test_the_cost_of_an_interpreter_defined_objective_is_reported(method, capsys):
    problem = rosenbrock_problem()

    native_seconds = _average_seconds(
        lambda: problem.solve_native(method, max_iterations=BUDGET)
    )
    interpreter_seconds = _average_seconds(
        lambda: UNCONSTRAINED_METHODS[method](
            problem, reexpressed_rosenbrock, reexpressed_rosenbrock_gradient
        ).solve()
    )

    with capsys.disabled():
        print(
            f"\n{method}: compiled {native_seconds * 1e6:.1f} us per solve, "
            f"interpreter-defined {interpreter_seconds * 1e6:.1f} us per solve, "
            f"cost factor {interpreter_seconds / native_seconds:.1f}x "
            f"(structural: two interpreter round trips per iteration)"
        )

    assert native_seconds > 0.0
    assert interpreter_seconds > 0.0
