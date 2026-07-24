"""Step-level control: the driver documents solve and step-with-budget as
continuations rather than restarts, and these assert that documented behavior."""

import numpy as np
import pytest

import argmin

TERMINAL_STATUSES = frozenset(
    status
    for status in argmin.SolverStatus.__members__.values()
    if status != argmin.SolverStatus.running
)

PROBLEM = argmin.Rosenbrock(2)
START = PROBLEM.initial_point()
ELSEWHERE = np.array([0.5, -0.25])
BOX = (np.full(2, -5.0), np.full(2, 5.0))


def objective(x):
    return PROBLEM.value(x)


def gradient(x):
    return PROBLEM.gradient(x)


def inactive_constraint(x):
    return np.array([10.0])


def inactive_jacobian(x):
    return np.zeros((1, 2))


def make_lbfgsb():
    return argmin.LbfgsbSolver(
        objective, START, gradient=gradient, max_iterations=500
    )


def make_bobyqa():
    lower, upper = BOX
    return argmin.BobyqaSolver(
        objective, START, lower_bounds=lower, upper_bounds=upper, max_iterations=2000
    )


def make_slsqp():
    return argmin.SlsqpSolver(
        objective,
        START,
        gradient=gradient,
        constraints=inactive_constraint,
        constraint_jacobian=inactive_jacobian,
        num_equality=0,
        num_inequality=1,
        max_iterations=500,
    )


FACTORIES = {"lbfgsb": make_lbfgsb, "bobyqa": make_bobyqa, "slsqp": make_slsqp}

# BOBYQA rebuilds its whole interpolation model on either reset entry point, so
# the two are behaviorally indistinguishable for it; only the policies that
# carry curvature across a reset can show the difference.
STATE_CARRYING = ("lbfgsb", "slsqp")

ALL_METHODS = pytest.mark.parametrize("method", sorted(FACTORIES))


@ALL_METHODS
def test_a_single_step_advances_exactly_one_iteration(method):
    solver = FACTORIES[method]()
    solver.step()
    assert solver.step_n(1).iterations == 2


@ALL_METHODS
def test_step_with_budget_continues_rather_than_restarting(method):
    solver = FACTORIES[method]()
    first = solver.step_n(3)
    second = solver.step_n(4)
    assert first.iterations == 3
    assert second.iterations == 7


@ALL_METHODS
def test_a_solve_after_partial_stepping_continues_from_there(method):
    stepped = FACTORIES[method]()
    stepped.step_n(5)
    continued = stepped.solve()

    fresh = FACTORIES[method]().solve()

    assert continued.iterations >= 5
    assert continued.iterations == fresh.iterations
    assert continued.objective_value == pytest.approx(fresh.objective_value, abs=1e-12)


@pytest.mark.parametrize("method", STATE_CARRYING)
def test_a_reset_returns_to_the_given_starting_point(method):
    solver = FACTORIES[method]()
    solver.step_n(6)
    solver.reset(ELSEWHERE)
    assert np.array_equal(solver.x, ELSEWHERE)

    solver.reset_clear(START)
    assert np.array_equal(solver.x, START)


# The interpolation method rebuilds its whole sample set inside the reset call
# and adopts the best of those points as the incumbent, so its iterate after a
# reset sits within the initial trust radius of the requested point rather than
# on it. Construction defers that bootstrap to the first step, which is why a
# freshly built solver does report the point it was given.
def test_a_reset_of_the_interpolation_method_resamples_around_the_given_point():
    solver = make_bobyqa()
    assert np.array_equal(solver.x, START)

    solver.step_n(6)
    solver.reset(ELSEWHERE)
    assert not np.array_equal(solver.x, ELSEWHERE)
    assert np.linalg.norm(solver.x - ELSEWHERE) <= 1.0


@ALL_METHODS
def test_a_reset_restarts_the_iteration_count(method):
    solver = FACTORIES[method]()
    solver.step_n(6)
    solver.reset(START)
    assert solver.step_n(2).iterations == 2


# The difference is asserted by what the next step does, not by reading any
# internal: a reset that kept the accumulated curvature takes a different first
# step than one that threw it away.
@pytest.mark.parametrize("method", STATE_CARRYING)
def test_the_two_resets_differ_by_observed_behavior(method):
    preserving = FACTORIES[method]()
    preserving.step_n(15)
    preserving.reset(START)
    preserving.step()

    clearing = FACTORIES[method]()
    clearing.step_n(15)
    clearing.reset_clear(START)
    clearing.step()

    assert not np.allclose(preserving.x, clearing.x)


@pytest.mark.parametrize("method", STATE_CARRYING)
def test_a_state_clearing_reset_reproduces_a_fresh_solver(method):
    reused = FACTORIES[method]()
    reused.step_n(15)
    reused.reset_clear(START)
    reused.step()

    fresh = FACTORIES[method]()
    fresh.step()

    assert np.array_equal(reused.x, fresh.x)


@ALL_METHODS
def test_the_status_is_running_until_the_driver_terminates(method):
    solver = FACTORIES[method]()
    assert solver.status() == argmin.SolverStatus.running

    solver.step()
    assert solver.status() == argmin.SolverStatus.running

    solver.solve()
    assert solver.status() in TERMINAL_STATUSES


@ALL_METHODS
def test_the_iterate_read_between_steps_improves_monotonically(method):
    solver = FACTORIES[method]()
    values = [objective(solver.x)]
    for _ in range(12):
        solver.step()
        values.append(objective(solver.x))

    assert all(later <= earlier for earlier, later in zip(values, values[1:]))
    assert values[-1] < values[0]


@ALL_METHODS
def test_a_step_budget_that_is_not_positive_is_rejected(method):
    solver = FACTORIES[method]()
    with pytest.raises(argmin.ArgminError) as raised:
        solver.step_n(0)
    assert raised.value.kind == argmin.ErrorKind.dimension_mismatch
