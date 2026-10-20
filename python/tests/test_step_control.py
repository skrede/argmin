"""Step-level control: the driver documents solve and step-with-budget as
continuations rather than restarts, and these assert that documented behavior."""

import numpy as np
import pytest

import argmin
import argmin.experimental as experimental

from step_factories import (
    BOX,
    ELSEWHERE,
    FACTORIES,
    START,
    STATE_CARRYING,
    make_bobyqa,
    objective,
)

TERMINAL_STATUSES = frozenset(
    status
    for status in argmin.SolverStatus.__members__.values()
    if status != argmin.SolverStatus.running
)

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
def test_a_reset_of_the_interpolation_method_restarts_as_a_fresh_solver_at_the_given_point():
    lower, upper = BOX
    fresh = experimental.BobyqaSolver(
        objective, ELSEWHERE, lower_bounds=lower, upper_bounds=upper, max_iterations=2000
    )
    solver = make_bobyqa()
    assert np.array_equal(solver.x, START)

    solver.step_n(6)
    solver.reset(ELSEWHERE)
    assert np.array_equal(solver.x, fresh.x)
    solver.step_n(20)
    fresh.step_n(20)
    assert np.array_equal(solver.x, fresh.x)


@ALL_METHODS
def test_a_reset_restarts_the_iteration_count(method):
    solver = FACTORIES[method]()
    solver.step_n(6)
    solver.reset(START)
    assert solver.step_n(2).iterations == 2


def post_reset_path(method, entry, steps):
    solver = FACTORIES[method]()
    solver.step_n(15)
    getattr(solver, entry)(START)
    path = []
    for _ in range(steps):
        solver.step()
        path.append(solver.x.copy())
    return np.array(path)


# The difference is asserted by what the next steps do, not by reading any
# internal: a reset that kept the accumulated curvature takes a different path
# than one that threw it away. A trust-region step can be rejected outright or
# cut to the boundary along steepest descent, neither of which reads the
# curvature, so the path is compared over three steps rather than one.
@pytest.mark.parametrize("method", STATE_CARRYING)
def test_the_two_resets_differ_by_observed_behavior(method):
    preserving = post_reset_path(method, "reset", 3)
    clearing = post_reset_path(method, "reset_clear", 3)
    assert not np.allclose(preserving, clearing)


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
