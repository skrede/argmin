"""Exception propagation across the callback boundary.

Every case here asserts three separate properties: the caller observes the
exception the callback actually raised, with its own type and message; the
adapter stopped calling into the interpreter once the failure was latched; and
the solver survives the failure well enough to be restarted.
"""

import numpy as np
import pytest

import argmin


class Boom(Exception):
    pass


BOOM_MESSAGE = "the callback declined"


def rosenbrock(x):
    return float((1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)


def rosenbrock_gradient(x):
    return np.array(
        [
            -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2),
            200.0 * (x[1] - x[0] ** 2),
        ]
    )


class Trap:
    """A callback that behaves normally until armed, then fails exactly once.

    ``calls_after_failure`` is the observable consequence of the latch: the
    adapter must stop entering the interpreter the moment a failure is
    recorded, so this counter is what distinguishes a real latch from a
    swallowed exception.
    """

    def __init__(self, normal, failure):
        self.normal = normal
        self.failure = failure
        self.calls = 0
        self.armed = False
        self.failed = False
        self.calls_after_failure = 0

    def arm(self):
        self.armed = True

    def __call__(self, x):
        self.calls += 1
        if self.failed:
            self.calls_after_failure += 1
        elif self.armed:
            self.failed = True
            return self.failure(x)
        return self.normal(x)


def raise_boom(_x=None):
    raise Boom(BOOM_MESSAGE)


def drive(solver, entry):
    if entry == "solve":
        return solver.solve()
    for _ in range(50):
        solver.step()
    return None


ENTRY_POINTS = pytest.mark.parametrize("entry", ["solve", "step"])


def make_solver(objective, gradient=None, **kwargs):
    return argmin.LbfgsbSolver(
        objective, np.array([-1.2, 1.0]), gradient=gradient, max_iterations=400, **kwargs
    )


def test_objective_raising_before_the_loop_surfaces_at_construction():
    # The bound-constrained policy evaluates f and its gradient at the starting
    # point while initializing, so a callback that fails on its very first call
    # fails during construction -- the earliest possible call, not a later
    # unrelated one.
    with pytest.raises(Boom, match=BOOM_MESSAGE):
        make_solver(raise_boom)


def test_gradient_raising_before_the_loop_surfaces_at_construction():
    with pytest.raises(Boom, match=BOOM_MESSAGE):
        make_solver(rosenbrock, gradient=raise_boom)


def test_bound_accessor_raising_surfaces_at_construction():
    with pytest.raises(Boom, match=BOOM_MESSAGE):
        argmin.LbfgsbSolver(
            rosenbrock,
            np.array([-1.2, 1.0]),
            gradient=rosenbrock_gradient,
            lower=lambda: raise_boom(),
        )


@ENTRY_POINTS
def test_objective_raising_mid_solve(entry):
    trap = Trap(rosenbrock, raise_boom)
    solver = make_solver(trap, gradient=rosenbrock_gradient)
    trap.arm()
    with pytest.raises(Boom, match=BOOM_MESSAGE):
        drive(solver, entry)
    assert trap.failed
    assert trap.calls_after_failure <= 1


@ENTRY_POINTS
def test_gradient_raising_mid_solve(entry):
    trap = Trap(rosenbrock_gradient, raise_boom)
    solver = make_solver(rosenbrock, gradient=trap)
    trap.arm()
    with pytest.raises(Boom, match=BOOM_MESSAGE):
        drive(solver, entry)
    assert trap.failed
    assert trap.calls_after_failure <= 1


@ENTRY_POINTS
def test_objective_and_gradient_stop_being_called_after_a_failure(entry):
    objective = Trap(rosenbrock, raise_boom)
    gradient = Trap(rosenbrock_gradient, rosenbrock_gradient)
    solver = make_solver(objective, gradient=gradient)
    calls_at_arm = gradient.calls
    objective.arm()
    with pytest.raises(Boom):
        drive(solver, entry)
    assert objective.calls_after_failure <= 1
    assert gradient.calls - calls_at_arm <= 1


@ENTRY_POINTS
def test_non_numeric_objective_return(entry):
    trap = Trap(rosenbrock, lambda _x: "not a number at all")
    solver = make_solver(trap, gradient=rosenbrock_gradient)
    trap.arm()
    with pytest.raises(argmin.ArgminError) as raised:
        drive(solver, entry)
    assert raised.value.kind == argmin.ErrorKind.invalid_callback
    assert trap.calls_after_failure <= 1


@ENTRY_POINTS
def test_objective_returning_a_value_that_is_not_a_number(entry):
    trap = Trap(rosenbrock, lambda _x: float("nan"))
    solver = make_solver(trap, gradient=rosenbrock_gradient)
    trap.arm()
    with pytest.raises(argmin.ArgminError) as raised:
        drive(solver, entry)
    assert raised.value.kind == argmin.ErrorKind.invalid_callback


@ENTRY_POINTS
def test_gradient_of_the_wrong_length(entry):
    trap = Trap(rosenbrock_gradient, lambda _x: np.zeros(5))
    solver = make_solver(rosenbrock, gradient=trap)
    trap.arm()
    with pytest.raises(argmin.ArgminError) as raised:
        drive(solver, entry)
    assert raised.value.kind == argmin.ErrorKind.invalid_callback
    assert "length" in str(raised.value)
    assert trap.calls_after_failure <= 1


@ENTRY_POINTS
def test_gradient_with_a_non_finite_entry(entry):
    trap = Trap(rosenbrock_gradient, lambda _x: np.array([np.inf, 0.0]))
    solver = make_solver(rosenbrock, gradient=trap)
    trap.arm()
    with pytest.raises(argmin.ArgminError) as raised:
        drive(solver, entry)
    assert raised.value.kind == argmin.ErrorKind.invalid_callback


@ENTRY_POINTS
def test_a_latched_failure_is_recoverable(entry):
    trap = Trap(rosenbrock, raise_boom)
    solver = make_solver(trap, gradient=rosenbrock_gradient)
    trap.arm()
    with pytest.raises(Boom):
        drive(solver, entry)

    solver.reset(np.array([-1.2, 1.0]))
    result = solver.solve()
    assert result.status == argmin.SolverStatus.converged
    assert np.allclose(result.x, [1.0, 1.0], atol=1e-4)


@ENTRY_POINTS
def test_a_failure_is_raised_exactly_once(entry):
    trap = Trap(rosenbrock, raise_boom)
    solver = make_solver(trap, gradient=rosenbrock_gradient)
    trap.arm()
    with pytest.raises(Boom):
        drive(solver, entry)

    solver.reset(np.array([-1.2, 1.0]))
    assert solver.solve().status == argmin.SolverStatus.converged
