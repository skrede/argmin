"""Exception propagation from the constraint and Jacobian callbacks.

The objective path is covered on its own; these cases hold the constrained
adapter to the same three properties: the caller observes the exception the
callback raised, the adapter stops entering the interpreter once the failure is
latched, and the solver can be restarted afterwards.
"""

import numpy as np
import pytest

import argmin

from nlp_factories import (
    CONSTRAINED_ADAPTER,
    FACTORIES,
    FEASIBLE_START,
    unit_disk,
    unit_disk_jacobian,
)

BOOM_MESSAGE = "the callback declined"

METHODS = pytest.mark.parametrize("method", CONSTRAINED_ADAPTER)
ENTRY_POINTS = pytest.mark.parametrize("entry", ["solve", "step"])


class Boom(Exception):
    pass


class Trap:
    def __init__(self, normal, failure):
        self.normal = normal
        self.failure = failure
        self.armed = False
        self.failed = False
        self.calls_after_failure = 0

    def arm(self):
        self.armed = True

    def __call__(self, x):
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


def armed_solver(method, callback, failure):
    normal = unit_disk if callback == "constraints" else unit_disk_jacobian
    trap = Trap(normal, failure)
    solver = FACTORIES[method](**{callback: trap})
    trap.arm()
    return solver, trap


@METHODS
def test_constraints_raising_before_the_loop_surface_at_construction(method):
    with pytest.raises(Boom, match=BOOM_MESSAGE):
        FACTORIES[method](constraints=raise_boom)


@METHODS
def test_a_jacobian_raising_before_the_loop_surfaces_at_construction(method):
    with pytest.raises(Boom, match=BOOM_MESSAGE):
        FACTORIES[method](constraint_jacobian=raise_boom)


@METHODS
@ENTRY_POINTS
@pytest.mark.parametrize("callback", ["constraints", "constraint_jacobian"])
def test_a_constraint_callback_raising_mid_solve(method, entry, callback):
    solver, trap = armed_solver(method, callback, raise_boom)
    with pytest.raises(Boom, match=BOOM_MESSAGE):
        drive(solver, entry)
    assert trap.failed
    assert trap.calls_after_failure <= 1


@METHODS
@ENTRY_POINTS
def test_a_constraint_vector_of_the_wrong_length_mid_solve(method, entry):
    solver, trap = armed_solver(method, "constraints", lambda _x: np.zeros(3))
    with pytest.raises(argmin.ArgminError) as raised:
        drive(solver, entry)
    assert raised.value.kind == argmin.ErrorKind.invalid_callback
    assert trap.calls_after_failure <= 1


@METHODS
@ENTRY_POINTS
def test_a_jacobian_of_the_wrong_shape_mid_solve(method, entry):
    solver, trap = armed_solver(method, "constraint_jacobian", lambda _x: np.zeros((2, 2)))
    with pytest.raises(argmin.ArgminError) as raised:
        drive(solver, entry)
    assert raised.value.kind == argmin.ErrorKind.invalid_callback
    assert trap.calls_after_failure <= 1


@METHODS
def test_a_jacobian_of_the_wrong_shape_is_rejected(method):
    with pytest.raises(argmin.ArgminError) as raised:
        FACTORIES[method](constraint_jacobian=lambda x: np.zeros((2, 2))).solve()
    assert raised.value.kind == argmin.ErrorKind.invalid_callback


@METHODS
@ENTRY_POINTS
@pytest.mark.parametrize("callback", ["constraints", "constraint_jacobian"])
def test_a_latched_constraint_failure_is_raised_exactly_once(method, entry, callback):
    solver, _ = armed_solver(method, callback, raise_boom)
    with pytest.raises(Boom):
        drive(solver, entry)

    solver.reset(FEASIBLE_START)
    assert solver.solve().status == argmin.SolverStatus.converged
