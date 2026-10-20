"""A COBYLA box with an empty or zero-width coordinate is refused at construction."""

import numpy as np
import pytest

import argmin

from surface_namespaces import bound

CobylaSolver = bound("CobylaSolver")

SUCCESS = frozenset(
    {
        argmin.SolverStatus.converged,
        argmin.SolverStatus.xtol_reached,
        argmin.SolverStatus.ftol_reached,
    }
)


class Boom(Exception):
    pass


def bowl(x):
    return float(x[0] ** 2 + x[1] ** 2)


def cap(x):
    return np.array([100.0 - x[0] - x[1]])


def make_solver(**bounds):
    return CobylaSolver(
        objective=bowl,
        x0=np.array([0.5, 0.2]),
        constraints=cap,
        num_equality=0,
        num_inequality=1,
        **bounds,
    )


def raise_boom():
    raise Boom("the bound callable declined")


def assert_invalid_bounds(raised, index):
    assert raised.value.kind == argmin.ErrorKind.invalid_bounds
    assert f"index {index}" in str(raised.value)


def test_an_inverted_box_is_refused_naming_its_coordinate():
    with pytest.raises(argmin.ArgminError) as raised:
        make_solver(lower_bounds=np.array([1.0, -1.0]), upper_bounds=np.array([0.0, 1.0]))
    assert_invalid_bounds(raised, 0)
    assert "exceeds" in str(raised.value)


def test_an_equal_coordinate_is_refused_naming_its_coordinate():
    with pytest.raises(argmin.ArgminError) as raised:
        make_solver(lower_bounds=np.array([-1.0, 0.5]), upper_bounds=np.array([1.0, 0.5]))
    assert_invalid_bounds(raised, 1)
    assert "equals" in str(raised.value)


def test_callable_bounds_returning_an_inverted_box_are_refused():
    with pytest.raises(argmin.ArgminError) as raised:
        make_solver(
            lower_bounds=lambda: np.array([1.0, -1.0]),
            upper_bounds=lambda: np.array([0.0, 1.0]),
        )
    assert_invalid_bounds(raised, 0)


def test_a_raising_bound_callable_surfaces_its_own_exception():
    with pytest.raises(Boom, match="the bound callable declined"):
        make_solver(lower_bounds=lambda: raise_boom())


def test_a_one_sided_box_constructs_and_solves():
    result = make_solver(lower_bounds=np.array([-1.0, 0.0])).solve()
    assert result.status in SUCCESS


def test_an_unbounded_problem_constructs_and_solves():
    result = make_solver().solve()
    assert result.status in SUCCESS


def test_the_narrow_coordinate_witness_evaluates_only_inside_its_box():
    lower = np.array([0.0, -10.0])
    upper = np.array([0.6, 10.0])
    points = []

    def objective(x):
        points.append(np.array(x))
        return float((x[0] - 0.3) ** 2 + (x[1] - 1.0) ** 2)

    def constraints(x):
        points.append(np.array(x))
        return np.array([100.0 - x[0] - x[1]])

    solver = CobylaSolver(
        objective=objective,
        x0=np.array([0.5, 1.0]),
        constraints=constraints,
        num_equality=0,
        num_inequality=1,
        lower_bounds=lower,
        upper_bounds=upper,
    )
    assert solver.clamped_evaluations == 0
    result = solver.solve()
    assert result.status in SUCCESS
    assert solver.clamped_evaluations == 0
    assert points
    assert all(np.all(point >= lower) and np.all(point <= upper) for point in points)
