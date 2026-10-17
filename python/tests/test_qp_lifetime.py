"""Ownership across the boundary, asserted by observation rather than by design.

The solvers reuse their buffers across calls, so a returned array that was a
view onto solver storage would change under a caller who merely held on to a
previous answer, and would dangle once the solver was collected. Both are
asserted here on values, not on the absence of a crash.
"""

import gc

import numpy as np
import pytest
import scipy.sparse as sp

OPTIMUM = np.array([0.3, 0.7])
FEASIBLE_START = np.array([0.5, 0.5])

ALL = ["dense_admm", "dense_active_set", "sparse_admm"]


def reference_problem():
    P = np.array([[4.0, 1.0], [1.0, 2.0]])
    q = np.array([1.0, 1.0])
    A = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    l = np.array([1.0, 0.0, 0.0])
    u = np.array([1.0, 0.7, 0.7])
    return P, q, A, l, u


def make_solver(argmin, kind):
    if kind == "sparse_admm":
        return argmin.SparseAdmmQpSolver()
    solver = getattr(
        argmin, "DenseAdmmQpSolver" if kind == "dense_admm" else "DenseActiveSetQpSolver"
    )(2, 3)
    if kind == "dense_active_set":
        solver.warm_start(FEASIBLE_START)
    return solver


def call_solve(solver, kind, P, q, A, l, u):
    if kind == "sparse_admm":
        return solver.solve(sp.csc_matrix(P), q, sp.csc_matrix(A), l, u)
    return solver.solve(P, q, A, l, u)


@pytest.mark.parametrize("kind", ALL)
def test_a_result_outlives_the_solver_that_produced_it(argmin, kind):
    solver = make_solver(argmin, kind)
    result = call_solve(solver, kind, *reference_problem())
    x = result.x
    y = result.y
    expected_x = x.copy()
    expected_y = y.copy()

    del solver
    gc.collect()

    assert np.array_equal(result.x, expected_x)
    assert np.array_equal(x, expected_x)
    assert np.array_equal(y, expected_y)
    assert np.allclose(x, OPTIMUM, atol=1e-6)


@pytest.mark.parametrize("kind", ALL)
def test_a_result_is_unchanged_by_a_later_solve_on_the_same_solver(argmin, kind):
    P, q, A, l, u = reference_problem()
    solver = make_solver(argmin, kind)
    first = call_solve(solver, kind, P, q, A, l, u)
    first_x = first.x
    expected = first_x.copy()

    second = call_solve(solver, kind, P, np.array([-3.0, 0.0]), A, l, u)
    assert not np.allclose(second.x, expected, atol=1e-3), "the second solve must move the answer"

    assert np.array_equal(first.x, expected)
    assert np.array_equal(first_x, expected)


@pytest.mark.parametrize("kind", ALL)
def test_mutating_a_returned_array_reaches_nothing(argmin, kind):
    P, q, A, l, u = reference_problem()
    solver = make_solver(argmin, kind)
    result = call_solve(solver, kind, P, q, A, l, u)

    x = result.x
    assert x.flags["OWNDATA"]
    x[:] = 1234.5

    assert np.allclose(result.x, OPTIMUM, atol=1e-6)
    again = call_solve(solver, kind, P, q, A, l, u)
    assert np.allclose(again.x, OPTIMUM, atol=1e-6)


@pytest.mark.parametrize("kind", ALL)
def test_no_entry_point_modifies_its_inputs(argmin, kind):
    P, q, A, l, u = reference_problem()
    pristine = tuple(a.copy() for a in (P, q, A, l, u))

    solver = make_solver(argmin, kind)
    call_solve(solver, kind, P, q, A, l, u)
    solver.resolve(q, l, u)

    for actual, expected in zip((P, q, A, l, u), pristine):
        assert np.array_equal(actual, expected)


@pytest.mark.parametrize("kind", ALL)
def test_results_survive_their_solvers_being_dropped_in_a_loop(argmin, kind):
    P, q, A, l, u = reference_problem()
    kept = []
    for _ in range(200):
        solver = make_solver(argmin, kind)
        result = call_solve(solver, kind, P, q, A, l, u)
        kept.append((result, result.x))
        del solver
    gc.collect()

    for result, x in kept:
        assert np.allclose(x, OPTIMUM, atol=1e-6)
        assert np.array_equal(result.x, x)
