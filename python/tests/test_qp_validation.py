"""Shape, range and state violations at every entry point of all three solvers.

Every case asserts the exception type and its kind. The C++ concept layer stops
a native caller at compile time; an interpreter caller can hand any shape to
anything, and in a sibling project that gap was an out-of-bounds read only
Python callers could reach. This module therefore also has to establish that
none of these violations takes the interpreter down, which is why the
verification runs it as its own process.
"""

import numpy as np
import pytest
import scipy.sparse as sp

FEASIBLE_START = np.array([0.5, 0.5])

ALL = ["dense_admm", "dense_active_set", "sparse_admm"]
DENSE = ["dense_admm", "dense_active_set"]


def reference_problem():
    P = np.array([[4.0, 1.0], [1.0, 2.0]])
    q = np.array([1.0, 1.0])
    A = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    l = np.array([1.0, 0.0, 0.0])
    u = np.array([1.0, 0.7, 0.7])
    return P, q, A, l, u


def make_solver(argmin, kind, n=2, m=3):
    if kind == "sparse_admm":
        return argmin.SparseAdmmQpSolver()
    solver = getattr(
        argmin, "DenseAdmmQpSolver" if kind == "dense_admm" else "DenseActiveSetQpSolver"
    )(n, m)
    if kind == "dense_active_set":
        solver.warm_start(FEASIBLE_START)
    return solver


def call_solve(solver, kind, P, q, A, l, u):
    if kind == "sparse_admm":
        return solver.solve(sp.csc_matrix(P), q, sp.csc_matrix(A), l, u)
    return solver.solve(P, q, A, l, u)


def posed_solver(argmin, kind):
    solver = make_solver(argmin, kind)
    call_solve(solver, kind, *reference_problem())
    return solver


def expect(argmin, kind_name):
    return getattr(argmin.ErrorKind, kind_name)


@pytest.mark.parametrize("kind", ALL)
def test_a_non_square_coefficient_matrix_is_rejected(argmin, kind):
    _, q, A, l, u = reference_problem()
    P = np.array([[4.0, 1.0, 0.0], [1.0, 2.0, 0.0]])
    with pytest.raises(argmin.ArgminError) as raised:
        call_solve(make_solver(argmin, kind), kind, P, q, A, l, u)
    assert raised.value.kind == expect(argmin, "dimension_mismatch")


@pytest.mark.parametrize("kind", ALL)
def test_a_constraint_matrix_of_the_wrong_width_is_rejected(argmin, kind):
    P, q, _, l, u = reference_problem()
    A = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(argmin.ArgminError) as raised:
        call_solve(make_solver(argmin, kind), kind, P, q, A, l, u)
    assert raised.value.kind == expect(argmin, "dimension_mismatch")


@pytest.mark.parametrize("kind", ALL)
def test_a_right_hand_side_of_the_wrong_length_is_rejected(argmin, kind):
    P, _, A, l, u = reference_problem()
    with pytest.raises(argmin.ArgminError) as raised:
        call_solve(make_solver(argmin, kind), kind, P, np.ones(3), A, l, u)
    assert raised.value.kind == expect(argmin, "dimension_mismatch")


@pytest.mark.parametrize("kind", ALL)
def test_mismatched_bound_lengths_are_rejected(argmin, kind):
    P, q, A, l, _ = reference_problem()
    with pytest.raises(argmin.ArgminError) as raised:
        call_solve(make_solver(argmin, kind), kind, P, q, A, l, np.ones(2))
    assert raised.value.kind == expect(argmin, "dimension_mismatch")


@pytest.mark.parametrize("kind", ALL)
def test_an_inverted_bound_pair_is_rejected(argmin, kind):
    P, q, A, l, u = reference_problem()
    inverted = l.copy()
    inverted[1] = 0.9
    with pytest.raises(argmin.ArgminError) as raised:
        call_solve(make_solver(argmin, kind), kind, P, q, A, inverted, u)
    assert raised.value.kind == expect(argmin, "invalid_bounds")


@pytest.mark.parametrize("kind", ALL)
@pytest.mark.parametrize("argument", ["P", "q", "A", "l", "u"])
def test_a_value_that_is_not_a_number_is_rejected_anywhere(argmin, kind, argument):
    data = dict(zip(("P", "q", "A", "l", "u"), (a.copy() for a in reference_problem())))
    data[argument].flat[0] = np.nan
    with pytest.raises(argmin.ArgminError) as raised:
        call_solve(make_solver(argmin, kind), kind, **data)
    assert raised.value.kind == expect(argmin, "non_finite_input")


@pytest.mark.parametrize("kind", ALL)
def test_an_infinite_coefficient_is_rejected(argmin, kind):
    P, q, A, l, u = reference_problem()
    P = P.copy()
    P[1, 1] = np.inf
    with pytest.raises(argmin.ArgminError) as raised:
        call_solve(make_solver(argmin, kind), kind, P, q, A, l, u)
    assert raised.value.kind == expect(argmin, "non_finite_input")


# Infinite bounds are how a one-sided constraint is spelled, so they are the one
# non-finite input that must pass.
@pytest.mark.parametrize("kind", ALL)
def test_an_infinite_bound_is_accepted(argmin, kind):
    P, q, A, l, u = reference_problem()
    l = np.array([1.0, -np.inf, -np.inf])
    u = np.array([1.0, np.inf, np.inf])
    result = call_solve(make_solver(argmin, kind), kind, P, q, A, l, u)
    assert result.status == argmin.QpStatus.solved
    assert np.allclose(result.x, [0.25, 0.75], atol=1e-6)


@pytest.mark.parametrize("kind", DENSE)
@pytest.mark.parametrize("axis", ["decision", "constraint"])
def test_a_request_beyond_the_constructed_capacity_is_rejected(argmin, kind, axis):
    if axis == "decision":
        P = np.eye(3)
        q = np.zeros(3)
        A = np.zeros((3, 3))
    else:
        P, q, _, _, _ = reference_problem()
        A = np.zeros((5, 2))
    rows = A.shape[0]
    l = -np.ones(rows)
    u = np.ones(rows)
    with pytest.raises(argmin.ArgminError) as raised:
        call_solve(make_solver(argmin, kind), kind, P, q, A, l, u)
    assert raised.value.kind == expect(argmin, "capacity_exceeded")


@pytest.mark.parametrize("kind", DENSE)
@pytest.mark.parametrize("dimensions", [(0, 3), (-1, 3), (2, 0), (2, -4)])
def test_a_non_positive_constructor_dimension_is_rejected(argmin, kind, dimensions):
    with pytest.raises(argmin.ArgminError) as raised:
        make_solver(argmin, kind, *dimensions)
    assert raised.value.kind == expect(argmin, "dimension_mismatch")


@pytest.mark.parametrize("kind", ALL)
def test_resolve_before_any_solve_is_rejected(argmin, kind):
    _, q, _, l, u = reference_problem()
    with pytest.raises(argmin.ArgminError) as raised:
        make_solver(argmin, kind).resolve(q, l, u)
    assert raised.value.kind == expect(argmin, "invalid_state")


@pytest.mark.parametrize("kind", ALL)
@pytest.mark.parametrize(
    "argument,kind_name",
    [
        ("q_length", "dimension_mismatch"),
        ("bound_length", "dimension_mismatch"),
        ("inverted", "invalid_bounds"),
        ("nan_q", "non_finite_input"),
        ("nan_l", "non_finite_input"),
        ("nan_u", "non_finite_input"),
        ("infinite_q", "non_finite_input"),
    ],
)
def test_resolve_rejects_the_same_violations(argmin, kind, argument, kind_name):
    _, q, _, l, u = reference_problem()
    solver = posed_solver(argmin, kind)
    q, l, u = q.copy(), l.copy(), u.copy()
    if argument == "q_length":
        q = np.ones(3)
    elif argument == "bound_length":
        u = np.ones(2)
    elif argument == "inverted":
        l[1] = 0.9
    elif argument == "nan_q":
        q[0] = np.nan
    elif argument == "nan_l":
        l[0] = np.nan
    elif argument == "nan_u":
        u[0] = np.nan
    elif argument == "infinite_q":
        q[1] = np.inf
    with pytest.raises(argmin.ArgminError) as raised:
        solver.resolve(q, l, u)
    assert raised.value.kind == expect(argmin, kind_name)


@pytest.mark.parametrize("kind", DENSE)
def test_a_warm_start_of_the_wrong_length_is_rejected(argmin, kind):
    with pytest.raises(argmin.ArgminError) as raised:
        make_solver(argmin, kind).warm_start(np.ones(5))
    assert raised.value.kind == expect(argmin, "dimension_mismatch")


def test_the_interpreter_survives_every_violation(argmin):
    P, q, A, l, u = reference_problem()
    for _ in range(200):
        for bad in (np.ones(3), np.full(2, np.nan)):
            with pytest.raises(argmin.ArgminError):
                make_solver(argmin, "dense_admm").solve(P, bad, A, l, u)
    result = make_solver(argmin, "dense_admm").solve(P, q, A, l, u)
    assert np.allclose(result.x, [0.3, 0.7], atol=1e-6)
