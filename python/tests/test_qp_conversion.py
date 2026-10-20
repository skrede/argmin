"""Round-trip conversion: dtype, memory order, contiguity, writability, container.

Every case below has an asserted outcome. A case that is converted must reach
the same answer as the plain double-precision contiguous call; a case that is
not converted must raise. Nothing is left to "did not crash".
"""

import numpy as np
import pytest
import scipy.sparse as sp

OPTIMUM = np.array([0.3, 0.7])
FEASIBLE_START = np.array([0.5, 0.5])

DENSE = ["dense_admm", "dense_active_set"]


def reference_problem():
    P = np.array([[4.0, 1.0], [1.0, 2.0]])
    q = np.array([1.0, 1.0])
    A = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    l = np.array([1.0, 0.0, 0.0])
    u = np.array([1.0, 0.7, 0.7])
    return P, q, A, l, u


def dense_solver(argmin, kind, n=2, m=3):
    solver = getattr(
        argmin, "DenseAdmmQpSolver" if kind == "dense_admm" else "DenseActiveSetQpSolver"
    )(n, m)
    if kind == "dense_active_set":
        solver.warm_start(FEASIBLE_START)
    return solver


def noncontiguous(values):
    padded = np.zeros(tuple(2 * extent for extent in values.shape))
    padded[tuple(slice(None, None, 2) for _ in values.shape)] = values
    view = padded[tuple(slice(None, None, 2) for _ in values.shape)]
    assert not view.flags["C_CONTIGUOUS"]
    return view


def read_only(values):
    copied = values.copy()
    copied.setflags(write=False)
    return copied


@pytest.mark.parametrize("kind", DENSE)
@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.int32])
def test_a_narrower_dtype_converts_to_the_same_answer(argmin, kind, dtype):
    P, q, A, l, u = reference_problem()
    result = dense_solver(argmin, kind).solve(P.astype(dtype), q.astype(dtype), A.astype(dtype), l, u)
    assert np.allclose(result.x, OPTIMUM, atol=1e-6)


@pytest.mark.parametrize("kind", DENSE)
@pytest.mark.parametrize("order", ["C", "F"])
def test_either_memory_order_converts_to_the_same_answer(argmin, kind, order):
    P, q, A, l, u = reference_problem()
    result = dense_solver(argmin, kind).solve(
        np.asarray(P, order=order), q, np.asarray(A, order=order), l, u
    )
    assert np.allclose(result.x, OPTIMUM, atol=1e-6)


@pytest.mark.parametrize("kind", DENSE)
def test_a_non_contiguous_slice_converts_to_the_same_answer(argmin, kind):
    P, q, A, l, u = reference_problem()
    result = dense_solver(argmin, kind).solve(
        noncontiguous(P), noncontiguous(q), noncontiguous(A), noncontiguous(l), noncontiguous(u)
    )
    assert np.allclose(result.x, OPTIMUM, atol=1e-6)


@pytest.mark.parametrize("kind", DENSE)
def test_a_read_only_array_converts_to_the_same_answer(argmin, kind):
    P, q, A, l, u = reference_problem()
    result = dense_solver(argmin, kind).solve(
        read_only(P), read_only(q), read_only(A), read_only(l), read_only(u)
    )
    assert np.allclose(result.x, OPTIMUM, atol=1e-6)


@pytest.mark.parametrize("kind", DENSE)
def test_a_zero_strided_broadcast_view_converts_to_the_same_answer(argmin, kind):
    P, _, A, l, u = reference_problem()
    broadcast_q = np.broadcast_to(np.float64(1.0), (2,))
    assert broadcast_q.strides == (0,)
    result = dense_solver(argmin, kind).solve(P, broadcast_q, A, l, u)
    assert np.allclose(result.x, OPTIMUM, atol=1e-6)


# Nested lists are rejected by the array caster rather than converted. The
# rejection is the assertion: an unconverted container must not reach a solver.
@pytest.mark.parametrize("kind", DENSE)
def test_a_list_of_lists_is_rejected(argmin, kind):
    P, q, A, l, u = reference_problem()
    with pytest.raises(TypeError):
        dense_solver(argmin, kind).solve(P.tolist(), q.tolist(), A.tolist(), l.tolist(), u.tolist())


def test_the_compressed_column_form_solves(argmin):
    P, q, A, l, u = reference_problem()
    result = argmin.SparseAdmmQpSolver().solve(sp.csc_matrix(P), q, sp.csc_matrix(A), l, u)
    assert np.allclose(result.x, OPTIMUM, atol=1e-6)


def test_the_newer_compressed_column_container_solves(argmin):
    P, q, A, l, u = reference_problem()
    result = argmin.SparseAdmmQpSolver().solve(sp.csc_array(P), q, sp.csc_array(A), l, u)
    assert np.allclose(result.x, OPTIMUM, atol=1e-6)


@pytest.mark.parametrize("container", [sp.csr_matrix, sp.coo_matrix, np.asarray])
def test_any_other_layout_is_rejected_by_name(argmin, container):
    P, q, A, l, u = reference_problem()
    with pytest.raises(argmin.ArgminError) as raised:
        argmin.SparseAdmmQpSolver().solve(container(P), q, container(A), l, u)
    assert raised.value.kind == argmin.ErrorKind.invalid_array
    assert "compressed sparse column" in str(raised.value)
    assert ".tocsc()" in str(raised.value)


def test_explicitly_stored_zeros_still_solve(argmin):
    P, q, A, l, u = reference_problem()
    with_stored_zeros = sp.csc_matrix(
        (
            np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            np.array([0, 1, 2, 0, 1, 2]),
            np.array([0, 3, 6]),
        ),
        shape=(3, 2),
    )
    assert with_stored_zeros.nnz == 6
    assert np.allclose(with_stored_zeros.toarray(), A)
    result = argmin.SparseAdmmQpSolver().solve(sp.csc_matrix(P), q, with_stored_zeros, l, u)
    assert np.allclose(result.x, OPTIMUM, atol=1e-6)
