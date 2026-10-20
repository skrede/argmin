"""A box passed to a constrained SQP solver reaches the policy and holds the iterate."""

import numpy as np
import pytest

import argmin

from nlp_factories import CONSTRAINED_ADAPTER, FACTORIES

LOWER = np.array([-1.0, -1.0])
UPPER = np.array([1.0, 1.0])

SOLVERS = {
    "kraft_slsqp": argmin.KraftSlsqpSolver,
    "filter_slsqp": argmin.FilterSlsqpSolver,
    "tr_sqp": argmin.TrSqpSolver,
}

SUCCESS = frozenset(
    {
        argmin.SolverStatus.converged,
        argmin.SolverStatus.xtol_reached,
        argmin.SolverStatus.ftol_reached,
    }
)

BOUND_FORMS = pytest.mark.parametrize(
    "bounds",
    [(LOWER, UPPER), (lambda: LOWER, lambda: UPPER)],
    ids=["arrays", "callables"],
)


def shifted_bowl(x):
    return float((x[0] - 3.0) ** 2 + (x[1] - 3.0) ** 2)


def shifted_bowl_gradient(x):
    return np.array([2.0 * (x[0] - 3.0), 2.0 * (x[1] - 3.0)])


def loose_cap(x):
    return np.array([10.0 - x[0] - x[1]])


def loose_cap_jacobian(x):
    return np.array([[-1.0, -1.0]])


def solve_in_the_box(method, lower, upper):
    solver = SOLVERS[method](
        objective=shifted_bowl,
        x0=np.array([0.0, 0.0]),
        gradient=shifted_bowl_gradient,
        constraints=loose_cap,
        constraint_jacobian=loose_cap_jacobian,
        num_equality=0,
        num_inequality=1,
        lower_bounds=lower,
        upper_bounds=upper,
    )
    return solver, solver.solve()


@pytest.mark.parametrize("method", sorted(SOLVERS))
@BOUND_FORMS
def test_a_box_holds_the_iterate_at_its_corner(method, bounds):
    solver, result = solve_in_the_box(method, *bounds)
    tolerance = solver.options().feasibility_tolerance
    assert np.all(result.x >= LOWER - tolerance)
    assert np.all(result.x <= UPPER + tolerance)
    np.testing.assert_allclose(result.x, UPPER, rtol=0.0, atol=1e-6)


@BOUND_FORMS
def test_the_trust_region_solver_certifies_the_corner(bounds):
    _, result = solve_in_the_box("tr_sqp", *bounds)
    assert result.status in SUCCESS


@pytest.mark.parametrize("method", ("filter_slsqp", "tr_sqp"))
def test_a_wrong_length_bound_is_rejected(method):
    with pytest.raises(argmin.ArgminError) as raised:
        FACTORIES[method](lower_bounds=np.array([-1.0, -1.0, -1.0]))
    assert raised.value.kind == argmin.ErrorKind.dimension_mismatch


@pytest.mark.parametrize("method", CONSTRAINED_ADAPTER)
def test_unset_bounds_construct_normally(method):
    assert FACTORIES[method](lower_bounds=None, upper_bounds=None).n == 2
