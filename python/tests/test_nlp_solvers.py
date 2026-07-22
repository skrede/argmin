"""The five curated nonlinear methods: each solves, terminates, and refuses
arguments it cannot use."""

import gc
import weakref

import numpy as np
import pytest

import argmin

X0 = np.array([-1.2, 1.0])
FEASIBLE_START = np.array([0.3, 0.3])

TERMINAL_STATUSES = frozenset(
    status
    for status in argmin.SolverStatus.__members__.values()
    if status != argmin.SolverStatus.running
)


def rosenbrock(x):
    return float((1.0 - x[0]) ** 2 + 5.0 * (x[1] - x[0] ** 2) ** 2)


def rosenbrock_gradient(x):
    return np.array(
        [
            -2.0 * (1.0 - x[0]) - 20.0 * x[0] * (x[1] - x[0] ** 2),
            10.0 * (x[1] - x[0] ** 2),
        ]
    )


def unit_disk(x):
    return np.array([1.0 - x[0] ** 2 - x[1] ** 2])


def unit_disk_jacobian(x):
    return np.array([[-2.0 * x[0], -2.0 * x[1]]])


def make_lbfgsb(**overrides):
    arguments = dict(
        objective=rosenbrock,
        x0=X0,
        gradient=rosenbrock_gradient,
        max_iterations=500,
    )
    arguments.update(overrides)
    return argmin.LbfgsbSolver(**arguments)


def make_bobyqa(**overrides):
    arguments = dict(
        objective=rosenbrock,
        x0=X0,
        lower_bounds=np.array([-5.0, -5.0]),
        upper_bounds=np.array([5.0, 5.0]),
        max_iterations=2000,
    )
    arguments.update(overrides)
    return argmin.BobyqaSolver(**arguments)


def make_cobyla(**overrides):
    arguments = dict(
        objective=rosenbrock,
        x0=FEASIBLE_START,
        constraints=unit_disk,
        num_equality=0,
        num_inequality=1,
        max_iterations=3000,
    )
    arguments.update(overrides)
    return argmin.CobylaSolver(**arguments)


def make_slsqp(**overrides):
    arguments = dict(
        objective=rosenbrock,
        x0=FEASIBLE_START,
        gradient=rosenbrock_gradient,
        constraints=unit_disk,
        constraint_jacobian=unit_disk_jacobian,
        num_equality=0,
        num_inequality=1,
        max_iterations=500,
    )
    arguments.update(overrides)
    return argmin.SlsqpSolver(**arguments)


def make_cmaes(**overrides):
    arguments = dict(objective=rosenbrock, x0=X0, seed=20240722, max_iterations=800)
    arguments.update(overrides)
    return argmin.CmaesSolver(**arguments)


FACTORIES = {
    "lbfgsb": make_lbfgsb,
    "bobyqa": make_bobyqa,
    "cobyla": make_cobyla,
    "slsqp": make_slsqp,
    "cmaes": make_cmaes,
}

BOUNDED = ("lbfgsb", "bobyqa", "cobyla", "slsqp")
CONSTRAINED = ("cobyla", "slsqp")

ALL_METHODS = pytest.mark.parametrize("method", sorted(FACTORIES))


@ALL_METHODS
def test_the_method_constructs_and_reports_its_dimension(method):
    solver = FACTORIES[method]()
    assert solver.n == 2
    assert repr(solver).endswith("(n=2)")


@ALL_METHODS
def test_the_method_solves_and_reports_a_terminal_status(method):
    solver = FACTORIES[method]()
    result = solver.solve()
    assert result.status in TERMINAL_STATUSES
    assert solver.status() in TERMINAL_STATUSES
    assert result.x.shape == (2,)


@ALL_METHODS
def test_the_method_materially_reduces_the_objective(method):
    solver = FACTORIES[method]()
    start = rosenbrock(solver.x)
    result = solver.solve()
    assert result.objective_value < 0.5 * start


@pytest.mark.parametrize("method", CONSTRAINED)
def test_a_constrained_result_is_feasible_to_the_driver_tolerance(method):
    solver = FACTORIES[method]()
    result = solver.solve()
    tolerance = solver.options().feasibility_tolerance
    assert result.constraint_violation <= tolerance
    assert unit_disk(result.x)[0] >= -tolerance


@ALL_METHODS
def test_the_options_snapshot_reads_back_the_effective_configuration(method):
    solver = FACTORIES[method]()
    driver = solver.options()
    assert driver.max_iterations > 0
    assert driver.gradient_threshold > 0.0
    assert repr(solver.policy_options()).endswith(")")


@ALL_METHODS
def test_an_unspecified_keyword_keeps_the_library_default(method):
    specified = FACTORIES[method](gradient_threshold=1e-3)
    unspecified = FACTORIES[method]()
    assert specified.options().gradient_threshold == 1e-3
    assert (
        unspecified.options().gradient_threshold
        == argmin.SolverOptions().gradient_threshold
    )


# Every policy bound here carries its own stall window, and the driver copies
# that policy value over the one the driver keyword set -- unconditionally,
# unlike the max-of-two rule the other forwarded hints use. The driver keyword
# is therefore inert for these five methods, which is only visible at all
# because the options snapshot reads the effective value back.
@ALL_METHODS
def test_the_policy_stall_window_overrides_the_driver_keyword(method):
    solver = FACTORIES[method](stall_window=7, policy_stall_window=9)
    assert solver.options().stall_window == 9

    ignored = FACTORIES[method](stall_window=7)
    assert ignored.options().stall_window == ignored.policy_options().stall_window
    assert ignored.options().stall_window != 7


def test_the_stochastic_method_is_reproducible_at_a_fixed_seed():
    first = make_cmaes(seed=4242).solve()
    again = make_cmaes(seed=4242).solve()
    other = make_cmaes(seed=4243).solve()
    assert first.objective_value == again.objective_value
    assert np.array_equal(first.x, again.x)
    assert other.objective_value != first.objective_value


def test_the_seed_is_visible_in_the_policy_options_snapshot():
    assert make_cmaes(seed=99).policy_options().seed == 99
    assert make_cmaes().policy_options().seed is not None


@ALL_METHODS
def test_a_non_callable_objective_is_rejected(method):
    with pytest.raises(argmin.ArgminError) as raised:
        FACTORIES[method](objective=3.0)
    assert raised.value.kind == argmin.ErrorKind.invalid_callback


@ALL_METHODS
def test_an_empty_starting_point_is_rejected(method):
    with pytest.raises(argmin.ArgminError) as raised:
        FACTORIES[method](x0=np.array([]))
    assert raised.value.kind == argmin.ErrorKind.dimension_mismatch


@ALL_METHODS
def test_a_wrong_length_starting_point_is_rejected_on_reset(method):
    solver = FACTORIES[method]()
    with pytest.raises(argmin.ArgminError) as raised:
        solver.reset(np.array([0.0, 0.0, 0.0]))
    assert raised.value.kind == argmin.ErrorKind.dimension_mismatch


@ALL_METHODS
def test_a_non_finite_starting_point_is_rejected(method):
    with pytest.raises(argmin.ArgminError) as raised:
        FACTORIES[method](x0=np.array([np.nan, 1.0]))
    assert raised.value.kind == argmin.ErrorKind.non_finite_input


@pytest.mark.parametrize("method", BOUNDED)
def test_a_wrong_length_bound_is_rejected(method):
    with pytest.raises(argmin.ArgminError) as raised:
        FACTORIES[method](lower_bounds=np.array([-1.0, -1.0, -1.0]))
    assert raised.value.kind == argmin.ErrorKind.dimension_mismatch


@pytest.mark.parametrize("method", CONSTRAINED)
def test_a_non_callable_constraint_is_rejected(method):
    with pytest.raises(argmin.ArgminError) as raised:
        FACTORIES[method](constraints=np.array([1.0]))
    assert raised.value.kind == argmin.ErrorKind.invalid_callback


@pytest.mark.parametrize("method", CONSTRAINED)
def test_a_constraint_vector_of_the_wrong_length_is_rejected(method):
    with pytest.raises(argmin.ArgminError) as raised:
        FACTORIES[method](constraints=lambda x: np.array([1.0, 2.0])).solve()
    assert raised.value.kind == argmin.ErrorKind.invalid_callback


@pytest.mark.parametrize("method", CONSTRAINED)
def test_a_constraint_count_that_is_negative_is_rejected(method):
    with pytest.raises(argmin.ArgminError) as raised:
        FACTORIES[method](num_inequality=-1)
    assert raised.value.kind == argmin.ErrorKind.dimension_mismatch


def test_a_jacobian_of_the_wrong_shape_is_rejected():
    with pytest.raises(argmin.ArgminError) as raised:
        make_slsqp(constraint_jacobian=lambda x: np.zeros((2, 2))).solve()
    assert raised.value.kind == argmin.ErrorKind.invalid_callback


# The cycle runs interpreter -> solver -> adapter -> bound method -> owner ->
# solver, and the collector cannot see the C++ leg of it without the wrapper's
# traverse and clear slots. Observing the owner is how the cycle becomes visible
# from the interpreter, since the solver itself is not weak-referenceable.
@ALL_METHODS
def test_the_solver_is_collectable_inside_a_reference_cycle(method):
    class Owner:
        def __init__(self):
            self.solver = None

        def objective(self, x):
            return rosenbrock(x)

    owner = Owner()
    owner.solver = FACTORIES[method](objective=owner.objective)
    owner.solver.solve()
    observer = weakref.ref(owner)

    del owner
    gc.collect()

    assert observer() is None
