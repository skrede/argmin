"""The solver factories and registries the nonlinear-solver test modules parametrize over."""

import numpy as np

import argmin
import argmin.experimental as experimental

X0 = np.array([-1.2, 1.0])
FEASIBLE_START = np.array([0.3, 0.3])


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
    return experimental.BobyqaSolver(**arguments)


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


def make_kraft_slsqp(**overrides):
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
    return argmin.KraftSlsqpSolver(**arguments)


def make_filter_slsqp(**overrides):
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
    return argmin.FilterSlsqpSolver(**arguments)


def make_tr_sqp(**overrides):
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
    return argmin.TrSqpSolver(**arguments)


def make_cmaes(**overrides):
    arguments = dict(objective=rosenbrock, x0=X0, seed=20240722, max_iterations=800)
    arguments.update(overrides)
    return experimental.CmaesSolver(**arguments)


FACTORIES = {
    "lbfgsb": make_lbfgsb,
    "bobyqa": make_bobyqa,
    "cobyla": make_cobyla,
    "kraft_slsqp": make_kraft_slsqp,
    "filter_slsqp": make_filter_slsqp,
    "tr_sqp": make_tr_sqp,
    "cmaes": make_cmaes,
}

BOUNDED = ("lbfgsb", "bobyqa", "cobyla", "kraft_slsqp")
CONSTRAINED = ("cobyla", "kraft_slsqp", "filter_slsqp", "tr_sqp")
CONSTRAINED_ADAPTER = ("filter_slsqp", "kraft_slsqp", "tr_sqp")
