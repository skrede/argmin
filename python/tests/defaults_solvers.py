"""Unconfigured solvers and bound-type discovery that test_defaults_sweep parametrizes over."""

import inspect

import numpy as np

import argmin


def rosenbrock(x):
    return float((1.0 - x[0]) ** 2 + 5.0 * (x[1] - x[0] ** 2) ** 2)


def unit_disk(x):
    return np.array([1.0 - x[0] ** 2 - x[1] ** 2])


def unit_disk_jacobian(x):
    return np.array([[-2.0 * x[0], -2.0 * x[1]]])


def feasible_start():
    return np.array([0.3, 0.3])


# Constructed with the problem arguments and NOTHING else: any option keyword
# here would make the snapshot report what the test asked for rather than what
# the library defaults to.
UNCONFIGURED = {
    "LbfgsbSolver": lambda: argmin.LbfgsbSolver(objective=rosenbrock, x0=feasible_start()),
    "BobyqaSolver": lambda: argmin.BobyqaSolver(objective=rosenbrock, x0=feasible_start()),
    "CobylaSolver": lambda: argmin.CobylaSolver(
        objective=rosenbrock,
        x0=feasible_start(),
        constraints=unit_disk,
        num_equality=0,
        num_inequality=1,
    ),
    "SlsqpSolver": lambda: argmin.SlsqpSolver(
        objective=rosenbrock,
        x0=feasible_start(),
        constraints=unit_disk,
        constraint_jacobian=unit_disk_jacobian,
        num_equality=0,
        num_inequality=1,
    ),
    "FilterSlsqpSolver": lambda: argmin.FilterSlsqpSolver(
        objective=rosenbrock,
        x0=feasible_start(),
        constraints=unit_disk,
        constraint_jacobian=unit_disk_jacobian,
        num_equality=0,
        num_inequality=1,
    ),
    "CmaesSolver": lambda: argmin.CmaesSolver(objective=rosenbrock, x0=feasible_start()),
}

POLICY_OPTION_TYPE = {
    "LbfgsbSolver": "LbfgsbOptions",
    "BobyqaSolver": "BobyqaOptions",
    "CobylaSolver": "CobylaOptions",
    "SlsqpSolver": "SlsqpOptions",
    "FilterSlsqpSolver": "FilterSlsqpOptions",
    "CmaesSolver": "CmaesOptions",
}


def bound_value_types():
    found = {}
    for name in argmin.__all__:
        value = getattr(argmin, name, None)
        if not inspect.isclass(value) or value.__module__ != argmin._argmin.__name__:
            continue
        if issubclass(value, BaseException) or hasattr(value, "__members__"):
            continue
        if any(
            not attribute.startswith("_") and callable(getattr(value, attribute, None))
            for attribute in dir(value)
        ):
            continue
        try:
            value()
        except TypeError:
            continue
        found[name] = value
    return found
