"""Name lookup across the top-level package and its deliberate opt-in submodule.

The surface gates must see both namespaces, or a name moved behind the opt-in
would quietly drop out of every coverage, sweep and naming check.
"""

import argmin
import argmin.experimental as experimental

NAMESPACES = (argmin, experimental)

EXPORTED = sorted(set(argmin.__all__) | set(experimental.__all__))

BOUND_MODULES = (argmin._argmin.__name__, argmin._argmin.experimental.__name__)


def namespace_of(name):
    for namespace in NAMESPACES:
        if name in namespace.__all__:
            return namespace
    raise AttributeError(f"{name} is exported by neither argmin nor argmin.experimental")


def bound(name):
    return getattr(namespace_of(name), name)
