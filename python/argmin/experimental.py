"""Importing this module is the opt-in to solvers that carry a documented known defect."""

from ._argmin.experimental import (
    BobyqaOptions,
    BobyqaSolver,
    CmaesDetectionOptions,
    CmaesOptions,
    CmaesRestartStrategy,
    CmaesSolver,
    TrustRegionOptions,
    solve_native,
)

__all__ = [
    "BobyqaOptions",
    "BobyqaSolver",
    "CmaesDetectionOptions",
    "CmaesOptions",
    "CmaesRestartStrategy",
    "CmaesSolver",
    "TrustRegionOptions",
    "solve_native",
]
