"""Per-tier solver settings each arm is pre-registered to run with.

Only the overridden fields are listed; every other option stays at the library default and is
recorded verbatim in the run manifest.
"""

from __future__ import annotations


def _admm(eps: float, polish: bool) -> dict:
    return {"eps_abs": eps, "eps_rel": eps, "max_iterations": 4000, "warm_start": True, "polish": polish}


def _osqp(eps: float, polishing: bool) -> dict:
    return {
        "eps_abs": eps, "eps_rel": eps, "max_iter": 4000, "warm_starting": True, "polishing": polishing,
        "adaptive_rho": True, "adaptive_rho_interval": 25, "verbose": False,
    }


def _proxqp(eps: float, **extra: str) -> dict:
    return {"eps_abs": eps, "eps_rel": eps, "max_iter": 4000,
            "initial_guess": "WARM_START_WITH_PREVIOUS_RESULT", **extra}


def _hpipm(mode: str, tol: float, iter_max: int) -> dict:
    return {"mode": mode, "tol_stat": tol, "tol_eq": tol, "tol_ineq": tol, "tol_comp": tol,
            "iter_max": iter_max, "warm_start": 1}


def _tinympc(tol: float) -> dict:
    return {"abs_pri_tol": tol, "abs_dua_tol": tol, "max_iter": 4000, "rho": 1.0, "adaptive_rho": False}


ARM_SETTINGS = {
    "argmin_dense_admm": {"mpc": _admm(1e-3, False), "high": _admm(1e-6, True)},
    "argmin_dense_active_set": {"mpc": {"tolerance": 1e-3, "max_iterations": 4000},
                                "high": {"tolerance": 1e-6, "max_iterations": 4000}},
    "qpoases": {"mpc": {"preset": "MPC", "print_level": "NONE", "nwsr_cap": 4000},
                "high": {"preset": "RELIABLE", "print_level": "NONE", "nwsr_cap": 4000}},
    "daqp": {"mpc": {"primal_tol": 1e-3, "iter_limit": 4000}, "high": {"primal_tol": 1e-6, "iter_limit": 4000}},
    "proxqp_dense": {"mpc": _proxqp(1e-3), "high": _proxqp(1e-6)},
    "osqp": {"mpc": _osqp(1e-3, False), "high": _osqp(1e-6, True)},
    "argmin_dense_admm_polished": {"mpc": _admm(1e-3, True)},
    "argmin_sparse_admm": {"mpc": _admm(1e-3, False), "high": _admm(1e-6, True)},
    "proxqp_sparse": {"mpc": _proxqp(1e-3, backend="SparseCholesky"),
                      "high": _proxqp(1e-6, backend="SparseCholesky")},
    "hpipm": {"mpc": _hpipm("SPEED", 1e-3, 50), "high": _hpipm("BALANCE", 1e-6, 100)},
    "tinympc": {"mpc": _tinympc(1e-3), "high": _tinympc(1e-6)},
    "argmin_sparse_admm_polished": {"mpc": _admm(1e-3, True)},
    "osqp_polished": {"mpc": _osqp(1e-3, True)},
}
