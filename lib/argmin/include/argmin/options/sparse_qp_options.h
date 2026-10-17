#ifndef HPP_GUARD_ARGMIN_OPTIONS_SPARSE_QP_OPTIONS_H
#define HPP_GUARD_ARGMIN_OPTIONS_SPARSE_QP_OPTIONS_H

#include "argmin/options/admm_defaults.h"
#include "argmin/types.h"

#include <cstdint>

namespace argmin
{

// Tuning knobs for the sparse operator-splitting QP solver.
//
// The brace-initialized literals are the OSQP solver-settings defaults, with
// two deliberate departures from OSQP:
//   (a) adaptive_rho_interval is an iteration count, not a fraction of measured
//       setup time -- this library takes no timing measurements, so the
//       schedule triggers on iterations rather than OSQP's default time-based
//       rule;
//   (b) polish defaults true (OSQP ships it false) because the polished,
//       SQP-inner-grade result is this solver's reason to exist.
//
// The field roster coincides with dense_qp_options today by construction, not
// by contract: the two solvers differ algorithmically (KKT form, scaling pass,
// polish) and their knobs are free to diverge without either type following the
// other.
//
// validate_convexity gates a pose-time symmetry check on P, not an inertia test: on this
// path positive semidefiniteness is assumed rather than verified.
//
// Reference: Stellato, Banjac, Goulart, Bemporad, Boyd (2020), "OSQP: An
//            operator splitting solver for quadratic programs"; OSQP solver
//            settings defaults.
template <scalar Scalar = double>
struct sparse_qp_options
{
    Scalar rho{admm_defaults<Scalar>::rho};
    Scalar sigma{admm_defaults<Scalar>::sigma};
    Scalar alpha{admm_defaults<Scalar>::alpha};
    Scalar eps_abs{admm_defaults<Scalar>::eps_abs};
    Scalar eps_rel{admm_defaults<Scalar>::eps_rel};
    Scalar eps_prim_inf{admm_defaults<Scalar>::eps_prim_inf};
    Scalar eps_dual_inf{admm_defaults<Scalar>::eps_dual_inf};
    std::uint32_t max_iterations{4000};
    std::uint16_t check_termination{25};
    bool adaptive_rho{true};
    std::uint16_t adaptive_rho_interval{50};
    Scalar adaptive_rho_tolerance{admm_defaults<Scalar>::adaptive_rho_tolerance};
    std::uint16_t scaling{10};
    bool polish{true};
    std::uint16_t polish_refine_iter{3};
    Scalar delta{admm_defaults<Scalar>::delta};
    bool warm_start{true};
    bool validate_convexity{true};
};

}

#endif
