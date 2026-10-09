#ifndef HPP_GUARD_ARGMIN_OPTIONS_DENSE_QP_OPTIONS_H
#define HPP_GUARD_ARGMIN_OPTIONS_DENSE_QP_OPTIONS_H

#include "argmin/options/admm_defaults.h"
#include "argmin/types.h"

#include <cstdint>

namespace argmin
{

// Tuning knobs for the dense operator-splitting QP solver.
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
// validate_convexity gates a pose-time symmetry and inertia check on P that costs
// one extra factorization per pose and nothing per resolve; it defaults on because
// an indefinite P can still yield a factorable condensed matrix and a non-global
// point reported under a clean status.
//
// Reference: Stellato, Banjac, Goulart, Bemporad, Boyd (2020), "OSQP: An
//            operator splitting solver for quadratic programs"; OSQP solver
//            settings defaults.
template <scalar Scalar = double>
struct dense_qp_options
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
