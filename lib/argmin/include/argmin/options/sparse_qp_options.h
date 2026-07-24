#ifndef HPP_GUARD_ARGMIN_OPTIONS_SPARSE_QP_OPTIONS_H
#define HPP_GUARD_ARGMIN_OPTIONS_SPARSE_QP_OPTIONS_H

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
// Reference: Stellato, Banjac, Goulart, Bemporad, Boyd (2020), "OSQP: An
//            operator splitting solver for quadratic programs"; OSQP solver
//            settings defaults.
struct sparse_qp_options
{
    double rho{0.1};
    double sigma{1e-6};
    double alpha{1.6};
    double eps_abs{1e-6};
    double eps_rel{1e-6};
    double eps_prim_inf{1e-4};
    double eps_dual_inf{1e-4};
    std::uint16_t max_iterations{4000};
    std::uint16_t check_termination{25};
    bool adaptive_rho{true};
    std::uint16_t adaptive_rho_interval{50};
    double adaptive_rho_tolerance{5.0};
    std::uint16_t scaling{10};
    bool polish{true};
    std::uint16_t polish_refine_iter{3};
    double delta{1e-6};
    bool warm_start{true};
};

}

#endif
