#ifndef HPP_GUARD_ARGMIN_OPTIONS_CONVERGENCE_DEFAULTS_H
#define HPP_GUARD_ARGMIN_OPTIONS_CONVERGENCE_DEFAULTS_H

#include "argmin/types.h"

#include <cstdint>

namespace argmin
{

template <scalar S>
struct convergence_defaults;

template <>
struct convergence_defaults<double>
{
    static constexpr double gradient_threshold = 1e-5;
    static constexpr double objective_threshold = 1e-10;
    // The default of Ipopt's tol option (Ipopt options reference), its bound on the scaled first-order optimality
    // error of Wachter and Biegler 2006, Math. Program. 106(1):25-57, Section 2.1.
    static constexpr double stationarity_threshold = 1e-8;
    static constexpr double step_threshold = 1e-8;
    // SNOPT's Major feasibility tolerance default (Gill, Murray and Saunders, User's Guide for SNOPT 7), applied here as
    // an absolute bound on the raw violation with no scaling by the constraint or iterate size.
    static constexpr double feasibility_tolerance = 1e-6;
    // The pair matches the FunctionTolerance and MaxStallGenerations defaults of MATLAB's ga, which stops on a test of
    // the same shape: the average relative change of the best value over the window falls below the tolerance.
    static constexpr double stall_threshold = 1e-6;
    static constexpr std::uint16_t stall_window = 50;
};

template <>
struct convergence_defaults<float>
{
    // A precision-class value keeps the double row's fraction of the available precision:
    // t_f = eps_f ^ (ln t_d / ln eps_d), rounded to one significant figure.
    static constexpr float gradient_threshold = 6e-3f;      // rule
    static constexpr float objective_threshold = 4e-5f;     // rule
    static constexpr float stationarity_threshold = 3e-4f;  // rule
    static constexpr float step_threshold = 3e-4f;          // rule
    static constexpr float feasibility_tolerance = 2e-3f;   // rule
    static constexpr float stall_threshold = 2e-3f;         // rule
    static constexpr std::uint16_t stall_window = 50;
};

}

#endif
