#ifndef HPP_GUARD_ARGMIN_TESTS_UNIT_ALLOC_GATE_SUPPORT_H
#define HPP_GUARD_ARGMIN_TESTS_UNIT_ALLOC_GATE_SUPPORT_H

// Zero-allocation correctness gates for the real-time-claimed policies.
//
// Every case arms the dual allocation sensor (Sensor A: the Eigen-native
// counting eigen_assert; Sensor B: the glibc C-allocator overrides on Linux)
// across a steady-state hot-loop window and asserts the armed window is
// allocation-free. These are argmin-only gates -- no NLopt, no external
// comparison library -- so they run under the plain `dev` preset that ships no
// comparison libraries.
//
// The gate target defines BOTH ARGMIN_BENCH_TRACE_ALLOC=1 (arming the LIVE
// sensor branch of the diagnostics header) and EIGEN_RUNTIME_NO_MALLOC (routing
// Eigen's runtime malloc gate into the counting hook). Without
// ARGMIN_BENCH_TRACE_ALLOC the header compiles to its no-op twin and every read
// is a vacuous 0; the alloc_sensor_liveness_canary case is the runtime proof
// that the LIVE branch is selected -- it counts an armed Eigen allocation and
// fails loud if the sensor is blind.
//
// This diagnostics header must precede any other Eigen-including header so its
// counting eigen_assert is defined before Eigen/Core is first parsed; every gate
// translation unit therefore includes this header first.
#include "argmin/detail/diagnostics/alloc_counter.h"
#include "argmin/detail/diagnostics/steady_state_driver.h"

#include "alloc_gate_fixtures.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>

namespace argmin::alloc_gate
{

namespace sensor = argmin::detail::bench;

struct window_counts
{
    std::size_t eigen;
    std::size_t c;
};

// Arm the sensor, run the measurement body, disarm, and return the observed
// counts. body() runs entirely inside the armed window; anything it observes
// (path-entry flags, mu changes) is captured through its own captures.
template <typename Body>
window_counts run_armed(Body&& body)
{
    sensor::reset_alloc_count();
    sensor::arm_alloc_trace();
    body();
    sensor::disarm_alloc_trace();
    return {sensor::read_eigen_malloc_count(), sensor::read_c_alloc_count()};
}

// Assert an armed steady-state window drove a pre-convergence, allocation-free
// hot loop.
inline void check_steady_zero(const sensor::steady_state_result& r)
{
    CHECK(r.eigen_malloc == 0);
    CHECK(r.c_alloc == 0);
    CHECK_FALSE(r.terminated_early);
}

}

#endif
