// Compile-isolation test for the SQP scaffolding header.
//
// Verifies that detail/sqp_common.h composes without unresolved
// dependencies and that sqp_state_buffers instantiates in both the
// fixed-N and dynamic-N flavors. The forward-declared helper
// functions are NOT called here because their bodies live in a
// follow-up plan; calling them would link-fail.
//
// Also exercises the no-op path of detail/bench/alloc_counter.h
// (without ARGMIN_BENCH_TRACE_ALLOC defined the API is a sized-zero
// counter).

#include "argmin/types.h"
#include "argmin/detail/sqp_common.h"
#include "argmin/detail/bench/alloc_counter.h"

#include <Eigen/Core>

#include <cstddef>

int main()
{
    argmin::detail::sqp_state_buffers<double, 4> fixed_buf;
    fixed_buf.resize(4, 1, 2);

    argmin::detail::sqp_state_buffers<double, argmin::dynamic_dimension> dyn_buf;
    dyn_buf.resize(4, 1, 2);

    argmin::detail::bench::reset_alloc_count();
    argmin::detail::bench::arm_alloc_trace();
    argmin::detail::bench::disarm_alloc_trace();
    const std::size_t count = argmin::detail::bench::read_alloc_count();
    (void)count;

    return 0;
}
