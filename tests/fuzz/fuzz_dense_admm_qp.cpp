#include "harness.h"
#include "qp_check.h"
#include "qp_input.h"
#include "qp_options.h"

#include "oracle_tolerances.h"

#include "argmin/qp/dense_admm_qp.h"

#include "argmin/options/dense_qp_options.h"

#include <span>
#include <cstddef>
#include <cstdint>

namespace
{

using namespace argmin::fuzzing;

using options_type = argmin::dense_qp_options<double>;
using solver_type = argmin::dense_admm_qp_solver<double>;

void seed_and_resolve(field_reader& fields, solver_type& solver, const qp_problem& p, const options_type& opts, qp_capacity capacity)
{
    const qp_sequence sequence = decode_sequence(fields);
    if(sequence.seed_after)
    {
        const qp_seed seed = decode_seed(fields, capacity);
        check_seed(solver.warm_start(seed.x, seed.y), seed_faults(seed, capacity));
    }
    if(!sequence.resolve)
        return;
    const qp_vectors v = decode_resolve(fields, p);
    if(unsatisfiable_infinite_row(v.l, v.u))
        return;
    error_set justified;
    hessian_faults(p, justified);
    vector_faults(v.q, v.l, v.u, justified);
    const auto outcome = solver.resolve(v.q, v.l, v.u, opts);
    if(admitted(outcome, justified))
        judge(p.P, v, p.A, *outcome, argmin::oracle::qp_tolerances_of(opts), true);
}

}

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size)
{
    field_reader fields{std::span<const std::uint8_t>(data, size)};
    const qp_problem p = decode_problem(fields, 3);
    const options_type opts = admm_options<options_type>(fields, p.shape);
    const qp_capacity capacity = decode_capacity(fields, p);
    if(empty_dense_problem(static_cast<std::int32_t>(p.q.size()), capacity.n) || unsatisfiable_infinite_row(p.l, p.u))
        return 0;
    solver_type solver(capacity.n, capacity.m);
    if((fields.byte() & 1) != 0)
        check_seed(solver.warm_start(padded(p.x_f, capacity.n), Eigen::VectorXd()), error_set());
    const auto outcome = solver.solve(p.P, p.q, p.A, p.l, p.u, opts);
    if(!admitted(outcome, solve_faults(p, admm_options_valid(opts), capacity)))
        return 0;
    judge(p.P, qp_vectors{p.q, p.l, p.u}, p.A, *outcome, argmin::oracle::qp_tolerances_of(opts), true);
    seed_and_resolve(fields, solver, p, opts, capacity);
    return 0;
}
