#include "harness.h"
#include "qp_check.h"
#include "qp_input.h"
#include "qp_options.h"

#include "oracle_tolerances.h"

#include "argmin/qp/qp_types.h"
#include "argmin/qp/dense_active_set_qp.h"

#include "argmin/options/dense_active_set_qp_options.h"

#include <span>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace
{

using namespace argmin::fuzzing;

using options_type = argmin::dense_active_set_qp_options<double>;
using solver_type = argmin::dense_active_set_qp_solver<double>;

bool start_violates(const Eigen::MatrixXd& A, const qp_vectors& v, const Eigen::VectorXd& x0, double equality_tolerance)
{
    const double slack = 0.5 * argmin::active_set_qp_defaults<double>::feas_tol;
    const Eigen::VectorXd ax = A * x0;
    for(Eigen::Index i = 0; i < ax.size(); ++i)
    {
        const bool equality = argmin::detail::is_equality_bound(v.l[i], v.u[i], equality_tolerance);
        if(equality ? std::abs(ax[i] - v.l[i]) > slack : (ax[i] < v.l[i] - slack || ax[i] > v.u[i] + slack))
            return true;
    }
    return false;
}

// The method's documented limits: at most n equality rows and a feasible start.
void contract_faults(const qp_problem& p, const qp_vectors& v, const Eigen::VectorXd& x0, const options_type& opts, error_set& justified)
{
    Eigen::Index equalities = 0;
    for(Eigen::Index i = 0; i < v.l.size(); ++i)
        equalities += argmin::detail::is_equality_bound(v.l[i], v.u[i], opts.equality_tolerance) ? 1 : 0;
    if(equalities > p.q.size())
        justified.allow(argmin::qp_error::invalid_problem);
    if(start_violates(p.A, v, x0, opts.equality_tolerance))
        justified.allow(argmin::qp_error::infeasible_start);
}

template <typename Result>
void judge_active_set(const qp_problem& p, const qp_vectors& v, const Result& result, const options_type& opts)
{
    if(active_set_near_singular_hessian(p.P))
        return;
    judge(p.P, v, p.A, result, argmin::oracle::qp_tolerances_of(opts), false);
}

void seed_and_resolve(field_reader& fields, solver_type& solver, const qp_problem& p, const options_type& opts, qp_capacity capacity, Eigen::VectorXd x0)
{
    const qp_sequence sequence = decode_sequence(fields);
    if(sequence.seed_after)
    {
        const qp_seed seed = decode_seed(fields, capacity);
        const error_set faults = seed_faults(seed, capacity);
        check_seed(solver.warm_start(seed.x, seed.y), faults);
        x0 = faults.mandatory() ? x0 : seed.x;
    }
    if(!sequence.resolve)
        return;
    const qp_vectors v = decode_resolve(fields, p);
    if(unsatisfiable_infinite_row(v.l, v.u))
        return;
    error_set justified;
    hessian_faults(p, justified);
    vector_faults(v.q, v.l, v.u, justified);
    contract_faults(p, v, x0, opts, justified);
    const auto outcome = solver.resolve(v.q, v.l, v.u, opts);
    if(admitted(outcome, justified))
        judge_active_set(p, v, *outcome, opts);
}

}

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size)
{
    field_reader fields{std::span<const std::uint8_t>(data, size)};
    const qp_problem p = decode_problem(fields, 3);
    const options_type opts = active_set_options<options_type>(fields);
    const qp_capacity capacity = decode_capacity(fields, p);
    if(empty_dense_problem(static_cast<std::int32_t>(p.q.size()), capacity.n) || unsatisfiable_infinite_row(p.l, p.u))
        return 0;
    solver_type solver(capacity.n, capacity.m);
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(capacity.n);
    if((fields.byte() & 1) != 0)
    {
        x0 = padded(p.x_f, capacity.n);
        check_seed(solver.warm_start(x0, Eigen::VectorXd()), error_set());
    }
    const qp_vectors v{p.q, p.l, p.u};
    error_set justified = solve_faults(p, active_set_options_valid(opts), capacity);
    if(!justified.mandatory())
        contract_faults(p, v, x0, opts, justified);
    const auto outcome = solver.solve(p.P, p.q, p.A, p.l, p.u, opts);
    if(!admitted(outcome, justified))
        return 0;
    judge_active_set(p, v, *outcome, opts);
    seed_and_resolve(fields, solver, p, opts, capacity, outcome->x);
    return 0;
}
