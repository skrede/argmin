#include "harness.h"
#include "qp_check.h"
#include "qp_input.h"
#include "qp_options.h"

#include "oracle_tolerances.h"

#include "argmin/qp/sparse_admm_qp.h"

#include "argmin/options/sparse_qp_options.h"

#include <Eigen/SparseCore>

#include <span>
#include <vector>
#include <cstddef>
#include <cstdint>

namespace
{

using namespace argmin::fuzzing;

using options_type = argmin::sparse_qp_options<double>;
using solver_type = argmin::sparse_admm_qp_solver<double>;
using sparse_type = solver_type::sparse_type;

// Every entry that is not exactly zero is stored, a NaN included, so the conversion drops no fault.
sparse_type to_sparse(const Eigen::MatrixXd& dense)
{
    std::vector<Eigen::Triplet<double>> entries;
    for(Eigen::Index j = 0; j < dense.cols(); ++j)
        for(Eigen::Index i = 0; i < dense.rows(); ++i)
            if(dense(i, j) != 0.0)
                entries.emplace_back(static_cast<std::int32_t>(i), static_cast<std::int32_t>(j), dense(i, j));
    sparse_type out(dense.rows(), dense.cols());
    out.setFromTriplets(entries.begin(), entries.end());
    return out;
}

void seed_and_resolve(field_reader& fields, solver_type& solver, const qp_problem& p, const options_type& opts)
{
    const qp_sequence sequence = decode_sequence(fields);
    if(sequence.seed_after)
    {
        const qp_seed seed = decode_seed(fields, {static_cast<std::int32_t>(p.q.size()), static_cast<std::int32_t>(p.A.rows())});
        if(!sparse_admm_non_finite_seed(seed.x, seed.y))
            solver.warm_start(seed.x, seed.y);
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
        judge(to_sparse(p.P), v, to_sparse(p.A), *outcome, argmin::oracle::qp_tolerances_of(opts), true);
}

}

// Positive semidefiniteness is the sparse solver's documented precondition, which it does not verify, so the target
// draws only the two positive semidefinite Hessian shapes.
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size)
{
    field_reader fields{std::span<const std::uint8_t>(data, size)};
    const qp_problem p = decode_problem(fields, 2);
    const options_type opts = admm_options<options_type>(fields, p.shape);
    if(unsatisfiable_infinite_row(p.l, p.u))
        return 0;
    solver_type solver;
    if((fields.byte() & 1) != 0)
        solver.warm_start(p.x_f, Eigen::VectorXd());
    const bool valid = admm_options_valid(opts);
    error_set justified = solve_faults(p, valid, {static_cast<std::int32_t>(p.q.size()), static_cast<std::int32_t>(p.A.rows())});
    if(valid && p.q.size() == 0)
        justified.demand(argmin::qp_error::dimension_mismatch);
    const auto outcome = solver.solve(to_sparse(p.P), p.q, to_sparse(p.A), p.l, p.u, opts);
    if(!admitted(outcome, justified))
        return 0;
    judge(to_sparse(p.P), qp_vectors{p.q, p.l, p.u}, to_sparse(p.A), *outcome, argmin::oracle::qp_tolerances_of(opts), true);
    seed_and_resolve(fields, solver, p, opts);
    return 0;
}
