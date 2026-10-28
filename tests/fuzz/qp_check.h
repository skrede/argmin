#ifndef HPP_GUARD_ARGMIN_TESTS_FUZZ_QP_CHECK_H
#define HPP_GUARD_ARGMIN_TESTS_FUZZ_QP_CHECK_H

#include "harness.h"
#include "qp_input.h"

#include "oracle/qp_oracle.h"

#include "argmin/qp/qp_types.h"

#include <Eigen/Core>

#include <cstdio>
#include <string>
#include <cstdint>
#include <optional>
#include <algorithm>

namespace argmin::fuzzing
{

struct qp_capacity
{
    std::int32_t n;
    std::int32_t m;
};

struct qp_sequence
{
    bool seed_after;
    bool resolve;
};

struct qp_seed
{
    Eigen::VectorXd x;
    Eigen::VectorXd y;
};

struct qp_vectors
{
    Eigen::VectorXd q;
    Eigen::VectorXd l;
    Eigen::VectorXd u;
};

inline qp_capacity decode_capacity(field_reader& fields, const qp_problem& p)
{
    const std::uint8_t shape = fields.byte();
    const auto n = static_cast<std::int32_t>(p.q.size());
    const auto m = static_cast<std::int32_t>(p.A.rows());
    if(shape % 4 != 3)
        return {n, m + shape / 4 % 2};
    return {std::max(0, n + shape / 4 % 3 - 1), std::max(0, m - shape / 16 % 2)};
}

inline qp_sequence decode_sequence(field_reader& fields)
{
    const std::uint8_t flags = fields.byte();
    return {(flags & 1) != 0, (flags & 2) != 0};
}

inline qp_seed decode_seed(field_reader& fields, qp_capacity capacity)
{
    const std::uint8_t shape = fields.byte();
    const std::int32_t x_size = shape % 4 == 3 ? std::max(0, capacity.n - 1) : capacity.n;
    const std::int32_t y_size = shape / 4 % 4 == 3 ? capacity.m + 1 : (shape / 4 % 4 == 2 ? 0 : capacity.m);
    qp_seed seed{bounded_matrix(fields, x_size, 1), bounded_matrix(fields, y_size, 1)};
    const std::uint8_t faults = fields.byte();
    inject(fields, (faults & 1) != 0, seed.x);
    inject(fields, (faults & 2) != 0, seed.y);
    return seed;
}

inline qp_vectors decode_resolve(field_reader& fields, const qp_problem& p)
{
    qp_vectors v{bounded_matrix(fields, static_cast<std::int32_t>(p.q.size()), 1), {}, {}};
    const Eigen::VectorXd x_f = bounded_matrix(fields, static_cast<std::int32_t>(p.q.size()), 1);
    decode_bounds(fields, p.A * x_f, v.l, v.u);
    inject_vector_faults(fields, v.q, v.l, v.u);
    return v;
}

inline Eigen::VectorXd padded(const Eigen::VectorXd& x, std::int32_t size)
{
    Eigen::VectorXd out = Eigen::VectorXd::Zero(size);
    const Eigen::Index shared = std::min<Eigen::Index>(size, x.size());
    out.head(shared) = x.head(shared);
    return out;
}

inline error_set seed_faults(const qp_seed& seed, qp_capacity capacity)
{
    error_set justified;
    if(seed.x.size() < capacity.n || seed.y.size() > capacity.m)
        justified.demand(qp_error::dimension_mismatch);
    else if(!seed.x.allFinite() || !seed.y.allFinite())
        justified.demand(qp_error::non_finite_input);
    return justified;
}

inline error_set solve_faults(const qp_problem& p, bool options_valid, qp_capacity capacity)
{
    error_set justified;
    if(!options_valid)
        justified.demand(qp_error::invalid_options);
    else if(capacity.n != p.q.size() || p.A.rows() > capacity.m)
        justified.demand(qp_error::capacity_exceeded);
    else
    {
        hessian_faults(p, justified);
        vector_faults(p.q, p.l, p.u, justified);
    }
    return justified;
}

inline void check_seed(const std::optional<qp_error>& error, const error_set& justified)
{
    if(error)
        require(justified.admits(*error), "a refused warm start is one its seed's structure justifies");
    else
        require(!justified.mandatory(), "a malformed warm start is refused");
}

template <typename Outcome>
bool admitted(const Outcome& outcome, const error_set& justified)
{
    if(outcome.has_value())
    {
        require(!justified.mandatory(), "an input with a structural fault is rejected");
        return true;
    }
    if(!justified.admits(outcome.error()))
        std::fprintf(stderr, "returned qp_error %d\n", static_cast<std::int32_t>(outcome.error()));
    require(justified.admits(outcome.error()), "a returned error is one the input's structure justifies");
    return false;
}

template <typename Result>
bool known_admm_failure(const Result& result, const std::string& failing)
{
    if(failing.rfind("certificate", 0) == 0)
        return admm_infeasible_without_certificate(result.status);
    return failing == "dual_sign_breach" && polished_dual_sign_unchecked(result.polished);
}

template <typename PMatrix, typename AMatrix, typename Result>
void judge(const PMatrix& P, const qp_vectors& v, const AMatrix& A, const Result& result, const oracle::qp_tolerances<double>& tol, bool admm)
{
    observe(result.objective_value);
    require(result.x.size() == v.q.size() && result.x.allFinite() != planted_violation, "a returned x is finite and of the problem's length");
    const oracle::qp_verdict verdict = oracle::judge_qp(P, v.q, A, v.l, v.u, result, tol);
    if(verdict.ok || (admm && known_admm_failure(result, verdict.failing)))
        return;
    std::fprintf(stderr, "status %d, failing leg %s\n", static_cast<std::int32_t>(result.status), verdict.failing.c_str());
    require(false, "a returned result meets the claim its status makes");
}

}

#endif
