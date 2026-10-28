#ifndef HPP_GUARD_ARGMIN_TESTS_FUZZ_HARNESS_H
#define HPP_GUARD_ARGMIN_TESTS_FUZZ_HARNESS_H

#include "argmin/qp/qp_types.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <span>
#include <cstdio>
#include <limits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace argmin::fuzzing
{

// A field past the end of the input reads as zero, so a short input decodes as its leading fields followed by
// zeros and the empty input is the all-zero decode.
class field_reader
{
public:
    explicit field_reader(std::span<const std::uint8_t> input) : m_input(input), m_offset(0) {}

    double scalar()
    {
        double value = 0.0;
        if(m_offset + sizeof(value) <= m_input.size())
            std::memcpy(&value, m_input.data() + m_offset, sizeof(value));
        m_offset += sizeof(value);
        return value;
    }

    std::uint8_t byte()
    {
        const std::uint8_t value = m_offset < m_input.size() ? m_input[m_offset] : std::uint8_t(0);
        ++m_offset;
        return value;
    }

private:
    std::span<const std::uint8_t> m_input;
    std::size_t m_offset;
};

// abort() rather than assert() so the check survives NDEBUG and libFuzzer keeps the offending input as an
// artifact instead of printing a line nobody reads.
inline void require(bool condition, const char* contract)
{
    if(condition)
        return;
    std::fprintf(stderr, "argmin fuzz oracle violated: %s\n", contract);
    std::abort();
}

inline volatile double observation_sink = 0.0;

// Without a consumer the solve behind a result is dead code the optimizer may delete, and the sanitizers then
// have nothing to inspect.
inline void observe(double value)
{
    observation_sink = value;
}

// The planted binary inverts one oracle leg, so a run of it proves that a violation turns the job red.
#ifdef ARGMIN_FUZZ_PLANT_VIOLATION
inline constexpr bool planted_violation = true;
#else
inline constexpr bool planted_violation = false;
#endif

// Known live defects. Each predicate names one input class the targets do not send, or one oracle leg they do
// not hold, until the owning fix lands; the fix deletes the predicate.

// The dense ADMM and dense active-set solvers have no n = 0 guard and reduce an empty matrix, which is undefined
// behavior.
inline bool empty_dense_problem(std::int32_t n, std::int32_t capacity)
{
    return n == 0 && capacity == 0;
}

// The QP solvers accept a row whose bounds are both +inf or both -inf, which no point satisfies: the ADMM solvers
// return a non-finite x under a dual_infeasible claim and the active-set solver drops the row and claims solved.
inline bool unsatisfiable_infinite_row(const Eigen::VectorXd& l, const Eigen::VectorXd& u)
{
    constexpr double inf = std::numeric_limits<double>::infinity();
    return (l.array() == inf).any() || (u.array() == -inf).any();
}

// The dense active-set solver accepts a P that is singular to round-off without detecting the singular reduced
// Hessian, and can then return a non-finite x under max_iterations or a solved claim at an infeasible point.
inline bool active_set_near_singular_hessian(const Eigen::MatrixXd& P)
{
    if(P.size() == 0 || !P.allFinite())
        return false;
    const Eigen::VectorXd eigenvalues = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(P, Eigen::EigenvaluesOnly).eigenvalues();
    return eigenvalues.minCoeff() <= 1e-12 * std::max(1.0, eigenvalues.maxCoeff());
}

// The ADMM polish is accepted on residuals and objective alone and never checks the signs of its dual, so a polished
// result can carry a wrong-signed multiplier on a one-sided row it wrongly took as active or shared with a dependent row.
inline bool polished_dual_sign_unchecked(bool polished)
{
    return polished;
}

// The sparse ADMM warm_start has no error channel and accepts a non-finite seed, after which a solve can claim
// solved with a non-finite x.
inline bool sparse_admm_non_finite_seed(const Eigen::VectorXd& x, const Eigen::VectorXd& y)
{
    return !x.allFinite() || !y.allFinite();
}

// The ADMM solvers return the last iterates, not the infeasibility certificates their termination test accepted:
// y under primal_infeasible, and x under dual_infeasible.
inline bool admm_infeasible_without_certificate(qp_solve_status status)
{
    return status == qp_solve_status::primal_infeasible || status == qp_solve_status::dual_infeasible;
}

}

#endif
