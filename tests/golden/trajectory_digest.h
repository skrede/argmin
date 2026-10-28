#ifndef HPP_GUARD_ARGMIN_TESTS_GOLDEN_TRAJECTORY_DIGEST_H
#define HPP_GUARD_ARGMIN_TESTS_GOLDEN_TRAJECTORY_DIGEST_H

#include <Eigen/Core>

#include <bit>
#include <string>
#include <cstdint>
#include <iomanip>
#include <sstream>

namespace argmin::golden
{

inline std::string to_hex(std::uint64_t value)
{
    std::ostringstream out;
    out << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

inline std::string to_hex(double value)
{
    return to_hex(std::bit_cast<std::uint64_t>(value));
}

// FNV-1a-64 (Fowler, Noll and Vo) over the bytes of each IEEE-754 bit pattern, taken low byte first so the
// digest does not depend on the host byte order. A signed zero or a NaN payload changes it.
class trajectory_digest
{
public:
    trajectory_digest() : hash_(offset_basis) {}

    void fold(double value)
    {
        const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
        for(std::int32_t byte = 0; byte < 8; ++byte)
        {
            hash_ ^= (bits >> (8 * byte)) & 0xffu;
            hash_ *= prime;
        }
    }

    template <typename Derived>
    void fold_all(const Eigen::DenseBase<Derived>& values)
    {
        for(Eigen::Index i = 0; i < values.size(); ++i)
            fold(static_cast<double>(values.derived().coeff(i)));
    }

    std::uint64_t value() const { return hash_; }

private:
    static constexpr std::uint64_t offset_basis = 14695981039346656037u;
    static constexpr std::uint64_t prime = 1099511628211u;

    std::uint64_t hash_;
};

// Only vector multipliers are folded: the scalar lambda of the least-squares states is a damping factor
// and the integer lambda of ISRES is a population size, and neither has an element access.
template <typename State>
void fold_multipliers(trajectory_digest& digest, const State& state)
{
    if constexpr(requires {
                     state.lambda.size();
                     state.lambda[0];
                 })
        digest.fold_all(state.lambda);
    if constexpr(requires { state.y_dual.size(); })
        digest.fold_all(state.y_dual);
    if constexpr(requires {
                     state.lambda_eq.size();
                     state.lambda_ineq.size();
                 })
    {
        digest.fold_all(state.lambda_eq);
        digest.fold_all(state.lambda_ineq);
    }
}

template <typename Solver, typename StepResult>
std::uint64_t step_digest(const Solver& solver, const StepResult& step)
{
    trajectory_digest digest;
    digest.fold_all(solver.state().x);
    digest.fold(static_cast<double>(step.objective_value));
    fold_multipliers(digest, solver.state());
    digest.fold(static_cast<double>(step.step_size));
    digest.fold(static_cast<double>(step.constraint_violation));
    return digest.value();
}

}

#endif
