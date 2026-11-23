#ifndef HPP_GUARD_ARGMIN_TESTS_SUPPORT_ORACLE_INJECTION_SCHEDULE_H
#define HPP_GUARD_ARGMIN_TESTS_SUPPORT_ORACLE_INJECTION_SCHEDULE_H

#include <Eigen/Core>

#include <limits>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <algorithm>

namespace argmin::oracle
{

enum class callback : std::uint8_t
{
    value,
    gradient,
    residuals,
    jacobian,
    constraints,
    constraint_jacobian
};

enum class non_finite : std::uint8_t
{
    nan,
    positive_infinity,
    negative_infinity
};

enum class injection_mode : std::uint8_t
{
    first_call,
    kth_call,
    from_kth_call
};

struct injection
{
    callback target;
    non_finite kind;
    injection_mode mode;
    std::uint32_t k;
};

// Replaces what one named callback returns at the calls the injection selects. Every point is kept at which the callback
// returned an injected value, and every point at which it returned its own finite one, since a point evaluated both ways
// is a point the solver also saw finite.
class injection_schedule
{
public:
    injection_schedule()
        : m_calls{0}
        , m_planned{}
        , m_finite_at{}
        , m_injected_at{}
    {}

    explicit injection_schedule(const injection& planned)
        : m_calls{0}
        , m_planned{planned}
        , m_finite_at{}
        , m_injected_at{}
    {}

    template <typename X, typename Out>
    void apply(callback target, const X& x, Out& out)
    {
        if(!m_planned || m_planned->target != target)
            return;
        if(!selects_next_call() || out.size() == 0)
        {
            m_finite_at.emplace_back(x.template cast<double>());
            return;
        }
        out.setConstant(static_cast<typename Out::Scalar>(injected_value()));
        m_injected_at.emplace_back(x.template cast<double>());
    }

    template <typename X, typename Scalar>
    Scalar apply_value(callback target, const X& x, Scalar value)
    {
        Eigen::Matrix<Scalar, 1, 1> out(value);
        apply(target, x, out);
        return out[0];
    }

    bool injects_first_call() const { return m_planned && (m_planned->mode == injection_mode::first_call || m_planned->k <= 1); }

    std::size_t injected_count() const { return m_injected_at.size(); }

    // A point at which the callback returned an injected value and never its own finite one.
    bool only_injected_at(const Eigen::VectorXd& x) const { return contains(m_injected_at, x) && !contains(m_finite_at, x); }

private:
    std::uint32_t m_calls;
    std::optional<injection> m_planned;
    std::vector<Eigen::VectorXd> m_finite_at;
    std::vector<Eigen::VectorXd> m_injected_at;

    bool selects_next_call()
    {
        ++m_calls;
        if(m_planned->mode == injection_mode::from_kth_call)
            return m_calls >= m_planned->k;
        return m_calls == (m_planned->mode == injection_mode::first_call ? 1u : m_planned->k);
    }

    double injected_value() const
    {
        if(m_planned->kind == non_finite::nan)
            return std::numeric_limits<double>::quiet_NaN();
        const double inf = std::numeric_limits<double>::infinity();
        return m_planned->kind == non_finite::positive_infinity ? inf : -inf;
    }

    static bool contains(const std::vector<Eigen::VectorXd>& points, const Eigen::VectorXd& x)
    {
        const auto same_bits = [&](const Eigen::VectorXd& p) { return p.size() == x.size() && std::memcmp(p.data(), x.data(), sizeof(double) * static_cast<std::size_t>(x.size())) == 0; };
        return std::ranges::any_of(points, same_bits);
    }
};

}

#endif
