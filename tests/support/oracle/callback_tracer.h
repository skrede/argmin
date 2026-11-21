#ifndef HPP_GUARD_ARGMIN_TESTS_SUPPORT_ORACLE_CALLBACK_TRACER_H
#define HPP_GUARD_ARGMIN_TESTS_SUPPORT_ORACLE_CALLBACK_TRACER_H

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <cstdint>
#include <utility>
#include <algorithm>

namespace argmin::oracle
{

struct callback_counts
{
    std::uint32_t value;
    std::uint32_t gradient;
    std::uint32_t hessian;
    std::uint32_t residuals;
    std::uint32_t jacobian;
    std::uint32_t constraints;
    std::uint32_t constraint_jacobian;
};

// Over every argument a solver handed the problem: whether all were finite and of the problem's dimension, and the
// largest distance any coordinate lay outside the box.
struct argument_domain
{
    bool finite;
    bool sized;
    double box_excursion;
};

template <typename Problem>
struct traced_capacity
{
};

template <has_constraint_count Problem>
struct traced_capacity<Problem>
{
    static constexpr std::int32_t constraint_count = Problem::constraint_count;
};

// Forwards every callback of the problem it wraps, so it satisfies the same formulation concepts, and records each call.
template <typename Problem>
class traced : public traced_capacity<Problem>
{
public:
    static constexpr std::int32_t problem_dimension = Problem::problem_dimension;

    explicit traced(const Problem& problem)
        : m_problem(problem)
        , m_counts{}
        , m_domain{true, true, 0.0}
    {}

    std::int32_t dimension() const { return m_problem.dimension(); }

    template <typename X>
    decltype(std::declval<const Problem&>().value(std::declval<const X&>())) value(const X& x) const
    {
        record(x, m_counts.value);
        return m_problem.value(x);
    }

    template <typename X, typename Out>
        requires requires(const Problem& p, const X& x, Out&& out) { p.gradient(x, std::forward<Out>(out)); }
    void gradient(const X& x, Out&& g) const
    {
        record(x, m_counts.gradient);
        m_problem.gradient(x, std::forward<Out>(g));
    }

    template <typename X, typename Out>
        requires requires(const Problem& p, const X& x, Out&& out) { p.hessian(x, std::forward<Out>(out)); }
    void hessian(const X& x, Out&& H) const
    {
        record(x, m_counts.hessian);
        m_problem.hessian(x, std::forward<Out>(H));
    }

    template <typename X, typename Out>
        requires requires(const Problem& p, const X& x, Out&& out) { p.residuals(x, std::forward<Out>(out)); }
    void residuals(const X& x, Out&& r) const
    {
        record(x, m_counts.residuals);
        m_problem.residuals(x, std::forward<Out>(r));
    }

    template <typename X, typename Out>
        requires requires(const Problem& p, const X& x, Out&& out) { p.jacobian(x, std::forward<Out>(out)); }
    void jacobian(const X& x, Out&& J) const
    {
        record(x, m_counts.jacobian);
        m_problem.jacobian(x, std::forward<Out>(J));
    }

    template <typename X, typename Out>
        requires requires(const Problem& p, const X& x, Out&& out) { p.constraints(x, std::forward<Out>(out)); }
    void constraints(const X& x, Out&& c) const
    {
        record(x, m_counts.constraints);
        m_problem.constraints(x, std::forward<Out>(c));
    }

    template <typename X, typename Out>
        requires requires(const Problem& p, const X& x, Out&& out) { p.constraint_jacobian(x, std::forward<Out>(out)); }
    void constraint_jacobian(const X& x, Out&& J) const
    {
        record(x, m_counts.constraint_jacobian);
        m_problem.constraint_jacobian(x, std::forward<Out>(J));
    }

    template <typename P = Problem>
    decltype(std::declval<const P&>().num_equality()) num_equality() const { return m_problem.num_equality(); }

    template <typename P = Problem>
    decltype(std::declval<const P&>().num_inequality()) num_inequality() const { return m_problem.num_inequality(); }

    template <typename P = Problem>
    decltype(std::declval<const P&>().num_residuals()) num_residuals() const { return m_problem.num_residuals(); }

    template <typename P = Problem>
    decltype(std::declval<const P&>().lower_bounds()) lower_bounds() const { return m_problem.lower_bounds(); }

    template <typename P = Problem>
    decltype(std::declval<const P&>().upper_bounds()) upper_bounds() const { return m_problem.upper_bounds(); }

    template <typename P = Problem>
    decltype(std::declval<const P&>().initial_point()) initial_point() const { return m_problem.initial_point(); }

    const callback_counts& counts() const { return m_counts; }

    const argument_domain& domain() const { return m_domain; }

    // A least-squares objective is evaluated through either value or residuals.
    std::uint32_t objective_calls() const { return m_counts.value + m_counts.residuals; }

private:
    const Problem& m_problem;
    mutable callback_counts m_counts;
    mutable argument_domain m_domain;

    template <typename X>
    void record(const X& x, std::uint32_t& count) const
    {
        ++count;
        const bool finite = x.allFinite();
        const bool sized = x.size() == m_problem.dimension();
        m_domain.finite = m_domain.finite && finite;
        m_domain.sized = m_domain.sized && sized;
        if constexpr(requires { m_problem.lower_bounds(); m_problem.upper_bounds(); })
        {
            if(finite && sized)
            {
                const Eigen::VectorXd point = x.template cast<double>();
                const double below = (m_problem.lower_bounds().template cast<double>() - point).maxCoeff();
                const double above = (point - m_problem.upper_bounds().template cast<double>()).maxCoeff();
                m_domain.box_excursion = std::max({m_domain.box_excursion, below, above});
            }
        }
    }
};

}

#endif
