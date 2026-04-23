#ifndef HPP_GUARD_NABLAPP_BENCHMARKS_COUNTING_PROBLEM_H
#define HPP_GUARD_NABLAPP_BENCHMARKS_COUNTING_PROBLEM_H

// Bench-side template wrapper for any nablapp test-function problem struct.
//
// Routes every problem entry point (value, gradient, constraints,
// constraint_jacobian, residuals, jacobian, lower_bounds, upper_bounds,
// dimension, num_equality, num_inequality, num_residuals, optimal_value,
// initial_point) through a shared four-counter eval tracker. Concept
// pass-through preserves all compile-time traits and member signatures
// required by nablapp's differentiable / bound_constrained / constrained
// / constrained_values / least_squares concepts, so any wrapped
// counting_problem<P> still satisfies the same concepts as the inner P.
//
// Counter semantics:
//   counts.f -- bumped on each value(x) call
//   counts.g -- bumped on each gradient(x, g) call
//   counts.c -- bumped on each constraints(x, c) call
//   counts.J -- bumped on each constraint_jacobian(x, J) call
//
// Least-squares residuals/jacobian intentionally do NOT bump counters;
// least-squares solvers call them from within their own evaluation and
// the bench schema decomposes f/g/c/J only on the NLP surface.
//
// Lifetime contract: the wrapped Problem& and eval_counts& must outlive
// the wrapper. Standard C++ lifetime idiom; the wrapper does not own
// either object.

#include "nablapp/test_functions/problem_class.h"
#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>

namespace nablapp::bench
{

// Four-counter eval tracker. reset() zeroes every counter for the next
// per-run snapshot.
struct eval_counts
{
    int f{0};
    int g{0};
    int c{0};
    int J{0};

    void reset() { f = g = c = J = 0; }
};

// Template wrapper around any nablapp problem struct. Forwards every
// concept-required member to the inner problem while bumping the shared
// eval counters. Compile-time traits (problem_dimension, pclass) are
// re-exposed verbatim so concept satisfaction transfers from Problem to
// counting_problem<Problem>.
template <typename Problem>
struct counting_problem
{
    static constexpr int problem_dimension = Problem::problem_dimension;
    static constexpr ::nablapp::problem_class pclass = Problem::pclass;

    const Problem* inner{nullptr};
    eval_counts* counts{nullptr};

    counting_problem() = default;
    counting_problem(const Problem& p, eval_counts& c) : inner{&p}, counts{&c} {}

    [[nodiscard]] int dimension() const { return inner->dimension(); }

    template <typename Vec>
    [[nodiscard]] auto value(const Vec& x) const
    {
        ++counts->f;
        return inner->value(x);
    }

    template <typename Vec, typename G>
    void gradient(const Vec& x, G& g) const
    {
        ++counts->g;
        inner->gradient(x, g);
    }

    template <typename Vec, typename C>
    void constraints(const Vec& x, C& c) const
        requires requires(const Problem& p, const Vec& v, C& cc) { p.constraints(v, cc); }
    {
        ++counts->c;
        inner->constraints(x, c);
    }

    [[nodiscard]] int num_equality() const
        requires requires(const Problem& p) { p.num_equality(); }
    {
        return inner->num_equality();
    }

    [[nodiscard]] int num_inequality() const
        requires requires(const Problem& p) { p.num_inequality(); }
    {
        return inner->num_inequality();
    }

    template <typename Vec, typename Jac>
    void constraint_jacobian(const Vec& x, Jac& J) const
        requires requires(const Problem& p, const Vec& v, Jac& JJ) { p.constraint_jacobian(v, JJ); }
    {
        ++counts->J;
        inner->constraint_jacobian(x, J);
    }

    [[nodiscard]] auto lower_bounds() const
        requires requires(const Problem& p) { p.lower_bounds(); }
    {
        return inner->lower_bounds();
    }

    [[nodiscard]] auto upper_bounds() const
        requires requires(const Problem& p) { p.upper_bounds(); }
    {
        return inner->upper_bounds();
    }

    template <typename Vec, typename R>
    void residuals(const Vec& x, R& r) const
        requires requires(const Problem& p, const Vec& v, R& rr) { p.residuals(v, rr); }
    {
        inner->residuals(x, r);
    }

    template <typename Vec, typename Jac>
    void jacobian(const Vec& x, Jac& J) const
        requires requires(const Problem& p, const Vec& v, Jac& JJ) { p.jacobian(v, JJ); }
    {
        inner->jacobian(x, J);
    }

    [[nodiscard]] int num_residuals() const
        requires requires(const Problem& p) { p.num_residuals(); }
    {
        return inner->num_residuals();
    }

    [[nodiscard]] auto optimal_value() const { return inner->optimal_value(); }
    [[nodiscard]] auto initial_point() const { return inner->initial_point(); }
};

}

#endif
