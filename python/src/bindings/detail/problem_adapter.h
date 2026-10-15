#ifndef HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_PROBLEM_ADAPTER_H
#define HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_PROBLEM_ADAPTER_H

#include "bindings/detail/callback_boundary.h"

#include "argmin/types.h"
#include "argmin/derivative/finite_difference.h"

namespace argmin::python
{

// Value only: the gradient is supplied by the library's own central-difference
// helper, so the tier is differentiable without a caller-supplied derivative.
class value_problem : public callback_boundary
{
public:
    static constexpr int problem_dimension = dynamic_dimension;

    using callback_boundary::callback_boundary;

    double value(const vector<double>& x) const { return invoke_value(x); }

    void gradient(const vector<double>& x, vector<double>& g) const
    {
        fd_gradient(*this, x, g);
    }
};

class gradient_problem : public callback_boundary
{
public:
    static constexpr int problem_dimension = dynamic_dimension;

    using callback_boundary::callback_boundary;

    double value(const vector<double>& x) const { return invoke_value(x); }

    void gradient(const vector<double>& x, vector<double>& g) const { invoke_gradient(x, g); }
};

class bounded_problem : public callback_boundary
{
public:
    static constexpr int problem_dimension = dynamic_dimension;

    using callback_boundary::callback_boundary;

    double value(const vector<double>& x) const { return invoke_value(x); }

    void gradient(const vector<double>& x, vector<double>& g) const
    {
        if(has_gradient())
            invoke_gradient(x, g);
        else
            fd_gradient(*this, x, g);
    }

    vector<double> lower_bounds() const
    {
        return invoke_bound(callbacks().lower_bounds, lower_is_callable(), lower_cache(), "lower");
    }

    vector<double> upper_bounds() const
    {
        return invoke_bound(callbacks().upper_bounds, upper_is_callable(), upper_cache(), "upper");
    }
};

class constrained_values_problem : public callback_boundary
{
public:
    static constexpr int problem_dimension = dynamic_dimension;

    using callback_boundary::callback_boundary;

    double value(const vector<double>& x) const { return invoke_value(x); }

    void gradient(const vector<double>& x, vector<double>& g) const
    {
        if(has_gradient())
            invoke_gradient(x, g);
        else
            fd_gradient(*this, x, g);
    }

    void constraints(const vector<double>& x, vector<double>& c) const
    {
        invoke_constraints(x, c);
    }

    int num_equality() const { return callbacks().num_equality; }
    int num_inequality() const { return callbacks().num_inequality; }

    // The derivative-free constrained policy paired with this tier constrains
    // its problem parameter on bound_constrained as well, so the tier carries
    // bounds even though nothing about a constraint-value formulation requires
    // them; a caller who supplies none gets the infinite fill the policy
    // already treats as unbounded.
    vector<double> lower_bounds() const
    {
        return invoke_bound(callbacks().lower_bounds, lower_is_callable(), lower_cache(), "lower");
    }

    vector<double> upper_bounds() const
    {
        return invoke_bound(callbacks().upper_bounds, upper_is_callable(), upper_cache(), "upper");
    }
};

class constrained_problem : public callback_boundary
{
public:
    static constexpr int problem_dimension = dynamic_dimension;

    using callback_boundary::callback_boundary;

    double value(const vector<double>& x) const { return invoke_value(x); }

    void gradient(const vector<double>& x, vector<double>& g) const
    {
        if(has_gradient())
            invoke_gradient(x, g);
        else
            fd_gradient(*this, x, g);
    }

    void constraints(const vector<double>& x, vector<double>& c) const
    {
        invoke_constraints(x, c);
    }

    int num_equality() const { return callbacks().num_equality; }
    int num_inequality() const { return callbacks().num_inequality; }

    void constraint_jacobian(const vector<double>& x, matrix<double>& J) const
    {
        invoke_jacobian(x, J);
    }
};

// Each adapter must satisfy exactly the tier its policy needs and no more.
// Concept satisfaction here is duck-typed, so an adapter that grows a member
// silently starts advertising a capability the caller never supplied -- an
// unconstrained problem claiming constraints, say. The negative half is the
// load-bearing one; the positive half only records intent.
static_assert(argmin::differentiable<value_problem>);
static_assert(!argmin::bound_constrained<value_problem>);

static_assert(argmin::differentiable<gradient_problem>);
static_assert(!argmin::bound_constrained<gradient_problem>);
static_assert(!argmin::constrained_values<gradient_problem>);

static_assert(argmin::bound_constrained<bounded_problem>);
static_assert(!argmin::constrained_values<bounded_problem>);

static_assert(argmin::constrained_values<constrained_values_problem>);
static_assert(!argmin::constrained<constrained_values_problem>);

static_assert(argmin::constrained<constrained_problem>);

}

#endif
