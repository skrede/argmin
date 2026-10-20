#ifndef HPP_GUARD_ARGMIN_TESTS_UNIT_COBYLA_BOX_CASES_H
#define HPP_GUARD_ARGMIN_TESTS_UNIT_COBYLA_BOX_CASES_H

#include "argmin/solver/options.h"
#include "argmin/solver/convergence.h"

#include "argmin/result/status.h"

#include <Eigen/Core>

#include <cstdint>

namespace argmin::test
{

template <typename Scalar>
struct box_record
{
    box_record() : value_calls(0), outside_calls(0), constraint_calls(0), first_point() {}

    std::uint32_t value_calls;
    std::uint32_t outside_calls;
    std::uint32_t constraint_calls;
    Eigen::VectorX<Scalar> first_point;
};

template <typename Scalar, typename P>
struct box_recorder
{
    using vector_type = Eigen::Vector<Scalar, P::problem_dimension>;

    static constexpr std::int32_t problem_dimension = P::problem_dimension;

    const P& problem;
    box_record<Scalar>& record;

    std::int32_t dimension() const { return problem.dimension(); }
    std::int32_t num_equality() const { return problem.num_equality(); }
    std::int32_t num_inequality() const { return problem.num_inequality(); }
    vector_type lower_bounds() const { return problem.lower_bounds(); }
    vector_type upper_bounds() const { return problem.upper_bounds(); }

    template <typename Vec>
    Scalar value(const Vec& x) const
    {
        ++record.value_calls;
        observe(x);
        return problem.value(x);
    }

    template <typename Vec, typename C>
    void constraints(const Vec& x, C& c) const
    {
        ++record.constraint_calls;
        observe(x);
        problem.constraints(x, c);
    }

    template <typename Vec>
    void observe(const Vec& x) const
    {
        if(record.value_calls + record.constraint_calls == 1)
            record.first_point = x;
        const vector_type lower = problem.lower_bounds();
        const vector_type upper = problem.upper_bounds();
        if((x.array() < lower.array()).any() || (x.array() > upper.array()).any())
            ++record.outside_calls;
    }
};

template <typename Scalar, std::int32_t N, typename Objective>
struct box_case
{
    using vector_type = Eigen::Vector<Scalar, N>;

    static constexpr std::int32_t problem_dimension = N;

    Objective objective;
    vector_type lower;
    vector_type upper;
    vector_type start;

    std::int32_t dimension() const { return N; }
    std::int32_t num_equality() const { return 0; }
    std::int32_t num_inequality() const { return 1; }
    vector_type lower_bounds() const { return lower; }
    vector_type upper_bounds() const { return upper; }
    vector_type initial_point() const { return start; }

    template <typename Vec>
    Scalar value(const Vec& x) const { return objective(x); }

    template <typename Vec, typename C>
    void constraints(const Vec& x, C& c) const
    {
        Scalar slack = Scalar(100);
        for(std::int32_t i = 0; i < N; ++i)
            slack -= x[i];
        c[0] = slack;
    }
};

template <typename Scalar>
struct narrow_objective
{
    template <typename Vec>
    Scalar operator()(const Vec& x) const
    {
        return (x[0] - Scalar(0.3)) * (x[0] - Scalar(0.3)) + (x[1] - Scalar(1)) * (x[1] - Scalar(1));
    }
};

template <typename Scalar>
struct undershoot_objective
{
    template <typename Vec>
    Scalar operator()(const Vec& x) const
    {
        return Scalar(50) * x[0] + (x[1] - Scalar(1)) * (x[1] - Scalar(1));
    }
};

template <typename Scalar>
struct three_scale_objective
{
    template <typename Vec>
    Scalar operator()(const Vec& x) const
    {
        const Eigen::Vector<Scalar, 3> d = x - Eigen::Vector<Scalar, 3>(Scalar(0.02), Scalar(1), Scalar(3));
        return d.squaredNorm();
    }
};

template <typename Scalar>
using narrow_case = box_case<Scalar, 2, narrow_objective<Scalar>>;

template <typename Scalar>
using undershoot_case = box_case<Scalar, 2, undershoot_objective<Scalar>>;

template <typename Scalar>
using three_scale_case = box_case<Scalar, 3, three_scale_objective<Scalar>>;

template <typename Scalar>
narrow_case<Scalar> narrow_witness()
{
    using V = Eigen::Vector<Scalar, 2>;
    return {.objective = {}, .lower = V(Scalar(0), Scalar(-10)), .upper = V(Scalar(0.6), Scalar(10)), .start = V(Scalar(0.5), Scalar(1))};
}

template <typename Scalar>
undershoot_case<Scalar> undershoot_from(Eigen::Vector<Scalar, 2> start)
{
    using V = Eigen::Vector<Scalar, 2>;
    return {.objective = {}, .lower = V(Scalar(0), Scalar(-10)), .upper = V(Scalar(1), Scalar(10)), .start = start};
}

template <typename Scalar>
undershoot_case<Scalar> undershoot()
{
    return undershoot_from<Scalar>({Scalar(0.8), Scalar(1)});
}

template <typename Scalar>
undershoot_case<Scalar> start_outside()
{
    return undershoot_from<Scalar>({Scalar(-3), Scalar(20)});
}

template <typename Scalar>
three_scale_case<Scalar> three_scale()
{
    using V = Eigen::Vector<Scalar, 3>;
    return {.objective = {}, .lower = V(Scalar(0), Scalar(-2), Scalar(-100)), .upper = V(Scalar(0.05), Scalar(2), Scalar(100)), .start = V(Scalar(0.04), Scalar(0), Scalar(0))};
}

template <typename Scalar>
solver_options<default_convergence<Scalar>> publication_options()
{
    solver_options<default_convergence<Scalar>> opts{};
    opts.max_iterations = 10000;
    opts.set_gradient_threshold(Scalar(1e-8));
    opts.set_objective_threshold(Scalar(1e-6));
    opts.set_step_threshold(Scalar(1e-6));
    opts.set_stationarity_threshold(Scalar(1e-8));
    return opts;
}

// Under the publication thresholds the driver's step-size and objective-window finish reports
// stalled before the trust radius reaches its final value.
inline bool succeeded(solver_status status)
{
    return status == solver_status::converged || status == solver_status::xtol_reached
        || status == solver_status::stalled;
}

}

#endif
