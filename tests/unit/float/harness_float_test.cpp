#include "float_gate.h"

#include "argmin/solver/options.h"
#include "argmin/solver/convergence.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/test_functions/booth.h"

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"

#include "argmin/types.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using namespace argmin;
using namespace argmin::float_gate;

namespace
{

template <scalar S, int N = dynamic_dimension>
struct descent_policy
{
    using scalar_type = S;

    template <int M>
    using rebind = descent_policy<S, M>;

    template <typename P>
    struct state_type
    {
        vector<S, N> x;
        S objective;
        P problem;
    };

    template <typename P, typename Convergence = default_convergence<S>>
    state_type<P> init(const P& problem, const vector<S, N>& x0, const solver_options<Convergence>&)
    {
        return state_type<P>{.x = x0, .objective = problem.value(x0), .problem = problem};
    }

    template <typename P>
    step_result<S> step(state_type<P>& s)
    {
        vector<S, N> g = s.x;
        s.problem.gradient(s.x, g);
        const vector<S, N> dx = -S(0.1) * g;
        const S previous = s.objective;
        s.x += dx;
        s.objective = s.problem.value(s.x);
        const bool improved = previous > s.objective;
        return step_result<S>{.objective_value = s.objective, .gradient_norm = g.norm(), .step_size = dx.norm(),
                              .objective_change = s.objective - previous, .improved = improved,
                              .x_norm = s.x.norm()};
    }

    template <typename P>
    void reset(state_type<P>& s, const Eigen::VectorX<S>& x0)
    {
        s.x = x0;
        s.objective = s.problem.value(s.x);
    }

    template <typename P>
    void reset_clear(state_type<P>& s, const Eigen::VectorX<S>& x0)
    {
        reset(s, x0);
    }
};

constexpr std::uint32_t max_iterations = 500;

template <scalar S>
solve_result<S, 2> drive(solver_options<default_convergence<S>> opts)
{
    const booth<S> problem;
    opts.max_iterations = max_iterations;
    step_budget_solver solver{descent_policy<S>{}, problem, problem.initial_point(), opts};
    return solver.solve();
}

}

TEST_CASE("the harness drives a float policy to convergence with float criteria at their float defaults", "[float]")
{
    const auto result_f = drive<float>(solver_options<default_convergence<float>>{});
    const auto result_d = drive<double>(solver_options<>{});

    REQUIRE(result_d.status == solver_status::converged);
    REQUIRE(result_f.status == solver_status::converged);
    CHECK(tracks(result_f.x, result_d.x, k_harness));
}
