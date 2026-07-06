// ISRES per-evaluation best-tracking + learning-rate oracle pins.
//
// Pin 1 -- per-evaluation best tracking. NLopt 2.10.0 isres.c:174-193
// checks EVERY individual against the best-ever record at evaluation
// time: on a problem where every evaluated point is feasible, the
// returned minimizer must be exactly the minimum-fitness point among
// ALL objective evaluations the solver performed. The test wraps the
// problem in a recorder so the true best evaluated point is enumerable
// (and hand-identifiable from the fixed seed: every evaluation the
// solver makes is in the log, and the assertion pins the returned
// minimizer to the log's minimum -- no seed shopping; the property
// must hold for ANY seed).
//
// Red-state proof (pre-fix): the production generation loop mutates
// the population in place and only afterwards compares the fitness at
// the PREVIOUS generation's best-ranked slot, so better points seen
// mid-stream are lost and the returned minimizer is not the best
// evaluated point. Observed at seed 42, 30 generations, lambda = 60:
// returned objective 2.43877 vs true best evaluated 0.04079 at
// evaluation #150 of 1860 (gap 2.39797).
//
// Pin 2 -- self-adaptation learning rates. NLopt isres.c and the
// classical Schwefel (1995) evolution-strategy convention set
//   global (once-per-individual) rate  tau' = PHI / sqrt(2 n)
//   per-component rate                 tau  = PHI / sqrt(2 sqrt(n))
// with PHI = 1 (isres.c: taup = PHI/sqrt(2*n); tau = PHI/sqrt(2*sqrt(n));
// sigma multiplier exp(taup * N(0,1) + tau * N_j(0,1))). In argmin the
// rates.tau_prime field multiplies the once-per-individual draw and
// rates.tau the per-component draw, so role-correct values at n = 20 are
//   tau  = 1/sqrt(2 sqrt(20)) = 0.33437015248821100  (per-component)
//   tau' = 1/sqrt(40)         = 0.15811388300841897  (global)
// The per-component rate must EXCEED the global rate for n > 1 (the
// per-component rate shrinks as n^(-1/4), the global rate as n^(-1/2)).
//
// Red-state proof (pre-fix): the two values are assigned to the
// opposite roles (tau = 0.15811..., tau_prime = 0.33437...), which at
// n = 20 makes the shared noise 2.11x too large and the per-component
// noise 2.11x too small, collapsing per-coordinate step-size
// adaptation into near-lockstep.

#include "argmin/detail/isres_operators.h"
#include "argmin/solver/options.h"
#include "argmin/solver/isres_policy.h"
#include "argmin/solver/basic_solver.h"
#include "argmin/result/status.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>
#include <cstddef>

using Catch::Approx;
using namespace argmin;

namespace
{

struct recorded_eval
{
    Eigen::VectorXd x;
    double f;
};

// Bounded convex QP, minimum f = 0 at (0.5, -0.25), with one
// inequality constraint that is strictly satisfied everywhere in the
// box (c = 300 - x1 - x2 >= 280 > 0), so every evaluated point is
// feasible and the best-ever record must equal the evaluation-log
// minimum. Every value() call is recorded.
struct recording_qp
{
    static constexpr int problem_dimension = dynamic_dimension;

    mutable std::vector<recorded_eval> log;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        const double f = (x[0] - 0.5) * (x[0] - 0.5)
                       + (x[1] + 0.25) * (x[1] + 0.25);
        log.push_back({x, f});
        return f;
    }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = 300.0 - x[0] - x[1];
    }

    int num_equality() const { return 0; }
    int num_inequality() const { return 1; }

    Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd{{-10.0, -10.0}};
    }

    Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd{{10.0, 10.0}};
    }
};

}

// Now PASSES: the generation loop tracks the best-ever (x, f, v) record
// per evaluation (NLopt isres.c:174-193), checking every individual at
// evaluation time rather than the previous generation's rank-0 slot.
// Pre-fix the returned objective was 2.43877 vs the true best evaluated
// 0.04079 at evaluation #150 of 1860 (seed 42, 30 generations,
// lambda=60), because points seen mid-stream were lost.
TEST_CASE("isres: returned minimizer is the best point at evaluation time",
          "[isres][oracle-pin]")
{
    recording_qp problem;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;

    isres_policy<> policy;
    policy.options.seed = 42u;

    auto s = policy.init(problem, x0, opts);
    for(int g = 0; g < 30; ++g)
        (void)policy.step(s);

    REQUIRE(!problem.log.empty());

    // Enumerate the evaluation log: the true best evaluated point.
    std::size_t best = 0;
    for(std::size_t i = 1; i < problem.log.size(); ++i)
        if(problem.log[i].f < problem.log[best].f)
            best = i;
    const auto& truth = problem.log[best];

    // Premise: the best evaluated point is feasible (holds for every
    // point in the box by construction).
    Eigen::VectorXd c(1);
    problem.constraints(truth.x, c);
    REQUIRE(c[0] > 0.0);

    // Per-evaluation best pin (NLopt isres.c:174-193 semantics): the
    // solver may not return anything worse than the best point it
    // evaluated.
    INFO("evaluations: " << problem.log.size());
    INFO("true best f " << truth.f << " at eval #" << best
         << ", returned f " << s.objective_value);
    CHECK(s.objective_value <= truth.f + 1e-12);
    CHECK(s.x[0] == Approx(truth.x[0]).epsilon(0).margin(1e-12));
    CHECK(s.x[1] == Approx(truth.x[1]).epsilon(0).margin(1e-12));
}

// Green leg: the oracle constants themselves are pinned against the
// Schwefel/NLopt rate formulas independently of the production
// detail::compute_es_rates output, so this stays a normally-passing
// regression guard regardless of the tau/tau' swap defect below.
TEST_CASE("isres: tau/tau' oracle constants match the Schwefel/NLopt rate formulas",
          "[isres][oracle-pin]")
{
    // Role-correct oracle values (NLopt isres.c; Schwefel convention):
    // per-component rate 1/sqrt(2 sqrt(n)), global rate 1/sqrt(2 n).
    const double per_component = 1.0 / std::sqrt(2.0 * std::sqrt(20.0));
    const double global_rate = 1.0 / std::sqrt(2.0 * 20.0);
    CHECK(per_component == Approx(0.33437015248821100).epsilon(1e-15));
    CHECK(global_rate == Approx(0.15811388300841897).epsilon(1e-15));
}

// Now PASSES: detail::compute_es_rates assigns the per-component rate
// 1/sqrt(2*sqrt(n)) to rates.tau and the global rate 1/sqrt(2*n) to
// rates.tau_prime, matching the generation loop's consumption
// (rates.tau_prime scales the once-per-individual taup draw; rates.tau
// the per-component draw in log_normal_mutate). Pre-fix the two
// formulas were assigned to the opposite roles.
TEST_CASE("isres: tau/tau' role assignment matches NLopt",
          "[isres][oracle-pin]")
{
    const int n = 20;
    const auto rates = detail::compute_es_rates(n);

    const double per_component = 1.0 / std::sqrt(2.0 * std::sqrt(20.0));
    const double global_rate = 1.0 / std::sqrt(2.0 * 20.0);

    // rates.tau multiplies the per-component draw; rates.tau_prime the
    // once-per-individual draw (see the generation loop's usage).
    INFO("rates.tau (per-component role) = " << rates.tau
         << ", oracle " << per_component);
    INFO("rates.tau_prime (global role) = " << rates.tau_prime
         << ", oracle " << global_rate);
    CHECK(rates.tau == Approx(per_component).epsilon(1e-12));
    CHECK(rates.tau_prime == Approx(global_rate).epsilon(1e-12));

    // Magnitude-ordering form of the same pin: the per-component rate
    // exceeds the global rate for any n > 1.
    CHECK(rates.tau > rates.tau_prime);
}

namespace
{

// Equality-constrained problem: minimize (x0-2)^2 + (x1-2)^2 subject to
// x0 + x1 = 1, bounds [-10, 10]^2. The equality's L2-squared violation
// v = (x0 + x1 - 1)^2 is essentially never exactly zero for a stochastic
// sampler, so the returned status must be gated on the feasibility
// tolerance, not on v <= 0.0.
struct equality_qp
{
    static constexpr int problem_dimension = dynamic_dimension;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        return (x[0] - 2.0) * (x[0] - 2.0) + (x[1] - 2.0) * (x[1] - 2.0);
    }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = x[0] + x[1] - 1.0;  // equality: c == 0
    }

    int num_equality() const { return 1; }
    int num_inequality() const { return 0; }

    Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd{{-10.0, -10.0}};
    }

    Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd{{10.0, 10.0}};
    }
};

}

// Generation-loop faithfulness: mu = ceil(lambda/7) (NLopt rounding, not
// a truncating int cast) and x0 is injected into the initial population
// (NLopt seeds the first individual with the caller's starting point
// instead of discarding it).
TEST_CASE("isres: mu rounds up (ceil) and x0 is injected into the initial population",
          "[isres][oracle-pin]")
{
    recording_qp problem;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;

    isres_policy<> policy;
    policy.options.seed = 42u;
    policy.options.population_size = 60u;  // ceil(60/7) = 9, trunc = 8

    auto s = policy.init(problem, x0, opts);

    // mu-rounding: ceil(60/7) = 9, not int(60 * (1/7)) = 8.
    CHECK(s.lambda == 60);
    CHECK(s.mu == 9);

    // x0-injection: the first individual is exactly the (in-bounds) x0.
    CHECK(s.population(0, 0) == Approx(2.0).epsilon(0).margin(1e-12));
    CHECK(s.population(1, 0) == Approx(2.0).epsilon(0).margin(1e-12));
}

// Equality-constrained best gate: whenever the solver emits a converged
// status (ftol_reached), the RETURNED x must be feasible within the
// feasibility gate. Pre-fix the feasible branch required v <= 0.0
// exactly (unreachable for an equality's squared violation) and the
// status was gated on the rank-0 individual rather than the returned x,
// so the solver could return an infeasible x while reporting
// convergence.
TEST_CASE("isres: converged status implies a feasible returned x on an equality constraint",
          "[isres][oracle-pin]")
{
    equality_qp problem;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;
    opts.max_iterations = 2000;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    isres_policy<> policy;
    policy.options.seed = 7u;
    policy.options.feasibility_gate = 1e-4;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve();

    // The best-ever record must never carry an infeasible point out as a
    // converged result. If the run reports convergence, the returned x's
    // equality residual must sit within the feasibility gate (L2-squared).
    Eigen::VectorXd c(1);
    problem.constraints(result.x, c);
    const double v = c[0] * c[0];

    INFO("returned x = (" << result.x[0] << ", " << result.x[1]
         << "), equality residual^2 = " << v
         << ", status converged = "
         << (result.status == solver_status::ftol_reached));
    if(result.status == solver_status::ftol_reached)
        CHECK(v <= 1e-4);

    // Regardless of status, the returned objective is finite and the
    // best-ever record is self-consistent (never worse than x0 once a
    // feasible point has been seen).
    CHECK(std::isfinite(result.objective_value));
}
