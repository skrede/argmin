#ifndef HPP_GUARD_ARGMIN_BENCHMARKS_BENCH_MICRO_GATE_H
#define HPP_GUARD_ARGMIN_BENCHMARKS_BENCH_MICRO_GATE_H

#include "bench_print.h"

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace argmin::bench
{

struct micro_gate
{
    double optimum{};
    double objective_tolerance{};
    double feasibility_tolerance{};
};

struct micro_observation
{
    std::string_view solver;
    double objective{};
    double constraint_violation{};
};

[[nodiscard]] inline auto forced_micro_gate_failure(std::string_view solver) -> bool
{
    const char* forced = std::getenv("ARGMIN_BENCH_FORCE_GATE_FAIL");
    if(forced == nullptr)
        return false;

    const std::string_view value{forced};
    if(value == "all" || value == solver)
        return true;
    if(value == "nlopt")
        return solver == "nlopt";
    if(value == "argmin")
        return solver != "nlopt";
    return false;
}

[[nodiscard]] inline auto objective_passes(const micro_observation& obs,
                                           const micro_gate& gate) -> bool
{
    const double scale = std::fmax(1.0, std::fabs(gate.optimum));
    return std::isfinite(obs.objective)
        && std::fabs(obs.objective - gate.optimum)
               <= gate.objective_tolerance * scale
        && !forced_micro_gate_failure(obs.solver);
}

[[nodiscard]] inline auto feasibility_passes(const micro_observation& obs,
                                             const micro_gate& gate) -> bool
{
    return std::isfinite(obs.constraint_violation)
        && obs.constraint_violation <= gate.feasibility_tolerance
        && !forced_micro_gate_failure(obs.solver);
}

[[nodiscard]] inline auto observation_passes(const micro_observation& obs,
                                             const micro_gate& gate) -> bool
{
    return objective_passes(obs, gate) && feasibility_passes(obs, gate);
}

inline void print_gated_comparison(std::string_view label,
                                   const micro_observation& lhs,
                                   const micro_observation& rhs,
                                   const micro_gate& gate)
{
    println("  GATED: {} comparison suppressed", label);
    println("    {} objective_ok={} feasibility_ok={} f={:.6e} cv={:.6e}",
            lhs.solver,
            objective_passes(lhs, gate),
            feasibility_passes(lhs, gate),
            lhs.objective,
            lhs.constraint_violation);
    println("    {} objective_ok={} feasibility_ok={} f={:.6e} cv={:.6e}",
            rhs.solver,
            objective_passes(rhs, gate),
            feasibility_passes(rhs, gate),
            rhs.objective,
            rhs.constraint_violation);
}

[[nodiscard]] inline auto comparison_passes(std::string_view label,
                                            const micro_observation& lhs,
                                            const micro_observation& rhs,
                                            const micro_gate& gate) -> bool
{
    if(observation_passes(lhs, gate) && observation_passes(rhs, gate))
        return true;
    print_gated_comparison(label, lhs, rhs, gate);
    return false;
}

[[nodiscard]] inline auto per_unit_us(double wall_us, std::uint32_t units) -> double
{
    if(units == 0)
        return std::numeric_limits<double>::infinity();
    return wall_us / static_cast<double>(units);
}

template <typename Problem, typename Vec>
[[nodiscard]] auto constraint_violation(const Problem& problem, const Vec& x) -> double
{
    const int eq_count = problem.num_equality();
    const int ineq_count = problem.num_inequality();
    if(eq_count + ineq_count == 0)
        return 0.0;

    Eigen::VectorXd c(eq_count + ineq_count);
    problem.constraints(x, c);

    double cv = 0.0;
    for(int i = 0; i < eq_count; ++i)
        cv = std::fmax(cv, std::fabs(c[i]));
    for(int i = 0; i < ineq_count; ++i)
        cv = std::fmax(cv, -c[eq_count + i]);
    return cv;
}

}

#endif
