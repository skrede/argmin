#ifndef HPP_GUARD_ARGMIN_TESTS_GOLDEN_TRAJECTORY_TRACE_H
#define HPP_GUARD_ARGMIN_TESTS_GOLDEN_TRAJECTORY_TRACE_H

#include "golden_file.h"
#include "status_text.h"
#include "family_traces.h"
#include "trajectory_digest.h"

#include "cases/case_options.h"

#include "argmin/result/status.h"

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <string>
#include <cstdint>
#include <utility>

namespace argmin::golden
{

inline constexpr std::int32_t step_cap = 50;

inline std::string case_name(const cases::case_id& id)
{
    return std::string(id.key) + "|" + std::string(id.problem_class) + "|" + std::string(id.problem);
}

template <typename Vector>
Vector perturbed_start(Vector x0, bool perturb)
{
    if(perturb)
        x0(0) = std::nextafter(x0(0), std::numeric_limits<double>::infinity());
    return x0;
}

template <typename Solver>
void trace_steps(traced_case& traced, Solver& solver)
{
    for(std::int32_t i = 0; i < step_cap && solver.status() == solver_status::running; ++i)
    {
        const auto step = solver.step();
        traced.steps.push_back(to_hex(step_digest(solver, step)));
    }
}

template <typename Result>
void trace_final(traced_case& traced, const Result& result)
{
    for(Eigen::Index i = 0; i < result.x.size(); ++i)
        traced.finals.emplace_back("x" + std::to_string(i), to_hex(static_cast<double>(result.x(i))));
    traced.finals.emplace_back("f", to_hex(static_cast<double>(result.objective_value)));
    traced.finals.emplace_back("status", status_text(result.status));
    traced.finals.emplace_back("iterations", std::to_string(result.iterations));
    traced.finals.emplace_back("evaluations", std::to_string(result.function_evaluations));
}

// The step trace and the final solve use separate solvers, because solve() continues a stepped solver
// rather than restarting it.
class tracing_visitor
{
public:
    tracing_visitor(trace_table& table, const trace_settings& settings) : table_(table), settings_(settings) {}

    template <typename Problem, typename Options, typename MakeSolver>
    void operator()(const cases::case_id& id, const Problem& problem, const Options& options, const MakeSolver& make_solver)
    {
        if(!settings_.only_key.empty() && id.key != settings_.only_key)
            return;
        traced_case traced{case_name(id), {}, {}};
        const auto x0 = perturbed_start(problem.initial_point(), id.key == settings_.perturb_key);
        auto stepping = make_solver(x0);
        trace_steps(traced, stepping);
        if(settings_.with_finals)
        {
            auto solving = make_solver(x0);
            trace_final(traced, solving.solve(options));
        }
        table_.push_back(std::move(traced));
    }

private:
    trace_table& table_;
    trace_settings settings_;
};

}

#endif
