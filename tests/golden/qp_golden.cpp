#include "family_traces.h"
#include "trajectory_trace.h"
#include "trajectory_digest.h"

#include "cases/qp_cases.h"

#include "argmin/qp/qp_types.h"

#include <cmath>
#include <limits>
#include <string>
#include <cstdint>
#include <utility>
#include <string_view>

namespace argmin::golden
{

namespace
{

std::string_view qp_status_name(qp_solve_status status)
{
    switch(status)
    {
    case qp_solve_status::solved:
        return "solved";
    case qp_solve_status::solved_inaccurate:
        return "solved_inaccurate";
    case qp_solve_status::max_iterations:
        return "max_iterations";
    case qp_solve_status::primal_infeasible:
        return "primal_infeasible";
    case qp_solve_status::dual_infeasible:
        return "dual_infeasible";
    }
    return "unknown";
}

void trace_qp_result(traced_case& traced, const qp_result<double>& result)
{
    for(Eigen::Index i = 0; i < result.x.size(); ++i)
        traced.finals.emplace_back("x" + std::to_string(i), to_hex(result.x(i)));
    for(Eigen::Index i = 0; i < result.y.size(); ++i)
        traced.finals.emplace_back("y" + std::to_string(i), to_hex(result.y(i)));
    traced.finals.emplace_back("iterations", std::to_string(result.iterations));
    const std::int32_t code = static_cast<std::int32_t>(result.status);
    traced.finals.emplace_back("status", std::to_string(code) + ":" + std::string(qp_status_name(result.status)));
}

// The QP solvers have no step interface, so a QP case records only its final state, and its
// perturbation moves the first linear cost coefficient instead of a start point.
class qp_tracing_visitor
{
public:
    qp_tracing_visitor(trace_table& table, const trace_settings& settings) : table_(table), settings_(settings) {}

    template <typename Options, typename MakeSolver>
    void operator()(const cases::case_id& id, const cases::qp_case& data, const Options& options, const MakeSolver& make_solver)
    {
        if(!settings_.only_key.empty() && id.key != settings_.only_key)
            return;
        traced_case traced{case_name(id), {}, {}};
        cases::qp_case posed = data;
        if(id.key == settings_.perturb_key)
            posed.q(0) = std::nextafter(posed.q(0), std::numeric_limits<double>::infinity());
        auto solver = make_solver();
        const auto result = cases::solve_qp(solver, posed, options);
        if(result.has_value())
            trace_qp_result(traced, *result);
        else
            traced.finals.emplace_back("status", "error:" + std::to_string(static_cast<std::int32_t>(result.error())));
        table_.push_back(std::move(traced));
    }

private:
    trace_table& table_;
    trace_settings settings_;
};

}

void trace_qp(trace_table& table, const trace_settings& settings)
{
    qp_tracing_visitor visitor(table, settings);
    cases::visit_qp_cases(visitor);
}

}
