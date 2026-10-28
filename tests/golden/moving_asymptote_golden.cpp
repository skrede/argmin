#include "family_traces.h"
#include "trajectory_trace.h"

#include "cases/moving_asymptote_cases.h"

namespace argmin::golden
{

void trace_moving_asymptote(trace_table& table, const trace_settings& settings)
{
    tracing_visitor visitor(table, settings);
    cases::visit_moving_asymptote_cases(visitor);
}

}
