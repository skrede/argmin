#include "family_traces.h"
#include "trajectory_trace.h"

#include "cases/stochastic_cases.h"

namespace argmin::golden
{

void trace_stochastic(trace_table& table, const trace_settings& settings)
{
    tracing_visitor visitor(table, settings);
    cases::visit_stochastic_cases(visitor);
}

}
