#include "family_traces.h"
#include "trajectory_trace.h"

#include "cases/gradient_cases.h"

namespace argmin::golden
{

void trace_gradient(trace_table& table, const trace_settings& settings)
{
    tracing_visitor visitor(table, settings);
    cases::visit_gradient_cases(visitor);
}

}
