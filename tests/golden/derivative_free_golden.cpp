#include "family_traces.h"
#include "trajectory_trace.h"

#include "cases/derivative_free_cases.h"

namespace argmin::golden
{

void trace_derivative_free(trace_table& table, const trace_settings& settings)
{
    tracing_visitor visitor(table, settings);
    cases::visit_derivative_free_cases(visitor);
}

}
