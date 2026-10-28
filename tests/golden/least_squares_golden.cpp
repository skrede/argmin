#include "family_traces.h"
#include "trajectory_trace.h"

#include "cases/least_squares_cases.h"

namespace argmin::golden
{

void trace_least_squares(trace_table& table, const trace_settings& settings)
{
    tracing_visitor visitor(table, settings);
    cases::visit_least_squares_cases(visitor);
}

}
