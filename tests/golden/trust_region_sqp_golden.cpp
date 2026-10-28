#include "family_traces.h"
#include "trajectory_trace.h"

#include "cases/trust_region_sqp_cases.h"

namespace argmin::golden
{

void trace_trust_region_sqp(trace_table& table, const trace_settings& settings)
{
    tracing_visitor visitor(table, settings);
    cases::visit_trust_region_sqp_cases(visitor);
}

}
