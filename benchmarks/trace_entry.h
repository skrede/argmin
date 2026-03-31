#ifndef HPP_GUARD_NABLAPP_BENCHMARKS_TRACE_ENTRY_H
#define HPP_GUARD_NABLAPP_BENCHMARKS_TRACE_ENTRY_H

namespace nablapp::bench
{

struct trace_entry
{
    int iteration{};
    double objective_value{};
    double gradient_norm{};
    double constraint_violation{};
};

}

#endif
