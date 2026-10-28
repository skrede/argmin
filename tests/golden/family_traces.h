#ifndef HPP_GUARD_ARGMIN_TESTS_GOLDEN_FAMILY_TRACES_H
#define HPP_GUARD_ARGMIN_TESTS_GOLDEN_FAMILY_TRACES_H

#include "golden_file.h"

#include <string_view>

namespace argmin::golden
{

// An empty only_key traces every case; a set one traces that key's cases alone.
struct trace_settings
{
    std::string_view perturb_key;
    std::string_view only_key;
    bool with_finals;
};

void trace_gradient(trace_table& table, const trace_settings& settings);
void trace_least_squares(trace_table& table, const trace_settings& settings);
void trace_sqp(trace_table& table, const trace_settings& settings);
void trace_trust_region_sqp(trace_table& table, const trace_settings& settings);
void trace_moving_asymptote(trace_table& table, const trace_settings& settings);
void trace_derivative_free(trace_table& table, const trace_settings& settings);
void trace_stochastic(trace_table& table, const trace_settings& settings);
void trace_qp(trace_table& table, const trace_settings& settings);

// Each family is traced in its own translation unit, in case-table order, so the golden's order is fixed.
template <typename Visitor>
void for_each_family(Visitor&& visitor)
{
    visitor(trace_gradient);
    visitor(trace_least_squares);
    visitor(trace_sqp);
    visitor(trace_trust_region_sqp);
    visitor(trace_moving_asymptote);
    visitor(trace_derivative_free);
    visitor(trace_stochastic);
    visitor(trace_qp);
}

inline trace_table trace_all_families(const trace_settings& settings)
{
    trace_table table;
    for_each_family([&](auto& trace) { trace(table, settings); });
    return table;
}

}

#endif
