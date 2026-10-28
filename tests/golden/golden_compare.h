#ifndef HPP_GUARD_ARGMIN_TESTS_GOLDEN_GOLDEN_COMPARE_H
#define HPP_GUARD_ARGMIN_TESTS_GOLDEN_GOLDEN_COMPARE_H

#include "golden_file.h"

#include <string>
#include <vector>
#include <cstddef>
#include <utility>
#include <algorithm>

namespace argmin::golden
{

inline const traced_case* find_case(const trace_table& table, const std::string& name)
{
    const auto it = std::ranges::find(table, name, &traced_case::name);
    return it == table.end() ? nullptr : &*it;
}

inline std::string final_value(const traced_case& traced, const std::string& field)
{
    const auto it = std::ranges::find(traced.finals, field, &std::pair<std::string, std::string>::first);
    return it == traced.finals.end() ? "<absent>" : it->second;
}

inline void compare_steps(std::vector<std::string>& report, const traced_case& golden, const traced_case& live)
{
    const auto [g, l] = std::ranges::mismatch(golden.steps, live.steps);
    if(g == golden.steps.end() && l == live.steps.end())
        return;
    const std::size_t iteration = static_cast<std::size_t>(g - golden.steps.begin());
    report.push_back(live.name + ": first diverging iteration " + std::to_string(iteration));
}

inline void compare_finals(std::vector<std::string>& report, const traced_case& golden, const traced_case& live)
{
    std::vector<std::string> fields;
    for(const auto& [field, value] : golden.finals)
        fields.push_back(field);
    for(const auto& [field, value] : live.finals)
        if(std::ranges::find(fields, field) == fields.end())
            fields.push_back(field);
    for(const std::string& field : fields)
    {
        const std::string expected = final_value(golden, field);
        const std::string actual = final_value(live, field);
        if(expected != actual)
            report.push_back(live.name + ": final " + field + " golden " + expected + " live " + actual);
    }
}

// Every value is compared as text of its exact bits, so +0.0 against -0.0 and two NaN payloads differ.
inline std::vector<std::string> compare_tables(const trace_table& golden, const trace_table& live)
{
    std::vector<std::string> report;
    for(const traced_case& traced : live)
    {
        if(const traced_case* expected = find_case(golden, traced.name))
        {
            compare_steps(report, *expected, traced);
            compare_finals(report, *expected, traced);
        }
        else
            report.push_back(traced.name + ": missing from the golden");
    }
    for(const traced_case& expected : golden)
    {
        if(find_case(live, expected.name) == nullptr)
            report.push_back(expected.name + ": missing from the live trace");
        else if(std::ranges::count(golden, expected.name, &traced_case::name) > 1)
            report.push_back(expected.name + ": listed more than once in the golden");
    }
    return report;
}

}

#endif
