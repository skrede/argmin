#ifndef HPP_GUARD_ARGMIN_TESTS_GOLDEN_GOLDEN_FILE_H
#define HPP_GUARD_ARGMIN_TESTS_GOLDEN_GOLDEN_FILE_H

#include "golden_identity.h"

#include <string>
#include <vector>
#include <cstddef>
#include <fstream>
#include <istream>
#include <ostream>
#include <utility>
#include <charconv>

namespace argmin::golden
{

struct traced_case
{
    std::string name;
    std::vector<std::string> steps;
    std::vector<std::pair<std::string, std::string>> finals;
};

using trace_table = std::vector<traced_case>;

struct golden_record
{
    std::string identity;
    trace_table table;
    std::vector<std::string> malformed;
};

inline void write_golden(std::ostream& out, const std::string& identity, const trace_table& table)
{
    out << "target_identity," << identity << '\n';
    for(const traced_case& traced : table)
    {
        for(std::size_t i = 0; i < traced.steps.size(); ++i)
            out << traced.name << '|' << i << ',' << traced.steps[i] << '\n';
        for(const auto& [field, value] : traced.finals)
            out << traced.name << "|final|" << field << ',' << value << '\n';
    }
}

inline void write_golden(const std::string& path, const trace_table& table)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    write_golden(out, running_identity(), table);
}

inline traced_case& case_named(trace_table& table, const std::string& name)
{
    if(table.empty() || table.back().name != name)
        table.push_back(traced_case{name, {}, {}});
    return table.back();
}

inline bool parse_step(trace_table& table, const std::string& head, const std::string& value)
{
    const std::size_t bar = head.rfind('|');
    std::size_t iteration = 0;
    const char* last = head.data() + head.size();
    if(bar == std::string::npos || std::from_chars(head.data() + bar + 1, last, iteration).ptr != last)
        return false;
    traced_case& traced = case_named(table, head.substr(0, bar));
    traced.steps.push_back(value);
    return traced.steps.size() == iteration + 1;
}

inline bool parse_line(golden_record& record, const std::string& line)
{
    const std::size_t comma = line.rfind(',');
    if(comma == std::string::npos)
        return false;
    const std::string head = line.substr(0, comma);
    const std::string value = line.substr(comma + 1);
    if(head == "target_identity")
        record.identity = value;
    else if(const std::size_t final_at = head.find("|final|"); final_at != std::string::npos)
        case_named(record.table, head.substr(0, final_at)).finals.emplace_back(head.substr(final_at + 7), value);
    else
        return parse_step(record.table, head, value);
    return true;
}

inline golden_record load_golden(std::istream& in)
{
    golden_record record{};
    std::string line;
    while(std::getline(in, line))
    {
        if(!line.empty() && line.back() == '\r')
            line.pop_back();
        if(!line.empty() && line.front() != '#' && !parse_line(record, line))
            record.malformed.push_back(line);
    }
    return record;
}

inline golden_record load_golden(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    return load_golden(in);
}

}

#endif
