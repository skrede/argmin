#include "golden_file.h"
#include "family_traces.h"
#include "golden_compare.h"
#include "golden_identity.h"

#include "cases/case_keys.h"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <string_view>

using namespace argmin;

namespace
{

std::string env_or(const char* name, const std::string& fallback)
{
    const char* value = std::getenv(name);
    return value == nullptr ? fallback : std::string(value);
}

// Only the authoritative leg enforces: elsewhere the golden belongs to another build identity, so a
// missing or foreign golden is reported and the comparison is skipped.
bool golden_applies(const golden::golden_record& record, const std::string& path)
{
    const bool enforce = env_or("ARGMIN_TRAJECTORY_GOLDEN_ENFORCE", "") == "1";
    std::string problem;
    if(record.table.empty())
        problem = "no trajectory golden at '" + path + "'";
    else if(record.identity != golden::running_identity())
        problem = "trajectory golden captured on '" + record.identity + "', this build is '" + golden::running_identity() + "'";
    if(problem.empty())
        return true;
    if(enforce)
        FAIL(problem);
    WARN(problem << "; not compared on this build");
    return false;
}

std::string joined(const std::vector<std::string>& lines)
{
    std::string text;
    for(const std::string& line : lines)
        text += line + '\n';
    return text;
}

void check_every_key_traced(const golden::trace_table& live)
{
    for(const std::string_view key : cases::case_keys)
    {
        INFO("solver key with no traced case: " << key);
        CHECK(std::ranges::any_of(live, [&](const golden::traced_case& traced) { return std::string_view(traced.name).substr(0, traced.name.find('|')) == key; }));
    }
}

void compare_with_golden(const golden::golden_record& record, const golden::trace_table& live)
{
    if(!record.malformed.empty())
        FAIL_CHECK("malformed golden lines:\n" << joined(record.malformed));
    const std::vector<std::string> report = golden::compare_tables(record.table, live);
    if(report.empty())
        return;
    if(const std::string live_out = env_or("ARGMIN_TRAJECTORY_GOLDEN_LIVE_OUT", ""); !live_out.empty())
        golden::write_golden(live_out, live);
    std::cout << "trajectory golden mismatch:\n" << joined(report);
    FAIL_CHECK("trajectory golden mismatch on " << report.size() << " lines, listed above unwrapped");
}

}

// A build that will neither capture nor compare skips the trace: the final solves take minutes in a
// debug build, and the oracle tests already solve every case.
TEST_CASE("every solver trajectory matches the all-family trajectory golden", "[oracle-pin][trajectory-golden]")
{
    std::cout << "trajectory golden: build " << golden::running_identity() << ", runtime CPU FMA " << golden::runtime_fma() << '\n';
    const std::string capture = env_or("ARGMIN_TRAJECTORY_GOLDEN_CAPTURE", "");
    const std::string path = env_or("ARGMIN_TRAJECTORY_GOLDEN", ARGMIN_TRAJECTORY_GOLDEN_DEFAULT);
    const golden::golden_record record = capture.empty() ? golden::load_golden(path) : golden::golden_record{};
    if(capture.empty() && !golden_applies(record, path))
        return;
    const std::string perturb_key = env_or("ARGMIN_TRAJECTORY_GOLDEN_PERTURB", "");
    const golden::trace_table live = golden::trace_all_families(golden::trace_settings{perturb_key, "", true});
    check_every_key_traced(live);
    if(!capture.empty())
    {
        golden::write_golden(capture, live);
        SUCCEED("trajectory golden captured to " << capture);
        return;
    }
    compare_with_golden(record, live);
}
