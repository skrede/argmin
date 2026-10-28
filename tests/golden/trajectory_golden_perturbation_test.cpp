#include "golden_file.h"
#include "family_traces.h"
#include "golden_compare.h"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>
#include <iterator>
#include <algorithm>

using namespace argmin;

namespace
{

std::string key_of(const golden::traced_case& traced)
{
    return traced.name.substr(0, traced.name.find('|'));
}

std::vector<std::string> keys_of(const golden::trace_table& table)
{
    std::vector<std::string> keys;
    for(const golden::traced_case& traced : table)
        if(std::ranges::find(keys, key_of(traced)) == keys.end())
            keys.push_back(key_of(traced));
    return keys;
}

golden::trace_table cases_of(const golden::trace_table& table, const std::string& key)
{
    golden::trace_table selected;
    std::ranges::copy_if(table, std::back_inserter(selected), [&](const golden::traced_case& traced) { return key_of(traced) == key; });
    return selected;
}

// These ISRES variants draw their whole first population inside the bounds and never read the start
// point, so a start perturbation cannot reach them; the control pins that their trace stays unchanged.
bool reads_start_point(const std::string& key)
{
    return key != "isres_original_argmin" && key != "isres_runarsson_yao_paper";
}

// A perturbed key diverges in at least one of its cases. Not every case diverges: one ulp can round away
// in the first update, as a subnormal step off a zero start coordinate does.
template <typename Trace>
void check_perturbations_diverge(Trace& trace)
{
    golden::trace_table base;
    golden::trace_table again;
    trace(base, golden::trace_settings{"", "", false});
    trace(again, golden::trace_settings{"", "", false});
    CHECK(golden::compare_tables(base, again).empty());
    for(const std::string& key : keys_of(base))
    {
        golden::trace_table perturbed;
        trace(perturbed, golden::trace_settings{key, key, false});
        const std::vector<std::string> report = golden::compare_tables(cases_of(base, key), perturbed);
        INFO("perturbed " << key);
        CHECK(report.empty() != reads_start_point(key));
    }
}

}

TEST_CASE("a one-ulp start perturbation diverges every gradient-based solver trajectory", "[trajectory-golden]")
{
    check_perturbations_diverge(golden::trace_gradient);
}

TEST_CASE("a one-ulp start perturbation diverges every least-squares solver trajectory", "[trajectory-golden]")
{
    check_perturbations_diverge(golden::trace_least_squares);
}

TEST_CASE("a one-ulp start perturbation diverges every SQP solver trajectory", "[trajectory-golden]")
{
    check_perturbations_diverge(golden::trace_sqp);
}

TEST_CASE("a one-ulp start perturbation diverges every trust-region SQP solver trajectory", "[trajectory-golden]")
{
    check_perturbations_diverge(golden::trace_trust_region_sqp);
}

TEST_CASE("a one-ulp start perturbation diverges every moving-asymptote solver trajectory", "[trajectory-golden]")
{
    check_perturbations_diverge(golden::trace_moving_asymptote);
}

TEST_CASE("a one-ulp start perturbation diverges every derivative-free solver trajectory", "[trajectory-golden]")
{
    check_perturbations_diverge(golden::trace_derivative_free);
}

TEST_CASE("a one-ulp start perturbation diverges every stochastic solver trajectory that reads its start", "[trajectory-golden]")
{
    check_perturbations_diverge(golden::trace_stochastic);
}

TEST_CASE("a one-ulp cost perturbation diverges every QP solver result", "[trajectory-golden]")
{
    check_perturbations_diverge(golden::trace_qp);
}
