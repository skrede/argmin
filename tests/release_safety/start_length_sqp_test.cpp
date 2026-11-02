#include "eigen_trap.h"
#include "wrong_length_start_probe.h"

#include "cases/sqp_cases.h"
#include "cases/trust_region_sqp_cases.h"

#include <catch2/catch_test_macros.hpp>

using namespace argmin;

TEST_CASE("every line-search SQP case rejects a start of the wrong length at construction and is unchanged on its own start", "[release-safety][start-length]")
{
    release_safety::wrong_length_start_probe probe;
    cases::visit_sqp_cases(probe);
    release_safety::check_every_key_constructed(probe, cases::sqp_keys);
}

TEST_CASE("every trust-region SQP case rejects a start of the wrong length at construction and is unchanged on its own start", "[release-safety][start-length]")
{
    release_safety::wrong_length_start_probe probe;
    cases::visit_trust_region_sqp_cases(probe);
    release_safety::check_every_key_constructed(probe, cases::trust_region_sqp_keys);
}
