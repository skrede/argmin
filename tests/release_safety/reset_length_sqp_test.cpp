#include "eigen_trap.h"
#include "wrong_length_reset_probe.h"

#include "cases/sqp_cases.h"
#include "cases/trust_region_sqp_cases.h"

#include <catch2/catch_test_macros.hpp>

using namespace argmin;

TEST_CASE("every line-search SQP case rejects a reset of the wrong length and recovers on its own start", "[release-safety][reset-length]")
{
    release_safety::wrong_length_reset_probe probe;
    cases::visit_sqp_cases(probe);
    release_safety::check_every_key_visited(probe, cases::sqp_keys);
}

TEST_CASE("every trust-region SQP case rejects a reset of the wrong length and recovers on its own start", "[release-safety][reset-length]")
{
    release_safety::wrong_length_reset_probe probe;
    cases::visit_trust_region_sqp_cases(probe);
    release_safety::check_every_key_visited(probe, cases::trust_region_sqp_keys);
}
