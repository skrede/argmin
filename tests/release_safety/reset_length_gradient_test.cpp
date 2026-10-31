#include "eigen_trap.h"
#include "wrong_length_reset_probe.h"

#include "cases/gradient_cases.h"
#include "cases/least_squares_cases.h"

#include <catch2/catch_test_macros.hpp>

using namespace argmin;

TEST_CASE("every gradient and augmented Lagrangian case rejects a reset of the wrong length and recovers on its own start", "[release-safety][reset-length]")
{
    release_safety::wrong_length_reset_probe probe;
    cases::visit_gradient_cases(probe);
    release_safety::check_every_key_visited(probe, cases::gradient_keys);
}

TEST_CASE("every least-squares case rejects a reset of the wrong length and recovers on its own start", "[release-safety][reset-length]")
{
    release_safety::wrong_length_reset_probe probe;
    cases::visit_least_squares_cases(probe);
    release_safety::check_every_key_visited(probe, cases::least_squares_keys);
}
