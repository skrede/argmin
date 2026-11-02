#include "eigen_trap.h"
#include "wrong_length_start_probe.h"

#include "cases/gradient_cases.h"
#include "cases/least_squares_cases.h"

#include <catch2/catch_test_macros.hpp>

using namespace argmin;

TEST_CASE("every gradient and augmented Lagrangian case rejects a start of the wrong length at construction and is unchanged on its own start", "[release-safety][start-length]")
{
    release_safety::wrong_length_start_probe probe;
    cases::visit_gradient_cases(probe);
    release_safety::check_every_key_constructed(probe, cases::gradient_keys);
}

TEST_CASE("every least-squares case rejects a start of the wrong length at construction and is unchanged on its own start", "[release-safety][start-length]")
{
    release_safety::wrong_length_start_probe probe;
    cases::visit_least_squares_cases(probe);
    release_safety::check_every_key_constructed(probe, cases::least_squares_keys);
}
