#include "eigen_trap.h"
#include "wrong_length_start_probe.h"

#include "cases/stochastic_cases.h"
#include "cases/derivative_free_cases.h"
#include "cases/moving_asymptote_cases.h"

#include <catch2/catch_test_macros.hpp>

using namespace argmin;

TEST_CASE("every moving-asymptote case rejects a start of the wrong length at construction and is unchanged on its own start", "[release-safety][start-length]")
{
    release_safety::wrong_length_start_probe probe;
    cases::visit_moving_asymptote_cases(probe);
    release_safety::check_every_key_constructed(probe, cases::moving_asymptote_keys);
}

TEST_CASE("every derivative-free case rejects a start of the wrong length at construction and is unchanged on its own start", "[release-safety][start-length]")
{
    release_safety::wrong_length_start_probe probe;
    cases::visit_derivative_free_cases(probe);
    release_safety::check_every_key_constructed(probe, cases::derivative_free_keys);
}

TEST_CASE("every stochastic case rejects a start of the wrong length at construction and is unchanged on its own start", "[release-safety][start-length]")
{
    release_safety::wrong_length_start_probe probe;
    cases::visit_stochastic_cases(probe);
    release_safety::check_every_key_constructed(probe, cases::stochastic_keys);
}
