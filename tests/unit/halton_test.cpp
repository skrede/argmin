#include "nablapp/sampling/halton.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using namespace nablapp;

TEST_CASE("van der Corput base 2", "[halton]")
{
    halton_sequence seq(1);

    // Known sequence: 1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8
    const double expected[] = {0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875};
    for(int i = 0; i < 7; ++i)
    {
        auto pt = seq.next();
        CHECK(pt(0) == Approx(expected[i]).epsilon(1e-12));
    }
}

TEST_CASE("van der Corput base 3", "[halton]")
{
    halton_sequence seq(1);

    // For base 3: 1/3, 2/3, 1/9, 4/9
    // But 1D Halton uses base 2. We need a way to test base 3.
    // The 2D Halton second component uses base 3.
    halton_sequence seq2(2);

    const double expected[] = {1.0 / 3.0, 2.0 / 3.0, 1.0 / 9.0, 4.0 / 9.0};
    for(int i = 0; i < 4; ++i)
    {
        auto pt = seq2.next();
        CHECK(pt(1) == Approx(expected[i]).epsilon(1e-12));
    }
}

TEST_CASE("2D Halton first point", "[halton]")
{
    halton_sequence seq(2);
    auto pt = seq.next();

    CHECK(pt(0) == Approx(0.5).epsilon(1e-12));
    CHECK(pt(1) == Approx(1.0 / 3.0).epsilon(1e-12));
}

TEST_CASE("sample(n) dimensions and bounds", "[halton]")
{
    halton_sequence seq(3);
    auto mat = seq.sample(10);

    CHECK(mat.rows() == 3);
    CHECK(mat.cols() == 10);

    for(int j = 0; j < 10; ++j)
    {
        for(int i = 0; i < 3; ++i)
        {
            CHECK(mat(i, j) >= 0.0);
            CHECK(mat(i, j) <= 1.0);
        }
    }
}

TEST_CASE("skip parameter offsets sequence", "[halton]")
{
    halton_sequence seq_no_skip(2);
    halton_sequence seq_skip(2, 100);

    auto pt_no_skip = seq_no_skip.next();
    auto pt_skip = seq_skip.next();

    // They must differ since skip=100 starts at index 101
    CHECK_FALSE(pt_no_skip.isApprox(pt_skip));
}

TEST_CASE("10D Halton uses all 10 primes", "[halton]")
{
    halton_sequence seq(10);
    auto pt = seq.next();

    CHECK(pt.size() == 10);
    for(int i = 0; i < 10; ++i)
    {
        CHECK(pt(i) > 0.0);
        CHECK(pt(i) < 1.0);
    }
}

TEST_CASE("reset restores sequence to beginning", "[halton]")
{
    halton_sequence seq(2);
    auto first = seq.next();
    seq.next();
    seq.next();
    seq.reset();
    auto after_reset = seq.next();

    CHECK(first.isApprox(after_reset));
}

TEST_CASE("index tracks current position", "[halton]")
{
    halton_sequence seq(2, 5);
    CHECK(seq.index() == 6);
    seq.next();
    CHECK(seq.index() == 7);
}
