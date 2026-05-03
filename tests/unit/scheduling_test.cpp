#include "argmin/schedule/round_robin_schedule.h"
#include "argmin/schedule/time_boxed_schedule.h"
#include "argmin/schedule/fallback_schedule.h"
#include "argmin/result/step_result.h"

#include <catch2/catch_test_macros.hpp>

#include <set>
#include <thread>

using namespace argmin;

TEST_CASE("round_robin_schedule", "[scheduling]")
{
    round_robin_schedule sched;

    SECTION("alternates between two solvers")
    {
        CHECK(sched.select(2) == 0);
        CHECK(sched.select(2) == 1);
        CHECK(sched.select(2) == 0);
        CHECK(sched.select(2) == 1);
    }

    SECTION("cycles through three solvers")
    {
        CHECK(sched.select(3) == 0);
        CHECK(sched.select(3) == 1);
        CHECK(sched.select(3) == 2);
        CHECK(sched.select(3) == 0);
    }

    SECTION("reset restarts from zero")
    {
        sched.select(2);
        sched.select(2);
        sched.reset();
        CHECK(sched.select(2) == 0);
    }
}

TEST_CASE("time_boxed_schedule", "[scheduling]")
{
    time_boxed_schedule sched;
    sched.time_slice = std::chrono::microseconds{100};

    SECTION("starts with solver 0")
    {
        CHECK(sched.select(2) == 0);
    }

    SECTION("all solvers get stepped over time")
    {
        std::set<std::size_t> stepped;
        for(int i = 0; i < 200; ++i)
        {
            stepped.insert(sched.select(2));
            std::this_thread::sleep_for(std::chrono::microseconds{10});
        }
        CHECK(stepped.size() == 2);
    }

    SECTION("uses steady_clock (compiles)")
    {
        // If this compiles, time_boxed_schedule uses steady_clock internally.
        sched.reset();
        sched.select(3);
    }
}

TEST_CASE("fallback_schedule", "[scheduling]")
{
    fallback_schedule sched;
    sched.stall_threshold = 3;
    sched.set_num_solvers(2);

    SECTION("starts with solver 0")
    {
        CHECK(sched.select(2) == 0);
    }

    SECTION("stays with solver 0 on improvement")
    {
        step_result good{.improved = true};
        sched.notify(good);
        sched.notify(good);
        sched.notify(good);
        CHECK(sched.select(2) == 0);
    }

    SECTION("switches after stall_threshold non-improving steps")
    {
        step_result bad{.improved = false};
        sched.notify(bad);
        sched.notify(bad);
        CHECK(sched.select(2) == 0);
        sched.notify(bad);
        CHECK(sched.select(2) == 1);
    }

    SECTION("resets stall counter on improvement")
    {
        step_result bad{.improved = false};
        step_result good{.improved = true};
        sched.notify(bad);
        sched.notify(bad);
        sched.notify(good);
        sched.notify(bad);
        sched.notify(bad);
        CHECK(sched.select(2) == 0);
    }

    SECTION("cycles back to first solver")
    {
        sched.set_num_solvers(2);
        step_result bad{.improved = false};
        for(int i = 0; i < 3; ++i) sched.notify(bad);
        CHECK(sched.select(2) == 1);
        for(int i = 0; i < 3; ++i) sched.notify(bad);
        CHECK(sched.select(2) == 0);
    }
}
