#include "argmin/expected.h"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <version>
#include <utility>
#include <type_traits>

// Behavioral pins for the portable argmin::expected alias. The same call
// surface -- construction, has_value()/operator bool, operator*, value(),
// error(), value_or(), unexpect_t in-place construction, and the
// expected<void, E> specialization -- must hold whether the alias resolves to
// std::expected (C++23 and later) or to the in-library C++20 fallback. A
// feature-test static_assert pins which target is selected.

using namespace argmin;

TEST_CASE("expected carries a value", "[expected]")
{
    expected<int, std::string> e = 42;

    REQUIRE(static_cast<bool>(e));
    REQUIRE(e.has_value());
    REQUIRE(*e == 42);
    REQUIRE(e.value() == 42);
}

TEST_CASE("expected carries an error", "[expected]")
{
    expected<int, std::string> e = argmin::unexpected<std::string>(std::string("boom"));

    REQUIRE_FALSE(static_cast<bool>(e));
    REQUIRE_FALSE(e.has_value());
    REQUIRE(e.error() == "boom");
}

TEST_CASE("expected value_or returns the fallback on error", "[expected]")
{
    expected<int, std::string> good = 7;
    expected<int, std::string> bad = argmin::unexpected<std::string>(std::string("bad"));

    REQUIRE(good.value_or(-1) == 7);
    REQUIRE(bad.value_or(-1) == -1);
}

TEST_CASE("expected converting construction from U", "[expected]")
{
    expected<std::string, int> e = "hello";

    REQUIRE(e.has_value());
    REQUIRE(*e == "hello");
}

TEST_CASE("expected in-place unexpect_t construction", "[expected]")
{
    expected<int, std::string> e(unexpect, "in-place");

    REQUIRE_FALSE(e.has_value());
    REQUIRE(e.error() == "in-place");
}

TEST_CASE("expected operator-> reaches the payload", "[expected]")
{
    expected<std::string, int> e = std::string("abc");

    REQUIRE(e->size() == 3);
}

TEST_CASE("expected<void, E> models a payload-free result", "[expected]")
{
    expected<void, std::string> ok{};
    REQUIRE(ok.has_value());
    REQUIRE(static_cast<bool>(ok));

    expected<void, std::string> err = argmin::unexpected<std::string>(std::string("void-err"));
    REQUIRE_FALSE(err.has_value());
    REQUIRE(err.error() == "void-err");

    expected<void, std::string> tagged(unexpect, "tagged");
    REQUIRE_FALSE(tagged.has_value());
    REQUIRE(tagged.error() == "tagged");
}

// The feature-test switch must select std::expected when the standard library
// provides it and the in-library fallback otherwise. Pinning the identity here
// guards against a mis-wired switch silently shipping the wrong target.
#if defined(__cpp_lib_expected) && __cpp_lib_expected >= 202202L
static_assert(std::is_same_v<expected<int, int>, std::expected<int, int>>,
              "argmin::expected must resolve to std::expected when available");
#else
static_assert(!std::is_same_v<expected<int, int>, expected<int, long>>,
              "fallback expected must be a distinct in-library template");
static_assert(std::is_same_v<expected<int, int>, detail::expected<int, int>>,
              "argmin::expected must resolve to the in-library fallback at C++20");
#endif
