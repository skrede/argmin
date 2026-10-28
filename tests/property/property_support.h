#ifndef HPP_GUARD_ARGMIN_TESTS_PROPERTY_PROPERTY_SUPPORT_H
#define HPP_GUARD_ARGMIN_TESTS_PROPERTY_PROPERTY_SUPPORT_H

#include <catch2/catch_test_macros.hpp>

#include <rapidcheck.h>

#include <string>
#include <cstdlib>
#include <sstream>
#include <utility>
#include <iostream>
#include <string_view>

namespace argmin::property
{

inline bool plant_violation()
{
    const char* value = std::getenv("ARGMIN_PROPERTY_PLANT_VIOLATION");
    return value != nullptr && std::string_view(value) == "1";
}

// rc::prop from rapidcheck/catch.h checks without a test id, so RapidCheck neither prints a reproduce
// string for a failure nor honors RC_PARAMS="reproduce=..."; here the description is the id.
template <typename Testable>
void check(const std::string& description, Testable&& testable)
{
    rc::detail::TestMetadata metadata;
    metadata.id = description;
    metadata.description = description;
    const rc::detail::TestResult result = rc::detail::checkTestable(std::forward<Testable>(testable), metadata);
    std::ostringstream message;
    rc::detail::printResultMessage(result, message);
    if(result.is<rc::detail::SuccessResult>())
        std::cout << "- " << description << '\n' << message.str() << '\n';
    INFO(description << '\n' << message.str());
    CHECK(result.is<rc::detail::SuccessResult>());
}

}

#endif
