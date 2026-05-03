cmake_minimum_required(VERSION 3.28)
project(argmin_integration_test)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(argmin CONFIG REQUIRED)

add_executable(integration_test main.cpp)
target_link_libraries(integration_test PRIVATE argmin::argmin)
