cmake_minimum_required(VERSION 3.28)
project(argmin_transitive_parent_consumer)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(transitive_parent CONFIG REQUIRED)

add_executable(transitive_consumer main.cpp)
target_link_libraries(transitive_consumer PRIVATE transitive_parent::transitive_parent)
