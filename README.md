# liepp

![CI](https://github.com/skrede/liepp/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)

C++ Lie group and kinematics library for robotics.

## Features

- **Lie groups:** SO(2), SE(2), SO(3), SE(3) with exp/log, adjoint, coadjoint, left/right Jacobians
- **Product of Exponentials kinematics:** screw-theory-based FK with compile-time unrolled PoE for 1-7 DOF chains
- **Space and body Jacobians:** reusing cached FK intermediate products for zero redundant computation
- **Compile-time frame safety:** `transform<From, To>` and `rotation<From, To>` wrappers with structural type checking
- **Policy-based IK solvers:** DLS, Levenberg-Marquardt, and SQP steppers composable with racing and fallback schedulers
- **Header-only:** no compiled library artifacts, Eigen as the sole required dependency

## Quick Install

### CMake FetchContent (recommended)

```cmake
include(FetchContent)
FetchContent_Declare(
    liepp
    GIT_REPOSITORY https://github.com/skrede/liepp.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(liepp)

target_link_libraries(my_app PRIVATE liepp::liepp)
```

This pulls liepp and its Eigen dependency automatically. No manual installation required.

### find_package

```cmake
find_package(liepp CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE liepp::liepp)
```

## Requirements

- C++23 compiler: GCC 14+, Clang 18+, MSVC 17.10+
- CMake 3.28+
- Eigen 3.4+ (auto-fetched via FetchContent)

## Documentation

- [Getting Started](docs/getting-started.md) -- zero to compiling in 5 minutes
- [Documentation Index](docs/README.md) -- API reference, background theory, guides
- [Examples](examples/) -- runnable programs for every feature area

## License

MIT
