set(SCAN_NAME "an installed-package consumer test builds at C++20")
set(SCAN_FILES "tests/integration/CMakeLists.txt")
set(SCAN_REGEX "CMAKE_CXX_STANDARD(:STRING)?[ \t=]+20")
set(SCAN_ABSENT TRUE)
