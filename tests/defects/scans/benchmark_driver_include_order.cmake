set(SCAN_NAME "the benchmark driver includes no project header after a standard header")
set(SCAN_FILES "benchmarks/benchmark_driver.cpp")
set(SCAN_REGEX "#include <[^>\r\n]+>\r?\n#include \"")
