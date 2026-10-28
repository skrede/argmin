set(SCAN_NAME "the time-budget drivers compute their deadline without overflowing at the maximum duration")
set(SCAN_FILES "lib/argmin/include/argmin/solver/*.h")
set(SCAN_REGEX "const auto deadline = t0 \\+ max_t;")
