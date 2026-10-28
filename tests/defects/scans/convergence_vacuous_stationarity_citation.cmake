set(SCAN_NAME "the stationarity threshold has a numeric basis rather than a definition that names no tolerance")
set(SCAN_FILES "lib/argmin/include/argmin/options/convergence_defaults.h;lib/argmin/include/argmin/solver/convergence.h")
set(SCAN_REGEX "\n[ \t]*[^/ \t\r\n][^\r\n]*\r?\n[ \t]*static constexpr double stationarity_threshold = 1e-8;[ \t]*\r?\n|1e-8 matches N&W 2e Definition[ \t\r\n/]*12\\.1's typical")
