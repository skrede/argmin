set(SCAN_NAME "the umbrella headers order their includes by file-name length, then name")
set(SCAN_FILES "lib/argmin/include/argmin/line_search.h;lib/argmin/include/argmin/schedule.h")
set(SCAN_REGEX "line_search/result\\.h\"\r?\n#include \"argmin/line_search/options\\.h\"\r?\n#include \"argmin/line_search/armijo\\.h\"|schedule/round_robin_schedule\\.h\"\r?\n#include \"argmin/schedule/time_boxed_schedule\\.h\"\r?\n#include \"argmin/schedule/fallback_schedule\\.h\"")
