set(SCAN_NAME "the build does not force verbose makefiles on itself or a parent project")
set(SCAN_FILES "CMakeLists.txt;lib/argmin/CMakeLists.txt;cmake/*.cmake")
set(SCAN_REGEX "set\\(CMAKE_VERBOSE_MAKEFILE ON\\)")
