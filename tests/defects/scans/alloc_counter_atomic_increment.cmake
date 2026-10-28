set(SCAN_NAME "the allocation counter links on a core without atomic read-modify-write instructions")
set(SCAN_FILES "lib/argmin/include/argmin/detail/diagnostics/alloc_counter.h")
set(SCAN_REGEX "g_eigen_malloc_count\\.fetch_add\\(")
