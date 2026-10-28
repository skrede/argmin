set(SCAN_NAME "the compact L-BFGS middle matrix is factored once per push, not on every product")
set(SCAN_FILES "lib/argmin/include/argmin/detail/compact_lbfgs.h")
set(SCAN_REGEX "lu_\\.compute\\(M_active_\\);[ \t\r\n]+dst = lu_\\.solve\\(rhs\\);")
