set(SCAN_NAME "LSI keeps the triangular E unpivoted and LSEI applies Q_c without materializing it")
set(SCAN_FILES "lib/argmin/include/argmin/detail/lsi.h;lib/argmin/include/argmin/detail/lsei.h")
set(SCAN_REGEX "ws\\.qr\\.compute\\(ws\\.E_dyn\\)|ws\\.Qc\\.setIdentity\\(\\)")
