set(SCAN_NAME "the active-set multiplier solve reuses the triangular factor instead of a pivoted QR per iteration")
set(SCAN_FILES "lib/argmin/include/argmin/detail/active_set_qp.h")
set(SCAN_REGEX "qr_rr_\\.compute\\(AY_pad_\\)")
