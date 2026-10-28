set(SCAN_NAME "the documentation states that fast-math disables the non-finite input guards")
set(SCAN_FILES "README.md;docs/*.md")
set(SCAN_REGEX "-ffast-math|-ffinite-math-only")
set(SCAN_ABSENT TRUE)
