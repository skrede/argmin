set(SCAN_NAME "library headers use fixed-width integers for loop counters and index narrowing")
set(SCAN_FILES "lib/argmin/include/argmin/*.h")
set(SCAN_REGEX "static_cast<int>|for[ \t]*\\([ \t]*int[ \t]")
