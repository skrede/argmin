set(SCAN_NAME "the KKT multiplier recovery bounds the active-row count by its workspace capacity before it resizes")
set(SCAN_FILES "lib/argmin/include/argmin/detail/lagrangian.h")
set(SCAN_REGEX "if\\(m_active == 0\\)[ \t\r\n]+return;[^\r\n]*[ \t\r\n]+(//[^\r\n]*[ \t\r\n]+)*if\\(ws\\.A_active\\.rows\\(\\) != m_active")
