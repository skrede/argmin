set(SCAN_NAME "no implementation-defined standard distribution draws on a seeded path")
set(SCAN_FILES "lib/argmin/include/argmin/*.h")
set(SCAN_REGEX "std::(uniform_real_distribution|normal_distribution)<[A-Za-z_:]+>[ \t]+[A-Za-z_]|std::generate_canonical[ \t]*<")
