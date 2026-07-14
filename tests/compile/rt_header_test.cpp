// Compile-only coverage for the real-time-safe umbrella header.
//
// rt.h re-exports the wall-clock-free solver surface and carries no runtime
// bodies of its own, but it was previously never included by any translation
// unit -- so its re-exports and the definition-site concept boundary it relies
// on were never actually compiled anywhere. Including it here forces that
// compilation, registering the header with the build (and coverage) instead of
// leaving it silently uninstantiated.

#include "argmin/rt.h"

int main() { return 0; }
