// Wall-clock-freedom probe for the step-budget / passive-step surface.
//
// step_budget_solver and stepper are the real-time embeddable drivers: a caller
// that budgets on iteration count (or owns its own scheduling) must be able to
// use them without any dependency on <chrono>. Wall-clock confinement is a
// structural property here, not a coding-discipline convention -- only the
// time-budget drivers may pull <chrono>.
//
// This translation unit includes ONLY those two headers. The companion CTest
// (see tests/compile/CMakeLists.txt) runs the compiler's own dependency scan
// over this file and fails if <chrono> appears anywhere in the transitive
// include graph -- a toolchain-authoritative check of the real include set, not
// an implementation-defined guard-macro proxy. Including the headers (rather
// than instantiating a concrete policy) is deliberate: it scopes the scan to
// exactly what the runner headers pull in, independent of any policy's own
// includes. That scoping is the point and is kept -- it is what lets a failure
// here name the drivers rather than some policy that happened to be linked.
//
// The policies are not left unscanned for it: policy_chrono_freedom_test.cpp
// includes every policy header and the same scan answers for that graph. The
// two probes are separate so that each claim fails on its own evidence.

#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/stepper.h"

int main() { return 0; }
