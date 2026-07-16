// Wall-clock-freedom probe for the solver policies.
//
// The companion probe (runner_chrono_freedom_test.cpp) covers the two
// real-time embeddable drivers, and deliberately includes nothing else so that
// its scan answers for the driver headers alone. That scoping is what makes the
// driver claim precise, and it is kept. But it also means the drivers were the
// only surfaces a scan ever answered for: every policy's wall-clock claim rested
// on reading the source rather than on a check. This translation unit is the
// other half -- it includes every policy header, so the same toolchain-
// authoritative dependency scan answers for the policies too.
//
// A policy is a header-only template: its entire implementation is the header,
// so a scan of what the header pulls in is a scan of the whole policy. The scan
// walks the compiler's transitive include graph, not the literal include lines,
// so a policy that reads no clock directly but reaches one through a dependency
// is caught.
//
// The union is deliberate. <chrono> reaching any one of these headers fails the
// scan, and a scan that passes has proved the property for all of them, because
// each policy's include graph is a subgraph of this one.
//
// Every policy is listed, including the two that carry an RNG: wall-clock
// freedom is unrelated to determinism, and a policy left out of this list would
// silently have no scan behind its claim.

#include "argmin/solver/projected_gradient_gn_policy.h"
#include "argmin/solver/augmented_lagrangian_policy.h"
#include "argmin/solver/ccsa_quadratic_policy.h"
#include "argmin/solver/filter_nw_sqp_policy.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/filter_trsqp_policy.h"
#include "argmin/solver/projected_gn_policy.h"
#include "argmin/solver/byrd_lbfgsb_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/cmaes_policy.h"
#include "argmin/solver/gcmma_policy.h"
#include "argmin/solver/isres_policy.h"
#include "argmin/solver/mma_policy.h"
#include "argmin/solver/lm_policy.h"

int main() { return 0; }
