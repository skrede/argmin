// Random-number-freedom probe for the deterministic policies.
//
// Most policies claim that identical inputs drive an identical trajectory, and
// that claim rested entirely on the reasoning "the policy carries no RNG" --
// true, but read off the source by hand and defended by nothing. This
// translation unit includes every policy that makes that claim on those grounds,
// so the companion CTest (see tests/compile/CMakeLists.txt) can prove it: no
// <random> in the transitive include graph, and no RNG facility named in any
// argmin header the graph reaches.
//
// A policy is a header-only template, so its header is its implementation and
// scanning what the header reaches scans the whole policy.
//
// Four policies are deliberately absent, and the omissions are the interesting
// part of this list:
//
//   cmaes, isres  -- genuinely stochastic. They pull <random>, so including them
//                    here would turn this scan red. Their determinism is a real
//                    claim, but it is a seeded-reproducibility claim and a
//                    seeded reproducibility test is what gates it.
//
//   cobyla        -- carries a self-contained linear congruential generator to
//                    jitter degenerate geometry steps (a behavior inherited from
//                    the NLopt/SGJ line; see cobyla_policy.h). It is seeded from
//                    the problem dimensions and so is perfectly reproducible,
//                    but it is still a generator: "carries no RNG" is false for
//                    it, and this scan must not be made to say otherwise. The
//                    token scan would reject it on lcg_rand, which is exactly
//                    the behavior wanted -- the omission is enforced, not
//                    merely intended.
//
//   projected_gn  -- carries no RNG either and would pass, but its cell cites a
//                    run-to-run bit-exactness test, which is strictly stronger
//                    than the absence of an RNG. Listing it here would add a
//                    weaker citation to a claim that already has a better one.
//
// So this list is exactly the set of policies whose determinism claim rests on
// this scan and nothing else. A policy added to the library belongs here unless
// one of the four reasons above applies to it.

#include "argmin/solver/projected_gradient_gn_policy.h"
#include "argmin/solver/augmented_lagrangian_policy.h"
#include "argmin/solver/ccsa_quadratic_policy.h"
#include "argmin/solver/filter_nw_sqp_policy.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/filter_trsqp_policy.h"
#include "argmin/solver/byrd_lbfgsb_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/gcmma_policy.h"
#include "argmin/solver/mma_policy.h"
#include "argmin/solver/lm_policy.h"

int main() { return 0; }
