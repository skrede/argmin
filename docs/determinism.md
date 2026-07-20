# Determinism: what is guaranteed, and what is deliberately not

"Deterministic" is not one property. It is a family of them with different
scopes: the same binary re-run on the same machine, two different instantiations
of the same math compiled into different machine code, or two different
architectures. argmin guarantees the first exactly, the second in a precise and
limited sense, and the third not at all — on purpose.

The tiers below are **not degrees of confidence**. They are different properties
with different scopes, and each one is either bit-exact or bounded; none of them
is "probably fine". Conflating them is how a library ends up implying a
bit-identity it never had — a reader who sees the word "deterministic" and takes
it to mean "the same answer on my workstation and on my board" has been misled by
a word rather than informed by one. So every tier below states its scope, and
names the test that goes red when the tier stops being true.

## The tiers

<!-- BEGIN GENERATED: tiers -->
| claim | guaranteed? | evidence |
|---|---|---|
| run-to-run and post-`reset()` bit-exactness | **yes** *(gated)* (on every platform, on both dimension axes, and across `reset()`) | projected_gn_policy: trajectory is run-to-run deterministic on a fixed target (exact ==, not a tolerance); gcc-14; clang-18; build (Xcode 15.4); build (Xcode 16.2); build (Windows) |
| cross-instantiation algorithmic-decision identity | **yes** *(gated)* (identical accept, reject, and termination decisions, pinned as exact step-count equality) | projected_gn_policy: fixed-N path reproduces the dynamic-N trajectory (the step-count REQUIRE); gcc-14; clang-18; build (Xcode 15.4); build (Xcode 16.2); build (Windows) |
| cross-instantiation numeric agreement | **yes** *(gated)* (within the per-element tolerance stated below and under a swept aggregate spread bound, wherever the test runs; not a comparison between architectures) | projected_gn_policy: fixed-N path reproduces the dynamic-N trajectory (the per-element tolerance check and the aggregate spread bound); gcc-14; clang-18; build (Xcode 15.4); build (Xcode 16.2); build (Windows) |
| cross-architecture or cross-instantiation bit-identity | **not claimed** (a deliberate anti-feature, not a gap) | none, and there will not be one: this is the claim the tiers above exist to not make, and its cost is stated below |
<!-- END GENERATED: tiers -->

This table is generated from `scripts/rt_matrix_tiers.json` — edit the data file,
not the table. Every test and job it names is resolved against live test
registration and against the workflows that run them, on the same footing as the
[RT-safety matrix](rt-safety-matrix.md): a tier citing a test that has been
renamed or deleted fails the check rather than quietly decaying back into prose.
**yes** *(gated)* means a named test goes red when the claim stops holding.

## Run-to-run and post-`reset()` bit-exactness

On a fixed target the same binary executing the same input runs the same
instruction stream, so the trajectory it produces must repeat bit-for-bit. There
is no tolerance here and no "approximately": the proving test compares every
recorded quantity of every step with exact `==`, not with a tolerance matcher.

This is the tier that matters most for an embedded or real-time integrator,
because it is the one that says: on the target you ship, the solver you shipped
answers the same way every time.

Two things are checked beyond a plain re-run. The test drives the dynamic-N and
the fixed-N instantiation on separate axes, so neither path receives the
guarantee on the other's evidence. And it exercises `reset()` — a full solve is
drained, the same solver instance is reset, the solve is drained again and
compared against the first run — so state leaking across solver reuse fails the
same exact assertion, instead of surfacing later as a run somebody cannot
reproduce.

The tier holds on every platform. The test carries no label and is filtered out
nowhere, so it runs in the unqualified test invocation of the Linux, macOS
(arm64), and Windows jobs the table names.

## Cross-instantiation algorithmic-decision identity

Compiling the same math at a compile-time-known dimension produces different
machine code from the dynamic-dimension path — unrolled fixed-size kernels rather
than dynamic loops. They are two different programs computing the same function,
and they round differently in the last bits.

What must not differ is the *decisions*. Which steps are accepted, which are
rejected, and when the solve terminates are discrete facts rather than
floating-point ones, and they are pinned with exact equality on the step count. A
decision-level disagreement trips there regardless of any numeric tolerance,
which is exactly why it is pinned separately: a tolerance may absorb a rounding
difference, but it must never be allowed to absorb the two paths taking different
branches.

The practical consequence: moving a policy from a dynamic dimension to a fixed
one to buy the allocation-free path (see
[Embedded cross-compiling](embedded.md)) gives you the same algorithm making the
same choices — not a differently-behaved solver that also happens to converge.

## Cross-instantiation numeric agreement

The numbers those identical decisions produce are *not* identical.
Instruction-level rounding — fused multiply-add contraction, which a compiler may
apply differently to unrolled and to looped code — perturbs the two trajectories
at the 1-ULP level from the first step, and a singular nonlinearity amplifies
that geometrically over the solve. What is guaranteed is agreement within a
stated per-element tolerance:

```cpp
const double tol = 1e-8 + 1e-8 * scale;   // scale = max(|dyn|, |fix|)
```

The tolerance is hybrid on purpose, and both legs apply to every element. The
absolute leg covers the terminal regime, where the quantities approach zero and a
relative comparison is meaningless. The relative leg covers the O(1) early
quantities, where a purely absolute bound would have to be loosened until it
caught nothing. The check applies to every recorded quantity of every step rather
than to a summary of them.

**The scope of this tier, stated precisely.** The check compares two
*instantiations* on the machine running it. It runs on x86_64 Linux, arm64 macOS,
and x86_64 Windows, so the tolerance is enforced on each of those architectures.
It does **not** compare a trajectory produced on one architecture against a
trajectory produced on another — and neither does anything else here. This page
therefore does not claim a cross-architecture numeric bound, and you should not
read one into it.

The architecture is nevertheless why this tier is a tolerance rather than an
equality: contraction behavior is architecture-dependent, so how far the two
instantiations drift apart depends on where you run them. On the platforms this
test runs on, the trajectory-wide maximum spread `max |dyn − fix|` has been
measured at `2.05887e-10` on arm64 (both Xcode legs of continuous integration,
identical), and exactly `0` on both x86_64 legs and on x86_64 Windows — the
spread is arm64-specific, attributable to FMA contraction. That magnitude is
recorded on every run, and it is now bounded: alongside the per-element check,
the aggregate maximum is pinned under `1.5e-9`, a single portable constant
placed roughly a factor of seven above the worst measured spread and a factor of
seven below the `1e-8` per-element wall. Drift toward that wall therefore trips
the aggregate bound loudly first, rather than accumulating silently under the
per-element tolerance. The relative aggregate is left unbounded on purpose: it is
structurally confined to `[0, 2]` regardless of correctness, so a bound on it
would be either vacuous or brittle; it is retained as a near-zero diagnostic
only.

## The anti-feature: bit-identity across architectures and instantiations

This is **never** claimed. It is not a limitation to apologize for, not a gap to
be closed later, and not an item waiting on someone to measure it. It is a
deliberate engineering choice, and it trades directly against the guarantees the
rest of this documentation makes.

Getting bit-identity would mean forcing every architecture and every
instantiation to emit the same floating-point operations in the same order. In
practice that means two things: disabling fused multiply-add contraction
library-wide, and refusing to let the compiler specialize the fixed-N path into
the unrolled code that makes it fast and allocation-free. Both of those *are* the
real-time pillar. The fixed-N path exists precisely because a compile-time
dimension lets the compiler emit different — and better — code; demanding that it
produce numerically identical output to the dynamic path is demanding that it
emit the same code, which is demanding that it not do the job it was introduced
to do.

So the trade is explicit, and argmin takes the far side of it: a fast,
allocation-free, bounded fixed-N path on the target, rather than a trajectory you
could reproduce bit-for-bit on a different machine. What you get instead is the
combination the tiers above describe, which is what a real-time integrator
actually needs — on the target you ship, the solver is bit-exact run to run, and
when you move between the dynamic and the fixed-N path, the algorithm makes
identical decisions and the numbers agree within a bounded tolerance.

Concretely, what follows from the choice:

- Do not pin a golden trajectory captured on one architecture and assert it
  bit-for-bit on another. Nothing here guarantees that it holds, so when it
  breaks, the break is correct behavior rather than a regression.
- Do not build a downstream expectation, test, or acceptance criterion on
  cross-architecture bit-identity. It is not supported, and it will not become
  supported.
- Do compare across architectures at a tolerance you choose and state — the way
  the cross-instantiation check does.

## See also

- [RT-safety matrix](rt-safety-matrix.md) — the per-policy
  `deterministic(seeded)?` column, and the rest of the real-time claims.
- [Embedded cross-compiling](embedded.md) — the fixed-N, allocation-free path
  this page's anti-feature protects.
