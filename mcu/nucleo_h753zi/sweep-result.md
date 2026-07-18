# EIGEN_STACK_ALLOCATION_LIMIT live ladder-sweep (NUCLEO-H753ZI probe image)

**Date:** 2026-07-18
**Toolchain:** arm-none-eabi-gcc **15.2.Rel1** (`15.2.1 20251203`, Arm GNU Toolchain, Build arm-15.86)
**Image:** `nucleo_h753zi_probe.elf` (`mcu -B build-arm-nucleo`, default firmware mode)
**Eigen:** 3.4.0 (FetchContent), `EIGEN_ALLOCA=__builtin_alloca`, `EIGEN_DONT_VECTORIZE`,
`EIGEN_MAX_ALIGN_BYTES=8` (`ARGMIN_EIGEN_MCU_DEFS`, `mcu/CMakeLists.txt:66-69`)

An earlier sweep was vacuous: `EIGEN_ALLOCA` was undefined, so the limit was dead code and every
rung produced a byte-identical binary. With `EIGEN_ALLOCA` defined the limit is live. The proof
below establishes that on two independent axes, and the measurement shows why a single
alloca→heap "straddle" rung is physically unbuildable on this image.

---

## Machine-checkable positive control

```
COMPILE_FLOOR_CONTROL: 256 compile-fail static_assert(512 <= 256)
FLOOR_RUNG_SHA256: d04944f4e06d316972deaf01f096137d61845586f27bc9f0b136563ccfe63953
PIN_RUNG_SHA256: 378ece5e1f986ee2f3f688a4d6be0ae4a32aae0ffdf13f917f8e489f81bf0495
```

`FLOOR_RUNG_SHA256` (limit=512, the lowest buildable rung) **differs from** `PIN_RUNG_SHA256`
(limit=8192). The `COMPILE_FLOOR_CONTROL` line records that a rung below the fixed-object floor
(256) fails to compile with a **limit-derived** static assertion — proving the limit reaches the
compiler. Together these falsify "the knob is dead code" on two independent axes.

---

## Measured stack-object sizes (the two axes the limit gates)

`EIGEN_STACK_ALLOCATION_LIMIT` gates **two** distinct object classes, and the second is easy to
overlook:

| Axis | Mechanism | Largest object | Where |
|------|-----------|----------------|-------|
| **Runtime dynamic temporary** | `size <= LIMIT ? alloca : aligned_malloc` ternary in `ei_declare_aligned_stack_constructed_variable` (`Memory.h:751`) | **223 B** (kraft's LSEI transformed-constraint triangular solve, `argmin/detail/lsi.h`; 216 B buffer + 7 B align pad) | measured host-side by the `EIGEN_ALLOCA` recording shim, `mcu/probe/measure_temporaries_main.cpp` |
| **Compile-time fixed object** | `static_assert(Size*sizeof(T) <= EIGEN_STACK_ALLOCATION_LIMIT)` in `check_static_allocation_size` (`DenseStorage.h:33`) | **512 B** (8×8 `TriangularMatrixMatrix` `SmallPanelWidth` panel buffer; also a 200 B 5×5) | surfaced by this sweep (compile failures below 512) |

Per-policy dynamic (alloca) largest temporary from the host shim (all four RT policies, unarmed
warmup + steady steps):

| Policy | Problem (fixed N) | Largest single dynamic temporary |
|--------|-------------------|----------------------------------|
| `kraft_slsqp` | hs071 (N=4) | **223 B** |
| `nw_sqp` | hs071 (N=4) | 0 B |
| `filter_nw_sqp` | hs071 (N=4) | 0 B |
| `lm` | rosenbrock_ls (N=2) | 0 B |

The largest dynamic temporary is **223 B**. Only kraft's triangular-solve path routes a temporary
through `EIGEN_ALLOCA`; the other three request none at their fixed N (the shim's correctness is
proven by kraft's non-zero reading).

**Consequence — the runtime alloca→heap flip for the 223 B temporary is unreachable in any
shippable image:** the compile floor (512 B) sits *above* the largest dynamic temporary (223 B),
so every limit low enough to spill the 223 B temporary to the heap (< 223) fails to compile first
(< 512). Every buildable image therefore keeps that temporary on the stack — a positive real-time
result, and the reason a "straddle 223 B" ladder is physically impossible.

---

## The ladder (per-rung build results)

Rungs reconfigured with `-DMCU_EIGEN_STACK_LIMIT=<rung>` (CACHE var wired to
`EIGEN_STACK_ALLOCATION_LIMIT`, `nucleo_h753zi/CMakeLists.txt:52-74`); no source edit to sweep.

| Rung (B) | Build | sha256(`.bin`) |
|----------|-------|----------------|
| 256 | **compile-fail** `OBJECT_ALLOCATED_ON_STACK_IS_TOO_BIG`, note `(512 <= 256)` | — |
| 384 | **compile-fail** note `(512 <= 384)` | — |
| 512 | OK (lowest buildable) | `d04944f4…e63953` |
| 1024 | OK | `d25bd4f4…255195` |
| 8192 (pin) | OK | `378ece5e…bf0495` |

All three buildable rungs produce **distinct** binaries. This is the opposite of the vacuous
sweep, where every rung was byte-identical.

### Methodology guard (stale-ELF trap)

A first pass appeared byte-identical because sub-512 rungs **fail to compile**, and a
`[ -f <elf> ]` existence check silently copied the *previous* successful build's stale `.elf`.
Corrected guard, applied to every rung above: **remove the `.elf` before each build AND gate the
`objcopy`/hash on the build's exit code** (not mere file existence). Only genuinely rebuilt ELFs
were hashed.

---

## Disassembly evidence (runtime alloca/heap decision tracks the limit)

The 512↔8192 binary delta is exactly the limit immediate at ~77–79 `cmp` sites — the runtime
`sizeof(T)*size <= LIMIT` alloca-vs-heap decision. Same site, only the immediate differs:

```
; --- limit=8192 build ---                     ; --- limit=512 build ---
lsls  r0, r2, #3      ; r0 = size in bytes      lsls  r0, r2, #3
cmp.w r0, #8192       ; size <= LIMIT ?         cmp.w r0, #512      ; <- swept immediate
bhi.n <heap fallback> ; > LIMIT -> aligned_malloc   bhi.n <heap fallback>
adds  r0, #8                                    adds  r0, #8
sub.w sp, sp, r0      ; else ALLOCA (SP bump)   sub.w sp, sp, r0
add   r4, sp, #8                                add   r4, sp, #8
```

Both branches are compiled in: the `bhi` target reaches an `aligned_malloc` call (253 `bl …malloc`
sites in the image) and the fall-through is the `sub sp` alloca. A vacuous build (`EIGEN_ALLOCA`
undefined) would show an *unconditional* `bl malloc` with **no** `cmp #limit` and **no** `sub sp`.
The limit is threaded through dozens of live runtime decisions; sweeping it moves every immediate.

---

## Pin: 8192 (unchanged)

The pin stays **8192 B**. It is a margin, not a tight optimum:

- **16×** the 512 B compile floor (largest fixed-size Eigen stack object).
- **~36.7×** the 223 B largest dynamic (alloca) temporary.

8192 sits well above both binding constraints; the margin swamps any codegen/toolchain delta, and
no compiling-rung hash showed sensitivity near 8192 (512, 1024, 8192 differ only in the expected
limit immediates), so the gcc-14 codegen fallback was not needed.

## arm link-gate cross-check

The `arm-no-exceptions-link` CI job builds `-DMCU_LINK_GATE_ONLY=ON` and never sets
`EIGEN_STACK_ALLOCATION_LIMIT` nor builds the nucleo image. Cross-checked locally with arm-gcc
15.2: `cmake -S mcu -B build-arm-gate -DMCU_LINK_GATE_ONLY=ON && cmake --build build-arm-gate` →
`argmin_arm_link_gate.elf` links clean, **0 undefined symbols**
(`arm-none-eabi-nm --undefined-only`).

## Image size at the pin (8192)

```
   text    data     bss     dec     hex
 376100     460   90484  467044   72064   nucleo_h753zi_probe.elf
```
Fits 2 MB flash; `.data`+`.bss` well within 128 KB DTCM (bss includes the 88 KB
`._user_heap_stack` reserve: 24 KB heap + 64 KB stack).

---

## Summary

The limit is **live** on two independent axes — compile-time fixed-object gating (256 fails,
`512<=256`) and runtime dynamic-temporary decisions (512/1024/8192 differ; `cmp #limit`/`bhi→malloc`/`sub sp`
disassembly). Largest dynamic temporary is 223 B, fixed-object compile floor is 512 B. Pin confirmed
at 8192 (16× / 36.7× margin). Link-gate clean under the CI configuration.
