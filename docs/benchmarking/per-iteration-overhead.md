# Per-iteration overhead

This document records what a `perf`-instrumented investigation of argmin's
per-iteration cost actually found on the Linux bench host, and it supersedes an
older prior that a large per-iteration overhead separated argmin from the
comparator libraries. On the current tree that premise does not survive
measurement: at the same algorithm argmin wins or ties the wall-time median
against nlopt across the solver families, and the residual per-iteration
differences that remain are already overcome by an iteration-count advantage at
the wall.

All numbers below are measured on this host and every one carries a command that
reproduces it. Where a figure differs from an earlier estimate, the measured
figure is the one stated.

## Host and toolchain

- AMD Ryzen 7 5800X3D, Linux, `cpufreq` governor `performance`,
  `perf_event_paranoid = 2` (so the userspace hardware instruction counter
  arms).
- gcc 16.1.1, Release, `-march=native -fno-math-errno -fno-trapping-math
  -DNDEBUG`; Eigen 3.4.1 via FetchContent.
- Comparator: nlopt 2.11.0 (dlib, ipopt and kthohr/optim are also linked; ceres
  and libcmaes are absent on this host and do not affect any statement here).
- Single-threaded, pinned with `taskset` to reduce scheduler noise. Wall times
  are quantized to the microsecond, so single-digit-microsecond cells carry
  visible rounding; the instruction counter is the deterministic, portable
  measure and is reported where the comparison is argmin-internal.

## The stale premise

An earlier reading put argmin an order of magnitude or more behind nlopt in
per-iteration cost, anchored on two cells: a Rosenbrock L-BFGS-B cell at roughly
24 microseconds per iteration against nlopt's 0.3, and a Himmelblau cell at 11.9
against 0.2 — ratios near 60x to 80x. Those anchors predate the optimization
work of the last several releases. Re-measured on the current tree they are
smaller by close to two orders of magnitude.

## Same-algorithm anchors (measured)

L-BFGS-B on the two anchor problems, one seed, from the publication run:

| cell | argmin | nlopt | per-iteration ratio |
|------|--------|-------|---------------------|
| rosenbrock_2, L-BFGS-B | 22 us / 22 it = 1.00 us/it | 7 us / 18 it = 0.39 us/it | ~2.6x |
| himmelblau, L-BFGS-B | 3 us / 10 it = 0.30 us/it | 4 us / 13 it = 0.31 us/it | ~1.0x |

The Himmelblau cell lands at parity within the microsecond rounding of a
three-versus-four-microsecond wall; the Rosenbrock cell is the robust anchor at
roughly 2.6x per iteration. Both sit two orders of magnitude below the stale
60x-80x anchors: the large per-iteration gap the older prior described is closed
on this tree.

The in-tree `micro_lbfgsb` whole-solve benchmark corroborates from the other
direction — argmin against nlopt over ten thousand repetitions in one process:

- wide bounds, all-free two-loop fast path: 21.8 us vs 19.1 us, ratio 1.1x;
- tight bounds, generalized-Cauchy-point branch: 11.9 us vs 6.8 us, ratio 1.7x.

## Same-algorithm wall medians (measured)

Median wall-time ratio (argmin / comparator) over every common problem, one
seed, from the same publication run. A ratio below one means argmin is faster at
the median:

| pair | n | wall-median ratio | per-iteration-median ratio |
|------|---|-------------------|----------------------------|
| lbfgsb vs nlopt_lbfgs | 13 | 0.57 | 0.83 |
| byrd_lbfgsb vs nlopt_lbfgs | 13 | 0.50 | 0.40 |
| cobyla vs nlopt_cobyla | 27 | 0.49 | 1.06 |
| bobyqa vs nlopt_bobyqa | 5 | 0.52 | 0.97 |
| mma vs nlopt_mma | 13 | 0.46 | 4.70 |
| kraft_slsqp_accurate vs nlopt_slsqp | 14 | 1.15 | 3.11 |
| augmented_lagrangian vs nlopt_auglag | 14 | 0.33 | 5.51 |

argmin wins the wall-time median on every same-algorithm pair except SLSQP,
where it is near parity (slightly above at 1.15). The worst individual wall
cells in the maxima are cap-exhaustion and stall artifacts, not per-iteration
overhead, and belong to convergence behavior rather than to this investigation.

**Conclusion for the broad nonlinear path: no broad solver change is justified
by the current profile.** The per-iteration gap the older prior named no longer
exists on this tree, so the broad "fix the per-iteration overhead" scope is
retired as unjustified-by-measurement. What remains are the localized quadratic
program levers below and a handful of low-payoff candidates, all of which must
clear an on-host A/B before any code moves.

## Residual per-iteration gaps

The constrained families still show a per-iteration disadvantage — SLSQP about
3x, MMA about 4.7x, augmented Lagrangian about 5.3x-5.5x at the median — but
each of those families already wins the wall-time median (or ties, for SLSQP)
because argmin reaches the stopping criterion in fewer iterations. A per-solve
consumer sees the wall, not the per-iteration cost, so these residuals do not
motivate work on their own. The SLSQP residual accuracy is a separate, already
tracked concern and is out of scope here.

## The quadratic-program polish path (measured)

The one place a localized per-step lever is visibly available is the sparse
operator-splitting QP polish. The polish re-analyzes and re-factorizes a freshly
shaped reduced KKT system on every resolve by construction, so its cost is a
per-call constant. The `micro_sparse_admm_polish` benchmark isolates that cost
on the control-shaped linear-MPC family by differencing a polish-on run against a
polish-off run over an identical problem sequence, in two warm-resolve regimes:

| cell | pertIC polish share | rollout polish share | pertIC hit rate |
|------|---------------------|----------------------|-----------------|
| ni=2 N=10 | 39.5% | 14.2% | 63.6% |
| ni=2 N=20 | 39.0% | 11.7% | 12.1% |
| ni=3 N=20 | 39.7% | 10.1% | 6.1% |
| ni=3 N=30 | 39.8% | 11.9% | 12.1% |

The polish is 39-40% of per-step cost at the termination-check iteration floor
(the pertIC regime, which stays at 25 iterations) and 10-14% under the closed-
loop rollout, where a moving reference balloons the ADMM iteration count so the
kernel dominates and the polish share falls. These shares replace an older
headline that put the polish at the overwhelming majority of per-step cost; that
figure did not reproduce, and the driver behind it is gone, so it is stated here
only to retire it.

A pattern-gated reuse of the polish factorization — skipping the re-analysis and
re-factorization when the active-set pattern, the delta and the pose are all
unchanged from the previous step — is implemented and A/B-measured on this host
by toggling it in-process over an identical problem sequence (`reuse-on` against
`reuse-off` in the benchmark output). The measured saving is strongly workload-
and size-dependent. On the smallest, most stable cell (ni=2 N=10 under gentle
initial-condition noise, where consecutive steps share an active-set pattern most
often, at a 64% consecutive-pattern rate) the reuse cuts **30.7% of per-step
cost** — 1.24M down to 0.86M userspace instructions per warm resolve. On the
larger perturbed-initial-condition cells the pre-polish active set churns enough
that the reuse almost never fires and the saving is **within 0.03% of zero**;
under the saturated tracking rollout the hit rate is zero on every cell and the
saving is **within 0.002% of zero** — the conservative element-wise pattern
compare on the miss path is effectively free and never a net per-step regression.
So this is a conditional lever that pays materially only in a small-and-stable
regime and is a measured no-op — not a regression — everywhere else on this
family; it is not a dominant per-step lever. The reuse is bit-identical to the
re-analyzing path by construction (P, A and delta are frozen across a resolve, so
an unchanged active set yields a bit-identical reduced KKT), which a committed
correctness test pins on both a stable-pattern and a pattern-churning sequence.

## Native versus Python-binding per-solve overhead (measured)

The per-iteration question extends to the Python binding: a per-solve call
through the binding pays a fixed marshalling and dispatch cost on top of the
native solve. Measured on this host on the small reference QP the binding test
suite uses (`min 0.5 xᵀPx + qᵀx` with `P = [[4,1],[1,2]]`, `q = [1,1]`, the
equality `x0 + x1 = 1` and box `0 <= xi <= 0.7`), median wall over 20,000 warm
`SparseAdmmQpSolver.solve` calls:

- native (direct C++ call): 6.0 us per solve;
- Python binding (`build/python` extension, scipy CSC in, numpy out): 16.5 us
  per solve.

The binding therefore adds about **10.5 us of fixed per-solve overhead, a factor
of ~2.7x on a solve this small.** That overhead is dominated by the per-call
dispatch and the array marshalling (scipy CSC to Eigen, numpy result
construction, the nanobind boundary and the GIL), and it is largely independent
of problem size, so it amortizes on the larger MPC-scale solves a real consumer
runs, where the native solve alone is already tens of microseconds. A Python
consumer that solves many small problems in a tight loop is the case where this
fixed cost matters; a consumer that makes one larger solve per control step is
not. No broad binding change is justified by this measurement; deeper
binding-path profiling is a conditional follow-up gated on a real consumer's
per-call cost proving material.

## Reproduction

All commands run from the repository root against a Release bench tree built as:

```
cmake -S . -B build/bench -DCMAKE_BUILD_TYPE=Release -DARGMIN_BUILD_TESTS=ON \
  -DARGMIN_BUILD_BENCHMARKS=ON -DARGMIN_BENCH_LIBCMAES=OFF -DARGMIN_CMAKE_FETCH_DEPS=ON
cmake --build build/bench -j4
```

Same-algorithm anchors and wall medians (the publication run, then read the
`solver`, `problem`, `solver_iters`, `wall_time_us` columns of the summary):

```
build/bench/benchmarks/publish_bench --output-dir <scratch> --seed-start 42 --seed-count 1
```

Whole-solve L-BFGS-B corroboration:

```
taskset -c 3 build/bench/benchmarks/micro_lbfgsb
```

QP polish per-step share and active-set hit rate:

```
taskset -c 3 build/bench/benchmarks/micro_sparse_admm_polish
```

Native versus Python-binding per-solve overhead. The native side is a
standalone driver compiled against the headers; the binding side times the same
solve through the built `build/python` extension.

Native driver (`native_qp_timing.cpp`):

```cpp
#include "argmin/qp/sparse_admm_qp.h"
#include "argmin/options/sparse_qp_options.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <cstdio>
#include <chrono>
#include <vector>
#include <algorithm>

int main()
{
    using sparse = Eigen::SparseMatrix<double, Eigen::ColMajor>;
    Eigen::MatrixXd Pd(2, 2); Pd << 4.0, 1.0, 1.0, 2.0;
    Eigen::MatrixXd Ad(3, 2); Ad << 1.0, 1.0, 1.0, 0.0, 0.0, 1.0;
    sparse P = Pd.sparseView(); sparse A = Ad.sparseView();
    P.makeCompressed(); A.makeCompressed();
    Eigen::VectorXd q(2); q << 1.0, 1.0;
    Eigen::VectorXd l(3); l << 1.0, 0.0, 0.0;
    Eigen::VectorXd u(3); u << 1.0, 0.7, 0.7;
    argmin::sparse_admm_qp_solver<double> solver;
    argmin::sparse_qp_options opts;
    for(int i = 0; i < 2000; ++i) (void)solver.solve(P, q, A, l, u, opts);
    std::vector<double> t; t.reserve(20000);
    for(int i = 0; i < 20000; ++i)
    {
        auto a = std::chrono::steady_clock::now();
        auto r = solver.solve(P, q, A, l, u, opts); (void)r;
        auto b = std::chrono::steady_clock::now();
        t.push_back(std::chrono::duration<double, std::micro>(b - a).count());
    }
    std::sort(t.begin(), t.end());
    std::printf("native solve median: %.3f us\n", t[t.size() / 2]);
    return 0;
}
```

```
g++ -std=c++20 -O2 -march=native -fno-math-errno -fno-trapping-math -DNDEBUG \
  -I lib/argmin/include -isystem build/bench/_deps/eigen3-src \
  native_qp_timing.cpp -o native_qp_timing
taskset -c 3 ./native_qp_timing
```

Binding driver (`bind_qp_timing.py`), timed through the built extension:

```python
import time
import numpy as np
import scipy.sparse as sp
import argmin

P = sp.csc_matrix(np.array([[4.0, 1.0], [1.0, 2.0]]))
A = sp.csc_matrix(np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]))
q = np.array([1.0, 1.0]); l = np.array([1.0, 0.0, 0.0]); u = np.array([1.0, 0.7, 0.7])
opts = argmin.SparseQpOptions()
solver = argmin.SparseAdmmQpSolver()
for _ in range(2000): solver.solve(P, q, A, l, u, opts)
t = []
for _ in range(20000):
    a = time.perf_counter(); solver.solve(P, q, A, l, u, opts); b = time.perf_counter()
    t.append((b - a) * 1e6)
t.sort(); print(f"binding solve median: {t[len(t) // 2]:.3f} us")
```

```
taskset -c 3 env PYTHONPATH=python .venv/bin/python bind_qp_timing.py
```
