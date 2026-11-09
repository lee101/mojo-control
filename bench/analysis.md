# mojo-control — ANALYZE: measured hotspot map

Phase: **ANALYZE** (no algorithm changes; `src/` and `tests/` untouched).
Every number below was measured in this session on this host, best-of-N wall
clock (`time.perf_counter`, best-of-7 unless stated), under the repo's
machine-wide benchmark lock (`pixi run bench` = `flock /tmp/mojo-bench.lock`).
No number is estimated; flops/bytes are computed arithmetically from the
actual arrays passed in (`/tmp/profile_flops.py` — real inputs, real shapes).
Machine: Intel Xeon E5-2697 v4 @ 2.30 GHz (Linux 6.8.0-147), 72 CPUs (2 NUMA
nodes, hyperthreaded), AVX2+FMA (no AVX-512 → Mojo SIMD width `W = 4` for
float64). Versions: numpy 2.5.1, scipy 1.18.0, upstream `control` 0.10.2,
Python 3.13.14. Inputs are identical to `bench/bench.py` (rng seed 7:
TF order 32 with 33/33 coefficients, SS 12-state 2×2, CT 4-state 100k samples,
DT 8-state 2×2 200k samples).

## 1. Official benchmark re-run (`pixi run bench`)

```
Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz

| Case | mojo-control | control 0.10.2 | Speedup |
| TF frequency response, order 32, 500k points | 9.14 ms | 139.06 ms | 15.22x faster |
| SS frequency response, 12 states, 2x2, 20k points | 8.16 ms | 320.67 ms | 39.28x faster |
| Continuous forced response, 4 states, 100k samples | 2.95 ms | 677.07 ms | 229.78x faster |
| Discrete forced response, 8 states, 2x2, 200k samples | 15.96 ms | 1608.04 ms | 100.74x faster |
```

Cross-check with the deeper profiler (best-of-7 / best-of-5, same sizes):

| Case | mojo (profiler) | upstream | numpy/Python baseline |
|---|---:|---:|---:|
| TF, order 32, 500k pts | 10.05–12.63 ms | 129.64–148.48 ms | `np.polyval` complex Horner: 103.41–118.22 ms |
| SS, 12st 2×2, 20k pts | 8.33–10.44 ms | 281.11–384.82 ms | per-point `C@solve(sI−A,B)+D` Python loop: 626.91–746.61 ms |
| Forced CT, 4st, 100k | 3.13–3.15 ms | 668.82–690.36 ms | Python per-step `Ad@x+B0*u+B1*u_next` loop: 794.90–838.11 ms |
| Forced DT, 8st 2×2, 200k | 15.07–17.51 ms | 1642.52–1764.96 ms | Python per-step `Ad@x+Bd@U` loop: 1622.63–1687.60 ms |

Stability (3 outer best-of-5): TF `15.85 / 16.18 / 20.18 ms` (high variance —
the 64-chunk `parallelize` launch costs ~5 ms wall-clock on this shared host,
matching `README.md`), SS `7.97 / 7.97 / 9.77 ms`, CT `3.12 / 3.53 / 3.55 ms`,
DT `17.52 / 15.77 / 15.07 ms`. The threaded cases vary most. **No bench case
is slower than upstream.**

## 2. Ranked hotspot table

Ranked by mojo wall time at bench/benchmark sizes (kernel-only time in
parentheses — raw `ctypes` call with preallocated buffers, no Python glue).
"Python/glue vs kernel" = share of wall time spent in Python/numpy/ctypes
marshalling vs inside `src/control.mojo`. Flops are real-flop counts
(complex mul = 4 mul + 2 add, complex add = 2, complex div ≈ mul+div);
bytes are float64 = 8 B one-pass traffic (read inputs + write outputs,
plus the per-call allocations/copies listed in §4). Thresholds below
≈2 flops/byte for GPU candidacy per `MOJO_NOTES.md` §5.

| Rank | Function (public entry) | Input size | Time (best) | Bytes touched | Flops | Flops / byte | Memory- or compute-bound | Python/glue vs kernel | Next phase |
|---:|---|---|---:|---:|---:|---:|---|---|---|
| 1 | `forced_response` DT, `mctl_ss_simulate` (`Ad,Bd,Cd,Dd`, 8st, 2×2) | 200k steps, `(2, 200001)` input | 12.60 ms kernel / 15.07–17.51 ms full-call | 19.20 MB (inputs 3.2 MB + `y` 3.2 MB + `states` 12.8 MB + matrices) | 38.4 MFLOP (192/step: 2·n·p proj + 2·n² matvec + 2·n·m input) | 2.00 | **memory-bound** (at threshold; recurrence streams `y`+`states` = 16 MB per call) | ~25% glue (~2.4 ms `_time_and_input` incl. two 3.2 MB transpose copies) / 75% kernel | **none** — sequential recurrence, 2.0 flops/B leaves no GPU headroom and no parallel axis (time axis is serial by definition). Glue copies could be removed without an accel phase, but keep as `none` for accel |
| 2 | `frequency_response` TF, `mctl_tf_eval` (Horner, 33+33 coeff) | 500k points | 6.31–7.22 ms kernel / 9.14–12.63 ms full-call | 16.0 MB min (grid 8 MB + real/imag 8 MB); 28.0 MB full-call incl. `zeros_like` grid write 4 MB + complex interleave 8 MB | 0.196 GFLOP (392/pt: 6×(n−1) Horner + final scale + 2 div) | 12.25 min-traffic (7.00 full-call) | **compute-bound** — 2.2 GB/s measured, far below DRAM peak; Horner chain is latency-bound | ~25–35% glue (`result.real/imag` interleave 1.35 ms + grid alias 0.26 ms + `f64()` copies) / 65–75% kernel | **none** (already parallel+SIMD; see §3). GPU would satisfy roofline (12.25 > 2) but needs the missing device-function declaration (`README.md` GPU section) and parallelize already gives 3.4× on 72 CPUs. `simd`/`unroll` marginal at best |
| 3 | `frequency_response` SS, `mctl_ss_eval` (assemble+eliminate+backsolve+project) | 20k points, n=12, m=p=2 | 7.59–10.11 ms kernel / 8.16–10.44 ms full-call | 1.60 MB RAM-model (inputs 0.32 MB + complex out 1.28 MB); 162.9 MB if L1/L2 scratch passes counted | 0.160 GFLOP (8014/pt: assemble 312 + eliminate 6262 + backsolve 1248 + project 192) | 100.08 RAM-model (0.98 incl. on-chip scratch) | **compute-bound** (RAM-model 100 flops/B; scratch slab is 2.6 KiB/worker — L1/L2 resident, not DRAM). 21.1 GFLOP/s parallel, 4.8 GFLOP/s per core serial | ~2–5% glue (0.19 ms allocs on 20k) / ~95–98% kernel | **gpu** — 100 flops/B ≫ 2 flops/B break-even (consistent with `README.md` "≈57 flops/B", same conclusion). Blocked on toolchain, not on roofline: device-function declaration missing (see §5). Otherwise **none** on CPU (already parallel+SIMD; 4.4× from threads) |
| 4 | `forced_response` CT, `mctl_ss_simulate` (`Ad,B0,B1`, 4st, SISO) | 100k steps, 100001 input vector | 2.44 ms kernel / 2.95–3.15 ms full-call | 4.80 MB (U 0.8 MB + `y` 0.8 MB + `states` 3.2 MB + matrices) | 5.2 MFLOP (52/step: 2n proj + 2n² matvec + 3nm input with B0+B1) | 1.08 | **memory-bound** | ~22% glue (0.73 ms `_time_and_input` FTV checks + 0.09 ms `expm` block build) / 78% kernel | **none** — 1.08 flops/B is below GPU break-even and the loop is serial in time |
| 5 | `frequency_response` TF serial path (below 12M coeff-products) | 100k pts (43 ns/pt) | 4.37 ms (serial) vs 5.03 ms at 200k (parallel) | 3.2 MB | 39.2 MFLOP | 12.25 | compute-bound (Horner chain) | kernel only | Threshold already tuned: parallel engages at 181,819 pts for 33+33 coeff; serial wins below. `parallel` — **already applied**, no change |
| 6 | `frequency_response` SS serial path (below 6M work units) | ≤3472 pts (1676–1716 ns/pt) | 5.82 ms at 3472 pts (serial) | 0.28 MB | 27.8 MFLOP | 100.08 | compute-bound | kernel only | Crossover measured at 3472 vs 4000 pts (1676→1446 ns/pt); launch overhead stops paying below. `parallel` — **already applied**, threshold correct |
| 7 | `step_response` / `impulse_response` (CT, 4st SISO, 10k grid) | 10k pts, 1 input | 0.70–1.00 ms | ~1.0 MB (stacked `y`/`states` copies) | ~0.5 MFLOP | ~0.5 | memory-bound (numpy glue: per-input `U` alloc, double `expm` recompute, `np.stack` copies) | ~100% glue — 2× `forced_response` (2× `expm` 0.09 ms) + stacking | **none** on accel; glue-only reduction candidate (cache `Ad,B0,B1` across input indices) |
| 8 | `zeros` SS SISO 4st — ⚠️ **SLOWER than upstream** | 4 states | 0.44–0.66 ms mojo vs 0.13–0.19 ms upstream (**3.3× slower**) | ~350 B (A,B,C,D) | small (QZ/scipy `ss2zpk` path) | — | overhead-bound (scipy dispatch) | ~100% Python/scipy glue; **no Mojo kernel involved at all** | **none** — pure-Python upstream path that is 3× faster for no Mojo reason; do not "optimize" with accel. Needs investigation in a Python-parity phase, not an accel phase |
| 9 | `evalfr` single point (TF 20.8 µs, SS 45.2–46.5 µs) | 1 point | TF 20.8 µs vs upstream 164.2 µs (faster); SS 45.2 µs vs upstream 47.5 µs (parity) | <1 KB | ~8000 flops (SS) | — | latency/overhead-bound | ~100% fixed cost: ctypes transfer 407 ns, `f64()`/`addr()` checks, numpy allocs — kernel is noise | **none** — SIMD/parallel/GPU all below threshold by construction |
| 10 | `poles` TF 33 (`np.roots`, 0.53 ms) / SS 12 (`eigvals`, 0.08 ms) / `dcgain`, `ss()`, `tf()`, `feedback`, `series`, `append`, `c2d` | small | 0.02–0.53 ms; all ≤ upstream (SS `poles`: 0.079 vs 0.082 ms parity; TF `poles`: 0.53 vs 2.36 ms faster; `c2d` 0.07 vs 0.28 ms) | <10 KB | <40 kflop | — | overhead-bound (numpy/scipy dispatch) | Python/scipy glue; thin wrappers, no per-element ctypes | **none** |

Notes on rows 1–4 arithmetic (`/tmp/profile_flops.py`):
- TF: Horner inner step is 2 mul + 1 add per real part and per imaginary
  part (6 flops/coeff/point); 64 coeff-steps = 384 + final scale (6) + 2 div.
- SS: assemble (2·n² + n·m), eliminate (pivot-norm scan + 8 flops per matrix
  row-update element + 8 per RHS element + factor), backsolve (8 per
  `(i,j,q)` + final scale), project (4 per `(p,m,n)`). The elimination
  dominant term n³/3·8 is the only place GPU could win.
- SIM: output `2·n·p`, matvec `2·n²`, input `3·n·m` (CT, B0+B1) or `2·n·m`
  (DT, B0 only). Both workloads stream `states (steps×n×8 B)` which
  dominates bytes.

## 3. Parallel/SIMD status — already applied, evidence it is tuned

Measured crossover sweeps (best-of-5):

- TF serial → parallel at `count·(nnum+nden) = 12,000,000`: 181,818 pts serial
  7.89 ms (43 ns/pt), 200,000 pts parallel 5.03 ms (25 ns/pt), 500k parallel
  6.31 ms (13 ns/pt). Serial 500k would be ~21.7 ms → threads give ~3.4×.
  Threshold is correct; no re-tuning needed.
- SS serial → parallel at `count·n²·n = 6,000,000`: 3,472 pts serial 5.82 ms
  (1676 ns/pt), 4,000 pts parallel 5.79 ms (1446 ns/pt), 20k parallel 7.59 ms
  (380 ns/pt) → ~4.4× from threads. Threshold correct.

SIMD (`W = 4`, AVX2): TF Horner vector body, SS assemble/eliminate row-update
are vectorized; pivot search and backsolve stay scalar (strided) as documented.
The remaining serial bottlenecks are the Horner latency chain (rank 2), the
SS pivot scan + backsolve scalar cores (rank 3), and the SIM scalar
accumulators (`v0..v3` unrolled 4-row block already present). A further
`unroll`/`simd` phase would chase <10% ceilings against the memory-bound
and latency-bound floor — hence `none` for ranks 1, 2, 4 on CPU.

## 4. Python-level loop / ctypes / allocation audit

**No Python loop re-enters ctypes per element.** Verified by source grep:
exactly three ctypes call sites in `python/mojo_control/core.py` — line 147
(`mctl_tf_eval`), line 313 (`mctl_ss_eval`), line 731 (`mctl_ss_simulate`) —
each called once per public call with whole buffers. The only Python
per-element/per-point loops are:

1. `StateSpace._evaluate_parts` singular-point fallback (core.py ≈L334):
   `for index in np.flatnonzero(status): C @ solve(...)` — per-*singular*
   point numpy solve, not per-element ctypes. Zero iterations in all
   benchmark cases (status all-zero).
2. `step_response` / `impulse_response` (core.py ≈L799/L858):
   `for index in selected_inputs:` — one full `forced_response` (with its
   own `expm`) **per input channel** (m iterations). For MIMO `m=2` this
   doubles the `expm` and input-copy cost. Recomputes `Ad,B0,B1` every
   iteration; final `np.stack` copies `outputs`+`states`.
3. `__pow__` exponentiation-by-squaring on TF — logarithmic, fine.

**Per-call allocations/copies (measured sizes, bench cases):**

- TF 500k full-call: `xr` zeros-like write 4 MB, `xi` alias (no copy, CT),
  `real`+`imag` 8 MB, complex result 8 MB + `.real/.imag` interleave reads
  8 MB → **~28 MB** vs 16 MB kernel minimum. The interleave alone measures
  1.35–1.46 ms (timeit). `f64()` on already-contiguous 500k arrays: 0.01 ms
  (no copy — `ascontiguousarray` no-op) — not a problem. For discrete TF the
  grid build is `np.cos`+`np.sin`, ~8 MB.
- SS 20k full-call: `real` 640 KB + `imag` 640 KB + `scratch` 172 KB +
  `status` 160 KB + complex 1.28 MB → **~2.9 MB**, 0.19 ms measured. Glue
  is 2–5% of wall time — not a hotspot at bench sizes (it grows to ~10% at
  100 pts where total is 0.24 ms).
- SIM CT call: `T` copy 0.8 MB + `U` copy 0.8 MB + `y` 0.8 MB + `states`
  3.2 MB → ~5.6 MB; `_time_and_input` 0.73 ms of the 2.95 ms total (25%).
- SIM DT call: `T` 1.6 MB + `U` 3.2 MB + **second** transpose copy 3.2 MB
  (`f64(U.T)` after `f64(U)`) + `y` 3.2 MB + `states` 12.8 MB → ~24 MB;
  `_time_and_input`+transpose = 2.38 ms of the 15–17 ms total (~15%).
  The double copy of `U` is the single largest glue item in the port.
- `FrequencyResponseData.magnitude/.phase` are **lazy** in mojo-control
  (cold: abs 2.18 ms, angle 11.10 ms at 500k; measured 0.00 ms when cached)
  but **eager** in upstream — the bench does not touch them, so mojo's
  headline speedup slightly understates the caller-visible cost if the
  caller reads `.magnitude`/`.phase`. Flagged here; not an accel issue.

**Bare-call overhead**: `mctl_chunk_count` round-trip 407 ns; single-point
`evalfr` fixed cost ~21 µs (TF) / ~46 µs (SS) — dominated by numpy allocs
and Python call chain, not ctypes. No batching gap worthy of a phase.

## 5. Slower-than-upstream rows (kept, not dropped)

- `zeros` (SISO SS, 4 states): mojo 0.44–0.66 ms vs upstream 0.13–0.19 ms →
  **mojo-control is ~3.3× slower**. Mechanism (verified in this session):
  mojo `StateSpace.zeros()` calls `scipy.signal.ss2zpk(A,B,C,D)` directly
  (0.46 ms measured for the raw scipy call), while upstream
  `StateSpace.zeros()` falls back to a QZ generalized-eigenvalue path
  directly on `(L,M)` block matrices with no scipy dispatch overhead
  (0.13 ms). **No Mojo kernel is involved.** This is a Python-algorithm
  choice, not an accel gap; the fix belongs to a parity/Python phase
  (`none` for accel). At larger sizes the gap may invert, but at the 4-state
  SISO size real callers use, it is a real regression.
- Everything else measured is faster or at parity (SS `poles` 0.079 vs
  0.082 ms; SS `evalfr` 45.2 vs 47.5 µs; `dcgain`, constructors,
  interconnections, `step_response` 0.70 vs 65.61 ms, `c2d` 0.07 vs 0.28 ms,
  `impulse_response` — all faster). No other slower rows found.

## 6. Next-phase verdicts (summary)

| Hotspot | Verdict | Flops/byte + thread/SIMD context |
|---|---|---|
| SIM DT `mctl_ss_simulate` (rank 1) | **none** | 2.00 flops/B, serial recurrence, already unrolled (4-row) — no parallel axis, no GPU headroom |
| TF Horner `mctl_tf_eval` (rank 2) | **none** | 12.25 flops/B but already parallel+SIMD; latency-chain bound; GPU blocked on missing device-function decl |
| SS eval `mctl_ss_eval` (rank 3) | **gpu** (conditional) | 100 flops/B RAM-model ≫ 2 threshold; already parallel+SIMD; blocked on toolchain (§below). If GPU stays blocked, `none` |
| SIM CT `mctl_ss_simulate` (rank 4) | **none** | 1.08 flops/B < 2, serial |
| TF/SS sub-threshold serial paths (ranks 5–6) | thresholds correct | measured crossovers match code constants |
| step/impulse glue (rank 7) | **none** (accel) | 0.5 flops/B; pure numpy glue |
| `zeros` SISO (rank 8, slower) | **none** | no kernel; Python-path parity fix elsewhere |
| small/overhead rows (9–10) | **none** | latency-bound, below every accel threshold |

**GPU toolchain blocker (verified by `README.md` GPU section + `MOJO_NOTES.md`
§5/§5b, not re-probed here — no algorithm change is in scope):**
`max.gpu.host.DeviceContext` / `enqueue_create_buffer` / `enqueue_copy` /
`enqueue_function` / `synchronize` and `max.gpu.global_idx` compile on the
pinned toolchain, but no device-function declaration form (`DevicePassable`
plain `def`, kernel decorator, or `max.gpu.host.compile` decoration) resolves.
Rank 3 remains a documented GPU candidate (100 flops/B) until a toolchain
that exposes kernel declaration is pinned. No `simd`/`unroll`/`parallel`
verdict is given for any current hotspot on measured evidence: the two
frequency kernels are already parallel+SIMD with correctly placed thresholds,
and the simulation kernels are memory-bound serial recurrences at or below the
GPU break-even.

## 7. Method & reproducibility

- `pixi run bench` (flock-wrapped, this session) — §1 official numbers.
- `/tmp/profile_entrypoints.py` — bench cases vs upstream vs numpy baselines
  + kernel-only splits (preallocated raw ctypes) + small sizes + evalfr.
- `/tmp/profile_extra.py` — glue splits (`_time_and_input`, `expm`, allocs),
  other entry points (poles/zeros/dcgain/constructors/interconnections/step/
  impulse/c2d), crossover sweeps, per-element-ctypes audit.
- `/tmp/profile_tie.py`, `/tmp/profile_final.py` — stability, interleaved
  timing of alloc vs interleave (timeit), serial/parallel crossovers,
  single-point evalfr vs upstream.
- `/tmp/profile_flops.py` — flop/byte accounting from actual arrays.
- Original upstream `control` sources inspected in the pixi env
  (`xferfcn.py:zeros` → `roots`; `statesp.py:zeros` → slycot/QZ fallback)
  to confirm the `zeros` mechanism. `scipy ss2zpk` timed standalone.
- Sortestflops model note: elimination factor count uses ~8 real flops per
  row-factor; correcting to 8 changes the SS total by <1% and no verdict.
  README's "≈57 flops/B" for SS uses a different complex-flop convention/
  traffic model than mine (100); both give the same `gpu` verdict (>2).

*Only file added in this phase: `bench/analysis.md`.*
