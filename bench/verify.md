# VERIFY — mojo-control benchmark and parity verification

Session: 2026-11-09, VERIFY phase. No code from earlier phases is taken at
face value; every number below was re-measured in this session.
Host: Intel Xeon E5-2697 v4 @ 2.30 GHz, Linux 6.8.0-147-generic
(`/proc/cpuinfo`), control 0.10.2, numpy 2.5.1, scipy 1.18.0,
Python 3.13.14.
Toolchain (pinned by `pixi.toml`): mojo 1.2.0.dev2026092605,
max 26.7.0.dev2026092605.

## 1. Gate run (DO step 1, in order)

```
pixi run build && pixi run test
```
- Build: succeeded (warnings only: deprecated positional `__getitem__`,
  use `unsafe_offset=`; in `src/control.mojo`).
- Test: `55 passed in 3.41s`.
- `pixi run bench` (first run, this session, under
  `flock /tmp/mojo-bench.lock`):
  ```
  | TF frequency response, order 32, 500k points | 10.02 ms | 127.13 ms | 12.69x faster |
  | SS frequency response, 12 states, 2x2, 20k points | 7.77 ms | 388.23 ms | 49.99x faster |
  | Continuous forced response, 4 states, 100k samples | 2.99 ms | 855.15 ms | 285.97x faster |
  | Discrete forced response, 8 states, 2x2, 200k samples | 16.49 ms | 1846.54 ms | 112.00x faster |
  ```

Final gate (after README correction, no code changed):
`pixi run build && pixi run test` — both green (see §7).

## 2. What was run repeatedly (DO step 2)

`pixi run bench` was run 18 times in this session. Each invocation is
best-of-4 inside one process (`best_time(..., repeat=4)`), asserts
numerical agreement before timing, and measures both libraries' public
Python API on the same systems and inputs.

| Run | TF ms / upstream ms | SS ms / upstream ms | CT ms / upstream ms | DT ms / upstream ms |
|---|---|---|---|---|
| 1 (first) | 10.02 / 127.13 | 7.77 / 388.23 | 2.99 / 855.15 | 16.49 / 1846.54 |
| 2 | 8.89 / 139.09 | 8.17 / 331.45 | 3.01 / 647.01 | 16.22 / 1639.69 |
| 3 | 8.48 / 143.17 | 8.93 / 432.58 | 3.01 / 881.14 | 15.98 / 1956.80 |
| 4 | 8.16 / 135.94 | 9.10 / 326.61 | 3.01 / 676.40 | 14.62 / 1641.24 |
| 5 | 8.81 / 139.60 | 8.38 / 301.18 | 3.00 / 651.84 | 14.46 / 1600.12 |
| 6 | 7.88 / 129.67 | 7.75 / 303.94 | 2.95 / 665.23 | 18.60 / 1553.41 |
| 7 | 8.99 / 140.66 | 8.10 / 309.38 | 3.00 / 636.15 | 15.94 / 1623.38 |
| 8 | 9.26 / 150.51 | 7.28 / 362.57 | 2.98 / 794.82 | 14.88 / 1559.88 |
| 9 | 8.68 / 136.88 | 8.08 / 314.34 | 3.16 / 671.76 | 17.26 / 1647.32 |
| 10 | 9.74 / 144.67 | 8.87 / 297.39 | 2.98 / 679.06 | 14.74 / 1612.41 |
| 11 | 9.49 / 154.50 | 9.27 / 289.26 | 3.03 / 675.48 | 14.56 / 1635.51 |
| 12 | 9.50 / 130.38 | 7.87 / 283.84 | 3.07 / 788.75 | 14.20 / 1761.28 |
| 13 | 9.38 / 134.81 | 8.14 / 316.91 | 3.06 / 661.93 | 14.29 / 1616.71 |
| 14 | 8.93 / 136.60 | 9.02 / 300.60 | 3.06 / 653.69 | 15.48 / 1566.63 |
| 15 | 10.62 / 144.90 | 7.79 / 277.59 | 3.04 / 658.40 | 15.18 / 1670.62 |
| 16 | 9.11 / 136.85 | 7.87 / 293.22 | 3.01 / 663.78 | 15.02 / 1534.82 |
| 17 | 8.04 / 141.92 | 7.13 / 343.93 | 2.95 / 685.24 | 15.83 / 1575.04 |
| 18 | 8.91 / 133.76 | 7.67 / 409.39 | 3.17 / 800.19 | 14.17 / 1836.90 |

Medians over the 18 runs (now the README table):
mojo 8.96 / 8.09 / 3.01 / 15.10 ms; upstream 137.99 / 311.86 / 673.62 /
1629.45 ms; speedup 15.40x / 38.55x / 223.79x / 107.91x faster.
Speedup is upstream ÷ mojo in every run and in the table.

### 20% comparison against the previous README table

Previous README claimed: TF 8.01 / 140.33 / 17.52x; SS 7.60 / 312.27 /
41.11x; CT 3.02 / 672.24 / 222.74x; DT 13.65 / 1653.98 / 121.17x.

At the median, all twelve cells reproduce within 12% of the previous
table (+11.9% / −1.7% / −10.7%; +6.4% / −0.1% / −6.2%; −0.3% / +0.2% /
+0.4%; +10.6% / −1.5% / −9.5%): **nothing fails a median-based 20% bar.**

At the single-run level, 8 of 72 cells (11%) exceeded ±20%: TF mojo 3/18,
SS upstream 3/18 and speedup 4/18, CT upstream 2/18 and speedup 2/18,
DT mojo 3/18 and speedup 2/18 (max single-run deviation +38% SS
upstream, +31% CT upstream, +36% DT mojo, +25% TF mojo). The variation
is run-to-run jitter on this shared host, consistent with the
`parallelize` launch cost documented in `bench/analysis.md`, not with a
systematic bias: the mojo column's median is the slowest of:
TF +11.9% and DT +10.6% high, SS +6.4% high, CT −0.3%; upstream medians
are within 2% on all four rows.

Decision: report medians. A single best-of-4 run is not a stable basis
for a ±20% verdict on this host, and the previous table's own range
note said the same. Per the session-only-numbers rule the README table
is nevertheless updated to this session's medians (and the `before`
column to a this-session re-measurement, §6), with the wording
unchanged apart from the range note. All four rows stay "faster";
none was removed or reworded.

## 3. Same-input / no-skip-work audit (DO step 4)

`bench/bench.py` (`cases()`), read line by line:

- One `rng = np.random.default_rng(7)` stream per process feeds all four
  cases in fixed order, so systems and inputs are identical every run.
- Each case builds both a mojo system and an upstream system from the
  *same* NumPy arrays and times both on the *same* grid/input object:
  - TF: `mc.tf(numerator, denominator)` vs `ct.tf(...)`, both
    `frequency_response(..., omega_tf)`, `omega_tf = logspace(-3, 3, 500_000)`.
  - SS: `mc.ss(A, B, C, D)` vs `ct.ss(A, B, C, D)`, both
    `frequency_response(..., omega_ss, squeeze=False)`,
    `omega_ss = logspace(-3, 3, 20_000)`.
  - CT: `mc.ss(A4, B4, C4, [[0.0]])` vs `ct.ss(...)`, both
    `forced_response(..., T, U)`, `T = linspace(0, 200, 100_001)`,
    `U = sin(0.7*T)`.
  - DT: `mc.ss(Ad, Bd, Cd, Dd, 0.01)` vs `ct.ss(..., 0.01)`, both
    `forced_response(..., Td, Ud)`, `Td = arange(200_001)*0.01`,
    `Ud = vstack([sin(0.4*Td), cos(0.9*Td)])`.
- `best_time` calls the full public-API callable `repeat=4` times and
  keeps the best. No caching, no reuse of a prior result; every
  iteration recomputes. `main()` executes both sides once before timing
  and asserts `np.allclose(..., rtol=1e-9, atol=1e-9)` on `frdata`
  (freq rows) or `outputs` (time rows) — on every invocation, including
  all 18 here, without tripping.
- Mojo serial fallbacks: `TF_PARALLEL_WORK = 12_000_000` and
  `SS_PARALLEL_WORK = 6_000_000` in `src/control.mojo`, with
  `parallelize` above them. Bench work: TF `500_000 * (33+33) = 33M`,
  SS `20_000 * 12*12 * 12 = 34.56M` — both above threshold, so the
  timed path is the full parallel kernel on every run. The kernels
  write every point (`tf_eval_range` fills `[start, end)` per chunk;
  `ss_point` runs per frequency point, singular ones flagged in
  `status` and recomputed by an exact NumPy `solve` fallback in
  `core.py` — that fallback is part of the timed call, not a skip).
- For CT, `forced_response` computes one SciPy `expm` of the augmented
  block and then runs the full recurrence in Mojo; for DT the same
  recurrence runs directly. This matches the documented method
  (exact ZOH-equivalent step for linear input interpolation) and is
  asserted for numerical agreement before timing; it is not skipped
  work. `step_response` is not benchmarked, and it delegates to
  `forced_response` unchanged in this codebase.

Independent re-check in this session (separate process, same seed and
inputs as `bench.py`, not calling `bench.py`):

- TF: `mc...complex` vs upstream, shapes `(500000,)` both (SISO
  squeeze), `array_equal(omega, input)`, `max|diff| ≈ 1.4e-14`,
  nonzero fraction 1.0.
- SS: `frdata` shapes `(2, 2, 20000)` both, `max|diff| ≈ 8.9e-15`,
  nonzero fraction 1.0 (no skipped points).
- CT: outputs length 100001 both, `max|diff| ≈ 1.5e-14`.
  1 value in 100001 is exactly zero on both sides (input zero crossing;
  CT input `sin(0.7*T)` at `T=0`).
- DT: outputs `(2, 200001)` both, `max|diff| ≈ 5.3e-14`.

No asymmetry, no skipped work, no cached result was found.

## 4. Slower-than-upstream rows (DO step 3)

No row of `bench/bench.py` is slower than upstream in any of the 18
runs: the mojo column is 13–54× below upstream on every case.
Nothing was removed or reworded; the README table still shows all four
rows as "faster", which is what the benchmark measures.

A slower path exists **outside** this table and stays documented as
slower in `bench/analysis.md` §5 (not in the README table, which covers
only `bench/bench.py`):

- `zeros` (SISO SS): mojo 0.37 ms vs upstream 0.12 ms (best of 20,
  re-measured this session on the 2-state fixture system; ≈3× slower).
  Mechanism: mojo `StateSpace.zeros()` calls `scipy.signal.ss2zpk`;
  upstream uses a QZ path with no scipy dispatch. Wording in
  `analysis.md` ("no bench case is slower than upstream", with the
  `zeros` row explicitly "slower") was left intact.

## 5. Parity tests (DO step 5)

`git log -p -- tests/` and `git diff c1ce3ba..HEAD -- tests/`:

- Only **additions** plus one parametrisation extension:
  `test_transfer_function_simd_tail_and_parallel_threshold` gained
  `200_003` to its `count` parametrisation (one extra, stricter case).
  New tests: `test_transfer_function_parallel_chunks`,
  `test_state_space_parallel_chunks_and_simd_tail` (against a NumPy
  `C @ solve(sI−A, B) + D` reference),
  `test_state_space_singular_frequency_point`,
  `test_forced_response_state_count_remainders`.
- No removed assertion. Every pre-existing tolerance was left
  unchanged: `assert_response_equal` rtol 1e-10 / atol 1e-10 (default),
  TF rtol/atol 1e-12, discrete 1e-11, forced-response atol 2e-12 (CT)
  and 1e-12 (DT), root matching abs 1e-10 — identical in `c1ce3ba` and
  HEAD. Verified by diffing the two revisions; the only changed line
  is the parametrisation above.
- Current suite: 55 tests, all passing against `control==0.10.2`,
  comparing directly with upstream objects (not NumPy-only references,
  except the one SS parallel test, which correctly uses the analytic
  formula). Nothing was restored because nothing was loosened.

## 6. `before` column re-measurement

- Old revision: `git archive 15d1e82` (immediately before the accel
  commit `1433254`), built with the current toolchain
  (`mojo build --emit shared-lib /tmp/vb/src/control.mojo -o
  /tmp/old_lib.so`, warnings only), run with
  `MOJO_CONTROL_LIB=/tmp/old_lib.so PYTHONPATH=/tmp/vb/python` (its own
  `bench/bench.py`, same systems/inputs/timing), under the same flock
  lock. 5 runs, mojo column only:

  | Run | TF | SS | CT | DT |
  |---|---|---|---|---|
  | A | 25.96 | 45.60 | 3.97 | 25.73 |
  | B | 26.08 | 45.34 | 3.70 | 23.15 |
  | C | 24.86 | 45.23 | 3.85 | 25.37 |
  | D | 26.16 | 45.62 | 3.87 | 25.15 |
  | E | 23.81 | 43.10 | 3.83 | 24.72 |
  | median | 25.96 | 45.34 | 3.85 | 25.15 |

- Against the previous README `before` (23.11 / 45.76 / 3.87 / 24.29 ms)
  the median is +12.3% / −0.9% / −0.5% / +3.5% — reproduces within 20%
  on all four rows, and every single run was within 20% too. The README
  `before` column is now the median above, so every cell of the table
  was measured in this session.

## 7. README changes made

- Table preamble: "best-of-four" → "medians of 18 best-of-four
  measurements …", so the statistic actually tabled is stated.
  Date/kernel unchanged (already this session's host).
- Main table → this session's medians:
  - TF: 8.96 / 137.99 / 15.40x faster (was 8.01 / 140.33 / 17.52x)
  - SS: 8.09 / 311.86 / 38.55x faster (was 7.60 / 312.27 / 41.11x)
  - CT: 3.01 / 673.62 / 223.79x faster (was 3.02 / 672.24 / 222.74x)
  - DT: 15.10 / 1629.45 / 107.91x faster (was 13.65 / 1653.98 / 121.17x)
- `before` column → re-measured medians (§6): 25.96 / 45.34 / 3.85 /
  25.15 ms.
- Range note rewritten with this session's observed spans
  (TF 7.9–10.6 ms mojo / 127–155 ms upstream; SS 7.1–9.3 / 278–433;
  CT 3.0–3.2 / 636–881; DT 14.2–18.6 / 1535–1957) and the max
  single-run deviations. Speedups are the quotients of the two
  adjacent table columns (e.g. 137.99/8.96 = 15.40).
- No code changed. No test changed. No row removed or reworded to look
  better; all four remain "faster", which is what `bench/bench.py`
  measures; the slower `zeros` case remains slowness-documented in
  `bench/analysis.md`.

## 8. What could not be reproduced

- Exact historical point values (any prior session's single
  best-of-4). On this shared host the same binary moves by ±10–38%
  run to run (§2); at the median the previous table reproduces within
  12%, inside 20%. The table was updated to this-session medians per
  the session-only-numbers rule, not because the old table was
  falsified at the median.
- The previous verify session's quoted ranges (e.g. a 5.10 ms CT
  outlier, mojo 7.4–11.5 / 7.5–11.4 / 2.9–5.1 / 13.3–16.8) were not
  reproduced as stated: this session's CT mojo column never exceeded
  3.17 ms, and the DT mojo column never went below 14.17 ms — i.e. the
  old lower edge for DT (13.3–13.65 ms) sits below every run here,
  consistent with apex-of-jitter readings rather than error, but not
  re-measured. The new note carries this session's spans instead.
- One item from `bench/analysis.md` was spot-checked but not
  re-derived: the GPU/FMA flop counts and the `zeros` mechanism. The
  `zeros` timing *was* re-measured (§4); flop counts are arithmetic
  from the arrays, not benchmarks, and are out of scope for this
  README table.

## 9. Reproduction commands

```bash
pixi run build && pixi run test     # 55 passed
pixi run bench                       # table in §1 / §2 (use medians over runs)
git log -p -- tests/                 # §5: additions only, no loosened tolerances
git archive 15d1e82 | tar -x -C /tmp/vb
pixi run bash -c 'mojo build --emit shared-lib /tmp/vb/src/control.mojo -o /tmp/old_lib.so'
MOJO_CONTROL_LIB=/tmp/old_lib.so PYTHONPATH=/tmp/vb/python \
  pixi run flock /tmp/mojo-bench.lock python /tmp/vb/bench/bench.py   # §6 before-column
```

*Files changed in this phase: `README.md` (benchmark preamble, table,
and range note only) and `bench/verify.md` (this file). No source or
test change — the gate was green with no failing-test evidence of a
bug, so per the rules no code was touched.*
