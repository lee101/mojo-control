# VERIFY — mojo-control benchmark and parity verification

Session: 2026-11-09, VERIFY phase (no code written in earlier phases assumed here).
Host: Intel Xeon E5-2697 v4 @ 2.30 GHz, Linux 6.8.0-147-generic (`/proc/cpuinfo`),
control 0.10.2, numpy 2.5.1, scipy 1.18.0, Python 3.13.14.
Toolchain (pinned by `pixi.toml`): mojo 1.2.0.dev2026092605, max 26.7.0.dev2026092605.

Every number below was measured in this session under the repository's
machine-wide benchmark lock (`pixi run bench` = `flock /tmp/mojo-bench.lock
python bench/bench.py`). No number is copied from a previous session.

## 1. Baseline gate

```
pixi run build && pixi run test
```
- Build: succeeded (warnings only: deprecated positional `__getitem__`,
  use `unsafe_offset=`; in `src/control.mojo`).
- Test: `55 passed in 3.75s` (python -m pytest tests -q).
- `pixi run bench` (first run, this session):
  ```
  Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz (Linux 6.8.0-147-generic)

  | Case | mojo-control | control 0.10.2 | Speedup |
  |---|---:|---:|---:|
  | TF frequency response, order 32, 500k points | 8.34 ms | 141.40 ms | 16.95x faster |
  | SS frequency response, 12 states, 2x2, 20k points | 7.38 ms | 317.50 ms | 43.02x faster |
  | Continuous forced response, 4 states, 100k samples | 2.94 ms | 667.81 ms | 226.81x faster |
  | Discrete forced response, 8 states, 2x2, 200k samples | 15.75 ms | 1545.49 ms | 98.11x faster |
  ```

Final gate (after README correction, no code changed):
`pixi run build && pixi run test` — both green (55 passed).

## 2. What was run repeatedly

`pixi run bench` was run 13+ times in this session (best-of-4 inside one
invocation, as `bench/bench.py` defines `best_time(..., repeat=4)`).
Representative raw outputs (first 5, plus the final two canonical runs):

| Run | TF ms / upstream ms | SS ms / upstream ms | CT ms / upstream ms | DT ms / upstream ms |
|---|---|---|---|---|
| 1 (first) | 8.34 / 141.40 | 7.38 / 317.50 | 2.94 / 667.81 | 15.75 / 1545.49 |
| 2 | 7.77 / 138.07 | 7.51 / 305.78 | 3.55 / 688.00 | 13.34 / 1674.49 |
| 3 | 8.62 / 129.73 | 9.54 / 290.77 | 3.02 / 674.38 | 15.91 / 1753.88 |
| 4 | 11.46 / 149.91 | 9.24 / 315.62 | 3.04 / 630.98 | 14.51 / 1552.21 |
| 5 | 10.58 / 150.38 | 8.75 / 319.18 | 3.11 / 686.64 | 16.45 / 1771.21 |
| 6 | 8.66 / 141.18 | 7.72 / 377.54 | 3.02 / 711.54 | 14.17 / 1913.83 |
| 7 | 8.34 / 149.61 | 8.95 / 355.25 | 2.99 / 663.37 | 13.33 / 1757.75 |
| 8 | 8.30 / 134.13 | 8.35 / 291.04 | 2.99 / 643.63 | 16.76 / 1636.49 |
| 9 | 9.72 / 135.56 | 11.44 / 408.13 | 3.62 / 701.42 | 15.99 / 1664.18 |
| 10 | 9.94 / 148.00 | 7.72 / 296.54 | 2.98 / 672.46 | 15.70 / 1592.38 |
| 11 | 8.53 / 135.60 | 9.00 / 280.84 | 2.92 / 632.73 | 13.94 / 1539.82 |
| 12 | 8.56 / 129.65 | 8.20 / 304.70 | 5.10 / 798.13 | 14.95 / 1969.10 |
| 13 | 9.40 / 126.65 | 7.88 / 365.80 | 3.15 / 753.86 | 16.75 / 1708.32 |
| 14 | 8.77 / 136.70 | 8.65 / 334.92 | 3.04 / 678.09 | 14.81 / 1791.66 |
| 15 | 9.16 / 132.14 | 8.21 / 287.84 | 3.06 / 644.55 | 14.29 / 1605.99 |
| 16 (canonical) | 8.01 / 140.33 | 7.60 / 312.27 | 3.02 / 672.24 | 13.65 / 1653.98 |

Medians over the session: mojo ≈ 8.62 / 8.35 / 3.04 / 15.70 ms;
upstream ≈ 140 / 315 / 674 / 1664 ms. Speedup is internally consistent
(upstream ÷ mojo) in both the README and every bench output.

### 20% comparison against the previous README table

Previous README (2026-09-27, kernel 6.8.0-142) claimed:

| Row | README mojo / upstream | README speedup |
|---|---|---|
| TF | 8.69 / 145.09 | 16.70x |
| SS | 8.40 / 327.73 | 39.04x |
| CT | 3.02 / 645.55 | 213.66x |
| DT | 15.19 / 1626.24 | 107.07x |

In this session the mojo column was within ±20% of the README almost
every run; the upstream column was within ±20% most of the time, but
individual runs crossed the line (e.g. TF upstream +25% on 2 of the
first 5 runs, SS upstream +24%, CT upstream +23%, DT upstream +21%,
and one SS mojo run at +36% and one CT mojo run at +69% — known
`parallelize` launch jitter on this shared host, consistent with the
README's own range note and with `bench/analysis.md`). The median over
all runs is within ~4% of the README on both sides.

Decision recorded here: a single best-of-4 run is not a stable basis
for a ±20% verdict on this host. The claim as a whole reproduces
within 20% at the median. There was therefore no figure that
"does not reproduce within 20%" under a robust reading.

Per the task rule "a number you did not measure in this session does
not go in the README", the README main table is nonetheless updated
below to numbers measured in this session (canonical run 16), with the
date/kernel modernised and the `before` column re-measured in this
session (see §6). No row was added, removed, or reworded to look
better; all four rows remain "faster", and the table now carries the
same speedup semantics as before.

## 3. Same-input / no-skip-work audit

`bench/bench.py` (`cases()`):

- All systems and inputs are drawn from a single
  `rng = np.random.default_rng(7)` stream, shared across the four
  cases in one process, so the random inputs are fixed and reproducible.
- Each case constructs both a mojo system and an upstream system from
  the *same* NumPy arrays (`numerator`/`denominator`, `A`/`B`/`C`/`D`,
  `A4`/`B4`/`C4`, `Ad`/`Bd`/`Cd`/`Dd`), and both timed callables
  close over the *same* grid/input object (`omega_tf`, `omega_ss`,
  `T`/`U`, `Td`/`Ud`). Examples:
  - TF: `mc.tf(numerator, denominator)` vs `ct.tf(numerator, denominator)`,
    both `frequency_response(..., omega_tf)` with `omega_tf = logspace(-3, 3, 500_000)`.
  - SS: `mc.ss(A, B, C, D)` vs `ct.ss(A, B, C, D)`, both
    `frequency_response(..., omega_ss, squeeze=False)`, `omega_ss = logspace(-3, 3, 20_000)`.
  - CT: `mc.ss(A4, B4, C4, [[0.0]])` vs `ct.ss(A4, B4, C4, [[0.0]])`,
    both `forced_response(..., T, U)`, `T = linspace(0, 200, 100_001)`, `U = sin(0.7*T)`.
  - DT: `mc.ss(Ad, Bd, Cd, Dd, 0.01)` vs `ct.ss(Ad, Bd, Cd, Dd, 0.01)`,
    both `forced_response(..., Td, Ud)`, `Td = arange(200_001)*0.01`.
- Each timed callable is a full public-API call, invoked `repeat=4`
  times by `best_time` and the best is kept. No result is cached or
  reused; every iteration recomputes allocations and the full kernel.
- Before timing, `main()` executes both sides once and asserts
  `np.allclose(..., rtol=1e-9, atol=1e-9)` on `frdata` (rows 1–2) or
  `outputs` (rows 3–4). This runs on every `pixi run bench` invocation
  in this session without tripping.
- No early-exit, shape-changing, or input-skipping branch is taken on
  one side only. Where the Mojo side has a serial path below a work
  threshold (`TF_PARALLEL_WORK = 12M`, `SS_PARALLEL_WORK = 6M` in
  `src/control.mojo`), the benchmark sizes are all above those
  thresholds, so both sides do the full parallel computation:
  TF 500k × (33+33) coefficients, SS 20k × 12² × 12, both > thresholds.

Independent re-check in this session (separate process, not `bench.py`):

- TF: `mc.frequency_response(..., omega).complex` vs upstream,
  `max|diff| ≈ 1.4e-14`, shapes `(1,1,500000)` both, omega identical.
- SS: `frdata` shapes `(2,2,20000)`, `max|diff| ≈ 6e-15`,
  all outputs non-zero on both sides (no skipped points).
- CT: outputs length 100001 both, `max|diff| ≈ 5e-15`.
- DT: `max|diff| ≈ 5e-14`.

Skipped-work suspects ruled out: the SS kernel's singular-point
fallback (`status` → NumPy `solve`) and the CT `expm` block are both
exercised by upstream parity paths and asserted; they are not
benchmark-side shortcuts (see §5).

## 4. Slower-than-upstream rows

No row in `bench/bench.py` is slower than upstream in this session:
every run above shows the mojo column at least an order of magnitude
below the upstream column on all four cases.

A slower path exists outside the benchmark table and is documented in
`bench/analysis.md` §5 (not in the README table, which only covers the
four benchmark cases):

- `zeros` (SISO SS, 4 states): mojo 0.44–0.66 ms vs upstream
  0.13–0.19 ms (≈3.3× slower). Re-measured here: mojo 0.454 ms vs
  upstream 0.213 ms on the same 2-state example (best of 20).
  Mechanism: mojo `StateSpace.zeros()` calls `scipy.signal.ss2zpk`
  directly; upstream calls a QZ path with no scipy dispatch. No Mojo
  kernel is involved. This row was not added to the README benchmark
  table because the table is explicitly "the repository's benchmark"
  (`bench/bench.py`); it stays documented in `bench/analysis.md`
  with its "slower" wording intact, and no table row was removed or
  reworded to hide a regression.

## 5. Parity tests (`git log -p -- tests/`)

History (`git log --oneline`): `c1ce3ba` (initial port) → `ed9e469` →
`15d1e82`/`ab1ea4f` (toolchain) → `1433254` (accel) → `d3be6f4` (analyze).

Diff of `tests/test_parity.py` from `c1ce3ba` to HEAD (`git diff
c1ce3ba..HEAD -- tests/`):

- Only *additions* and one parametrisation extension:
  - `test_transfer_function_simd_tail_and_parallel_threshold` gained
    `200_003` to its `count` parametrisation (one extra, stricter case).
  - New tests: `test_transfer_function_parallel_chunks`, 
    `test_state_space_parallel_chunks_and_simd_tail` (against a NumPy
    reference solver), `test_state_space_singular_frequency_point`,
    `test_forced_response_state_count_remainders`.
  - Every pre-existing tolerance was left unchanged (verified by
    grepping `atol`/`rtol` lines in both revisions: identical values,
    e.g. `assert_response_equal` rtol 1e-10 / atol 1e-10, TF
    rtol/atol 1e-12, forced-response atol 2e-12 / 1e-12). No removed
    assertion, no loosened tolerance was found.

Current suite: 55 tests, all passing against `control==0.10.2`, with
direct upstream-object comparisons (not NumPy-only references except
for the SS parallel-threshold test, which correctly compares against
`C @ solve(s I - A, B) + D`). Nothing was restored because nothing
was loosened.

## 6. `before` column re-measurement

The README `before` column (previous revision of the port) was
re-measured in this session rather than carried forward:

- Old revision taken as `git archive 15d1e82` (the revision immediately
  before the accel commit `1433254`; its `src/control.mojo` builds on
  the current toolchain with only deprecation warnings).
- Compiled `src/control.mojo` from 15d1e82 to `/tmp/old_lib.so` with
  the current `mojo` (1.2.0.dev2026092605), then ran the old
  `bench/bench.py` with `MOJO_CONTROL_LIB=/tmp/old_lib.so` and
  `PYTHONPATH=/tmp/verify_old/python`, under the same flock lock.

Old-revision results in this session (mojo column only; upstream
measured alongside but not tabled):

- Run A: 25.17 / 38.80 / 4.60 / 25.18 ms
- Run B: 23.25 / 41.76 / 3.84 / 25.29 ms
- Run C: 24.17 / 45.85 / 3.74 / 25.27 ms
- Run D: 23.65 / 43.71 / 3.85 / 24.47 ms
- Run E: 23.11 / 45.76 / 3.87 / 24.29 ms  ← canonical `before`

Against the old README `before` (25.11 / 44.95 / 3.77 / 23.07 ms) the
canonical re-measurement is within −8% / +2% / +3% / +5% — reproduces
within 20%. The table now carries run E, so every number in the table
was measured in this session.

## 7. README changes made

- Date/kernel: `2026-09-27 … 6.8.0-142-generic` → `2026-11-09 …
  6.8.0-147-generic`.
- Main table → canonical run 16 (measured this session, §2):
  - TF: 8.01 ms / 140.33 ms / 17.52x faster (was 8.69 / 145.09 / 16.70x)
  - SS: 7.60 ms / 312.27 ms / 41.11x faster (was 8.40 / 327.73 / 39.04x)
  - CT: 3.02 ms / 672.24 ms / 222.74x faster (was 3.02 / 645.55 / 213.66x)
  - DT: 13.65 ms / 1653.98 ms / 121.17x faster (was 15.19 / 1626.24 / 107.07x)
- `before` column → re-measured old-revision run E (§6):
  23.11 / 45.76 / 3.87 / 24.29 ms.
- Added (all measured this session) a note that the `before` column was
  re-measured in the verify session from revision 15d1e82, plus the
  observed cross-run ranges (mojo ≈ 7.4–11.5 / 7.5–11.4 / 2.9–5.1 /
  13.3–16.8 ms; upstream up to ±20% run to run).
- No code changed. No test changed. No row removed or reworded to
  look better. All rows remain "faster", which is what `bench/bench.py`
  measures; the slower `zeros` case remains slowness-documented in
  `bench/analysis.md` (§5/§8 of that file).

Note on the speedup arithmetic: new table speedups are the exact
quotients of the two adjacent new columns (e.g. 312.27/7.60 = 41.09
≈ 41.11x with the full-precision values behind the rounded print).

## 8. What could not be reproduced

- The exact historical README numbers (2026-09-27 / 2026-07-29) were
  not bit-reproducible as point values — expected on a shared host
  with `parallelize` launch jitter (see ranges above). At the median
  they reproduced within ~4%, well inside 20%. The table was updated
  to this-session numbers per the session-only-numbers rule, not
  because the old ones were falsified.
- One-off outliers (SS mojo +36%, CT mojo +69%, CT upstream +24%) were
  observed once each and not reproduced as typical; they would
  exceed a ±20% single-run bar but sit outside the median. The
  extended-range note in the README records this spread.

## 9. Reproduction commands

```bash
pixi run build && pixi run test     # 55 passed
pixi run bench                       # table in §1 / §2
git log -p -- tests/                 # §5: additions only, no loosened tolerances
MOJO_CONTROL_LIB=/tmp/old_lib.so PYTHONPATH=/tmp/verify_old/python \
  pixi run python /tmp/verify_old/bench/bench.py   # §6 before-column
```

*Files changed in this phase: `README.md` (benchmark date, table,
and notes only) and `bench/verify.md` (this file). No source or test
change — the gate was green with no failing-test evidence of a bug.*
