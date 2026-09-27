# mojo-control

`mojo-control` is a Mojo implementation of the compute-heavy core of the
[Python Control Systems Library](https://python-control.readthedocs.io/). It
provides a NumPy-facing Python API with the same names and call signatures for
the covered state-space and transfer-function subset. Import it as
`mojo_control` instead of `control`.

This is a focused port, not a reimplementation of every part of
python-control. Its useful fast paths are batched transfer-function evaluation,
dense state-space frequency response, and long continuous- or discrete-time
state recurrences.

## Install

The checked-in Pixi environment pins the tested Mojo nightly together with the
matching `max` package, and installs Python, NumPy, SciPy, pytest, and
`control==0.10.2`:

```bash
pixi install
pixi run build
```

The build produces `dist/libmojo-control.so`. The Pixi activation environment
adds `python/` to `PYTHONPATH`, so no separate editable install is required.

## Usage

```python
import numpy as np
import mojo_control as ct

plant = ct.tf([1.0], [1.0, 2.0, 1.0])
closed_loop = ct.feedback(plant, 1)

omega = np.logspace(-2, 2, 500)
response = ct.frequency_response(closed_loop, omega)

time = np.linspace(0.0, 10.0, 2001)
step = ct.step_response(closed_loop, time)

print(response.magnitude.shape)  # (500,)
print(step.outputs[-1])          # approximately 0.5
```

The same code is checked in and can be run from the repository:

```bash
pixi run python examples/basic.py
```

## Covered subset

- SISO `TransferFunction`, `tf`, `tfdata`, `tf("s")`, and `tf("z")`
- Dense float64 SISO/MIMO `StateSpace`, `ss`, and `ssdata`
- Transfer-function addition, subtraction, multiplication, division, integer
  powers, and feedback
- State-space addition, subtraction, scalar multiplication, series
  multiplication, append, and feedback
- `series`, `parallel`, `append`, and `feedback`
- `evalfr`, `frequency_response`, poles, SISO zeros, and DC gain for continuous
  and discrete systems
- `forced_response`, `step_response`, and `impulse_response` on equally spaced
  time grids
- `c2d` through SciPy's zero-order hold, first-order hold, impulse, bilinear,
  Euler, backward-difference, and generalized bilinear transforms
- Python-control-like response objects with `.omega`, `.magnitude`, `.phase`,
  `.frdata`, `.time`, `.outputs`, and `.states`

The parity suite compares all of these paths directly with
`control==0.10.2`. It covers MIMO frequency and time response, continuous and
discrete systems, representation conversion, and interconnections.

## Not covered

- MIMO transfer-function matrices and MIMO transmission zeros
- Named-signal interconnections, label metadata, nonlinear systems, FRD
  systems, delays, or parameterized models
- Plotting, root-locus and stability-margin tools, identification, optimal or
  robust control, model reduction, and stochastic-system helpers
- Irregular time grids, discrete time steps that skip samples, and
  `interpolate=True`
- Automatic frequency ranges expressed in hertz and prewarped discretization

`forced_response` requires an explicit time vector. `step_response` and
`impulse_response` provide a simple default grid, but an explicit grid is
recommended when matching an existing analysis. Unsupported options raise
instead of silently taking a different numerical path.

## Benchmarks

Measured on 2026-09-27 on an Intel Xeon E5-2697 v4 at 2.30GHz, Linux
6.8.0-142-generic. These are best-of-four wall-clock measurements of each
library's public Python API after loading the shared library. The benchmark
first asserts numerical agreement and is run through the repository's
machine-wide benchmark lock:

```bash
pixi run bench
```

| Case | mojo-control | control 0.10.2 | Speedup | before |
|---|---:|---:|---:|---:|
| TF frequency response, order 32, 500k points | 8.69 ms | 145.09 ms | 16.70x faster | 25.11 ms |
| SS frequency response, 12 states, 2x2, 20k points | 8.40 ms | 327.73 ms | 39.04x faster | 44.95 ms |
| Continuous forced response, 4 states, 100k samples | 3.02 ms | 645.55 ms | 213.66x faster | 3.77 ms |
| Discrete forced response, 8 states, 2x2, 200k samples | 15.19 ms | 1626.24 ms | 107.07x faster | 23.07 ms |

`before` is this same benchmark against the previous revision of the port,
re-measured on this machine under the same lock. Across eight runs of the
optimized build the mojo-control column ranged over 8.4-14.0 ms, 7.8-9.7 ms,
2.95-5.10 ms and 14.0-15.5 ms. The two threaded frequency cases vary most
because the `parallelize` launch alone costs about 5 ms of wall clock per call
on this toolchain and the host is shared; the single 5.10 ms continuous
response reading is an outlier against the other seven. Results are machine-
and problem-dependent; rerun the benchmark for the systems that matter to
you.

The largest gains come from moving per-frequency dense solves and per-sample
state updates out of Python. On top of that, the frequency kernels are now
split across CPU workers, the state-space assembly and elimination are
SIMD-vectorized, the recurrence drops its per-step state copy, and the
complex outputs are interleaved without a temporary.

### Parallelism

Both frequency kernels use `max.algorithm.parallelize` with an explicit worker
count, on independent chunks of the frequency grid, above a work threshold
(12M coefficient-products for transfer functions, 6M for state-space solves).
Each chunk gets its own scratch slab so the workers never share the matrix.
Below the thresholds the same kernels run serially, because below roughly
5.5 ms of serial work the thread launch costs more than it saves. The
simulation kernel is inherently sequential in time and is not parallelized.

### GPU

There is no GPU path, and it is not because the host API is missing. On the
pinned toolchain (`mojo 1.2.0.dev2026092605`, `max 26.7.0.dev2026092605`)
`max.gpu.host.DeviceContext` with `enqueue_create_buffer`, `enqueue_copy`,
`enqueue_function` and `synchronize` compiles, and `max.gpu.global_idx`,
`thread_idx`, `block_idx` and `block_dim` compile. What is not reachable is
any way to *declare* a device function: a plain `def` is not `DevicePassable`,
`max.gpu` exports no kernel decorator, and `max.gpu.host.compile` rejects
every decoration form. `DeviceFunction` is parameterized by that missing
function type. So the launch path is present and the entry point is not.

The state-space frequency kernel is otherwise a good GPU candidate - about 57
flops per byte at 12 states and 2x2, well above the 2 flops per byte
break-even - so this is worth revisiting on a toolchain that exposes the
device-function declaration. The transfer-function kernel is roughly 5 flops
per byte and the time-domain recurrence is far below that, so neither would
benefit much from a GPU even once one can be built.

## How it works

`src/control.mojo` is one compilation unit exported as a C ABI shared library.
The Python layer passes caller-owned NumPy buffers as integer addresses through
ctypes. Every numerical buffer is C-contiguous float64; state-space matrices
are row-major, inputs and states are time-major inside the simulation kernel,
and complex arrays cross as separate real and imaginary buffers. The Mojo side
does not allocate or retain Python-owned memory.

Transfer functions use batched complex Horner evaluation with hardware-width
float64 SIMD and an explicit scalar tail. Continuous and discrete frequency
grids are passed as contiguous real and imaginary NumPy buffers without an
intermediate complex grid, and magnitude and phase arrays are materialized
only when requested.

State-space frequency response forms `x I - A` for each point and performs a
partial-pivoted complex elimination while solving every input column together.
Assembling the matrix, negating `A` into it, and the row-update inner loop are
SIMD-vectorized; the pivot search and the back substitution stay scalar
because they are strided. The kernel writes its results in `(outputs, inputs,
points)` order so the Python layer can fill the complex output buffer directly
instead of transposing and copying it.

For continuous time response, SciPy computes one exact augmented matrix
exponential for linear input interpolation; Mojo then performs the full state
recurrence and output projection. Discrete systems enter the same recurrence
directly. The recurrence keeps its two state buffers and swaps pointers
instead of copying, unrolls four state rows at a time so the independent dot
products overlap instead of serializing on one accumulator chain, and skips
the second input matrix entirely for discrete systems, where it is zero by
construction.

The library needs the `max` package alongside `mojo` in `pixi.toml`; it ships
in lockstep and provides `max.algorithm.parallelize`.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The test task contains direct upstream parity and boundary-safety cases. The
project is available under the MIT license.
