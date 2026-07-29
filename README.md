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

The checked-in Pixi environment pins the tested Mojo nightly and installs
Python, NumPy, SciPy, pytest, and `control==0.10.2`:

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

Measured on 2026-07-29 on an Intel Xeon E5-2697 v4 at 2.30GHz, Linux
6.8.0-136-generic. These are best-of-four wall-clock measurements of each
library's public Python API after loading the shared library. The benchmark
first asserts numerical agreement and is run through the repository's
machine-wide benchmark lock:

```bash
pixi run bench
```

| Case | mojo-control | control 0.10.2 | Speedup |
|---|---:|---:|---:|
| TF frequency response, order 32, 500k points | 12.84 ms | 199.38 ms | 15.52x faster |
| SS frequency response, 12 states, 2x2, 20k points | 48.79 ms | 385.44 ms | 7.90x faster |
| Continuous forced response, 4 states, 100k samples | 6.22 ms | 808.81 ms | 129.99x faster |
| Discrete forced response, 8 states, 2x2, 200k samples | 39.14 ms | 1792.06 ms | 45.79x faster |

The largest gains come from moving per-frequency dense solves and per-sample
state updates out of Python. Transfer-function evaluation also vectorizes
complex Horner steps across frequency points and uses multiple CPU workers
only for large batches. Results are machine- and problem-dependent; rerun the
benchmark for the systems that matter to you.

There is no GPU path.

## How it works

`src/control.mojo` is one compilation unit exported as a C ABI shared library.
The Python layer passes caller-owned NumPy buffers as integer addresses through
ctypes. Every numerical buffer is C-contiguous float64; state-space matrices
are row-major, inputs and states are time-major inside the simulation kernel,
and complex arrays cross as separate real and imaginary buffers. The Mojo side
does not allocate or retain Python-owned memory.

Transfer functions use batched complex Horner evaluation with hardware-width
float64 SIMD, an explicit scalar tail, and thresholded parallel chunks above
131,072 points. Continuous and discrete frequency grids are passed as
contiguous real and imaginary NumPy buffers without an intermediate complex
grid, and magnitude and phase arrays are materialized only when requested.
State-space frequency response forms `x I - A` for each point and performs a
partial-pivoted complex elimination while solving every input column together.
For continuous time response, SciPy computes one exact augmented matrix
exponential for linear input interpolation; Mojo then performs the full state
recurrence and output projection. Discrete systems enter the same recurrence
directly.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The test task contains direct upstream parity and boundary-safety cases. The
project is available under the MIT license.
