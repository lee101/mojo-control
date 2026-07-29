"""Benchmarks against python-control on identical systems and inputs."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import control as ct
import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import mojo_control as mc  # noqa: E402


def best_time(function, repeat=4):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def cases():
    rng = np.random.default_rng(7)

    numerator = rng.normal(size=33)
    denominator = np.r_[1.0, rng.normal(scale=0.05, size=32)]
    omega_tf = np.logspace(-3, 3, 500_000)
    mojo_tf, control_tf = mc.tf(numerator, denominator), ct.tf(numerator, denominator)
    yield (
        "TF frequency response, order 32, 500k points",
        lambda: mc.frequency_response(mojo_tf, omega_tf),
        lambda: ct.frequency_response(control_tf, omega_tf),
    )
    n = 12
    A = rng.normal(scale=0.08, size=(n, n)) - np.diag(np.linspace(1, 3, n))
    B = rng.normal(size=(n, 2))
    C = rng.normal(size=(2, n))
    D = rng.normal(scale=0.01, size=(2, 2))
    omega_ss = np.logspace(-3, 3, 20_000)
    mojo_ss, control_ss = mc.ss(A, B, C, D), ct.ss(A, B, C, D)
    yield (
        "SS frequency response, 12 states, 2x2, 20k points",
        lambda: mc.frequency_response(mojo_ss, omega_ss, squeeze=False),
        lambda: ct.frequency_response(control_ss, omega_ss, squeeze=False),
    )

    A4 = rng.normal(scale=0.1, size=(4, 4)) - np.diag([1.0, 1.5, 2.0, 3.0])
    B4 = rng.normal(size=(4, 1))
    C4 = rng.normal(size=(1, 4))
    T = np.linspace(0, 200, 100_001)
    U = np.sin(0.7 * T)
    mojo_ct, control_ct = mc.ss(A4, B4, C4, [[0.0]]), ct.ss(A4, B4, C4, [[0.0]])
    yield (
        "Continuous forced response, 4 states, 100k samples",
        lambda: mc.forced_response(mojo_ct, T, U),
        lambda: ct.forced_response(control_ct, T, U),
    )

    n = 8
    Ad = rng.normal(scale=0.025, size=(n, n)) + np.eye(n) * 0.85
    Bd = rng.normal(size=(n, 2))
    Cd = rng.normal(size=(2, n))
    Dd = np.zeros((2, 2))
    Td = np.arange(200_001) * 0.01
    Ud = np.ascontiguousarray(
        np.vstack([np.sin(0.4 * Td), np.cos(0.9 * Td)])
    )
    mojo_dt = mc.ss(Ad, Bd, Cd, Dd, 0.01)
    control_dt = ct.ss(Ad, Bd, Cd, Dd, 0.01)
    yield (
        "Discrete forced response, 8 states, 2x2, 200k samples",
        lambda: mc.forced_response(mojo_dt, Td, Ud),
        lambda: ct.forced_response(control_dt, Td, Ud),
    )


def main():
    print(f"Machine: {cpu_name()} ({platform.system()} {platform.release()})")
    print()
    print("| Case | mojo-control | control 0.10.2 | Speedup |")
    print("|---|---:|---:|---:|")
    for name, ours, theirs in cases():
        ours_result = ours()
        theirs_result = theirs()
        if hasattr(ours_result, "frdata"):
            assert np.allclose(
                ours_result.frdata, theirs_result.frdata, rtol=1e-9, atol=1e-9
            )
        else:
            assert np.allclose(
                ours_result.outputs, theirs_result.outputs, rtol=1e-9, atol=1e-9
            )
        ours_seconds = best_time(ours)
        theirs_seconds = best_time(theirs)
        ratio = theirs_seconds / ours_seconds
        label = "faster" if ratio >= 1 else "slower"
        print(
            f"| {name} | {ours_seconds * 1e3:.2f} ms | "
            f"{theirs_seconds * 1e3:.2f} ms | {ratio:.2f}x {label} |"
        )


if __name__ == "__main__":
    main()
