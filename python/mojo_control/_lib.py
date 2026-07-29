"""ctypes bridge to the Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB_PATH = os.environ.get(
    "MOJO_CONTROL_LIB", os.path.join(ROOT, "dist", "libmojo-control.so")
)

I = ctypes.c_int64

_SIGNATURES = {
    "mctl_tf_eval": ([I] * 9, None),
    "mctl_ss_eval": ([I] * 17, None),
    "mctl_ss_simulate": ([I] * 14, None),
}

_library: ctypes.CDLL | None = None


def build() -> str:
    subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        check=True,
        cwd=ROOT,
    )
    return LIB_PATH


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB_PATH):
            build()
        _library = ctypes.CDLL(LIB_PATH)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def f64(value, *, copy: bool = False) -> np.ndarray:
    if np.iscomplexobj(value):
        raise TypeError("complex values are not supported; expected real float64 data")
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 0:
        return result.copy() if copy else result
    if copy:
        return np.array(result, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(result, dtype=np.float64)


def addr(array: np.ndarray) -> int:
    if array.dtype != np.float64 and array.dtype != np.int64:
        raise TypeError("FFI buffers must contain float64 or int64 data")
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    if array.size == 0:
        raise ValueError("empty buffers cannot cross the FFI boundary")
    return int(array.ctypes.data)
