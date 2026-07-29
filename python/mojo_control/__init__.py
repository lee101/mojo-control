"""A Mojo-accelerated subset of Python Control Systems Library."""

from ._lib import build
from .core import (
    FrequencyResponseData,
    StateSpace,
    TimeResponseData,
    TransferFunction,
    append,
    c2d,
    dcgain,
    evalfr,
    feedback,
    forced_response,
    frequency_response,
    impulse_response,
    parallel,
    poles,
    series,
    ss,
    ssdata,
    step_response,
    tf,
    tfdata,
    zeros,
)

__version__ = "0.1.0"

__all__ = [
    "StateSpace",
    "TransferFunction",
    "FrequencyResponseData",
    "TimeResponseData",
    "ss",
    "tf",
    "ssdata",
    "tfdata",
    "evalfr",
    "frequency_response",
    "forced_response",
    "step_response",
    "impulse_response",
    "feedback",
    "series",
    "parallel",
    "append",
    "poles",
    "zeros",
    "dcgain",
    "c2d",
    "build",
]
