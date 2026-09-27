"""Control-system objects and operations backed by Mojo kernels."""

from __future__ import annotations

from functools import reduce
from numbers import Number
from operator import index
from typing import Iterator

import numpy as np
from scipy import linalg, signal

from ._lib import addr, f64, lib, scratch_chunks


def _dt(value):
    if value is None:
        return 0
    if value is True:
        return True
    value = float(value)
    return int(value) if value == 0 else value


def _common_dt(left, right):
    left, right = _dt(left), _dt(right)
    if left == right:
        return left
    if left is None or left == 0:
        if right in (None, 0):
            return 0
    if left is True and right not in (None, 0):
        return right
    if right is True and left not in (None, 0):
        return left
    raise ValueError("Systems have incompatible timebases")


def _poly(value) -> np.ndarray:
    result = f64(value).ravel()
    if result.size == 0:
        raise ValueError("Polynomial coefficients cannot be empty")
    nonzero = np.flatnonzero(result)
    return np.array([0.0]) if not nonzero.size else result[nonzero[0] :]


class FrequencyResponseData:
    def __init__(self, response: np.ndarray, omega: np.ndarray, squeeze=None):
        self.frdata = np.asarray(response, dtype=np.complex128)
        self.fresp = self.frdata
        self.omega = np.asarray(omega, dtype=np.float64)
        do_squeeze = squeeze is not False
        data = np.squeeze(self.frdata) if do_squeeze else self.frdata
        self.complex = data
        self._magnitude = None
        self._phase = None

    @property
    def magnitude(self):
        if self._magnitude is None:
            self._magnitude = np.abs(self.complex)
        return self._magnitude

    @property
    def phase(self):
        if self._phase is None:
            self._phase = np.angle(self.complex)
        return self._phase

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.magnitude
        yield self.phase
        yield self.omega

    def __len__(self) -> int:
        return 3


class TimeResponseData:
    def __init__(
        self,
        time: np.ndarray,
        outputs: np.ndarray,
        states: np.ndarray | None = None,
        *,
        return_states: bool = False,
    ):
        self.time = np.asarray(time)
        self.t = self.time
        self.outputs = np.asarray(outputs)
        self.y = self.outputs
        self.states = None if states is None else np.asarray(states)
        self.x = self.states
        self.return_x = return_states

    def __iter__(self):
        yield self.time
        yield self.outputs
        if self.return_x:
            yield self.states

    def __len__(self):
        return 3 if self.return_x else 2


class TransferFunction:
    """SISO transfer function with descending-power coefficients."""

    def __init__(self, *args, **kwargs):
        dt_kw = kwargs.pop("dt", None)
        kwargs.pop("name", None)
        if kwargs:
            raise TypeError(f"Unsupported keyword arguments: {', '.join(kwargs)}")
        if len(args) == 1 and isinstance(args[0], TransferFunction):
            source = args[0]
            numerator = source._num.copy()
            denominator = source._den.copy()
            dt = source.dt if dt_kw is None else dt_kw
        elif len(args) in (2, 3):
            numerator, denominator = args[:2]
            dt = args[2] if len(args) == 3 else (0 if dt_kw is None else dt_kw)
        else:
            raise TypeError("TransferFunction expects (num, den[, dt])")
        self._num = _poly(numerator)
        self._den = _poly(denominator)
        if np.all(self._den == 0):
            raise ValueError("Transfer function denominator cannot be zero")
        self.num = [[self._num]]
        self.den = [[self._den]]
        self.dt = _dt(dt)
        self.ninputs = self.noutputs = 1
        self.nstates = None
        self.shape = (1, 1)

    def issiso(self):
        return True

    def _evaluate_parts(self, xr, xi, shape):
        xr = f64(xr).ravel()
        xi = f64(xi).ravel()
        if xr.size != xi.size:
            raise ValueError("real and imaginary point buffers must have equal length")
        if xr.size == 0:
            return np.empty(shape, dtype=np.complex128)
        real = np.empty(xr.size)
        imag = np.empty(xr.size)
        lib().mctl_tf_eval(
            addr(self._num),
            addr(self._den),
            self._num.size,
            self._den.size,
            addr(xr),
            addr(xi),
            addr(real),
            addr(imag),
            xr.size,
        )
        result = np.empty(xr.size, dtype=np.complex128)
        result.real = real
        result.imag = imag
        return result.reshape(shape)

    def __call__(self, x, squeeze=None):
        points = np.asarray(x, dtype=np.complex128)
        flat = np.ascontiguousarray(points.ravel())
        result = self._evaluate_parts(flat.real, flat.imag, points.shape)
        if squeeze is False:
            return result.reshape((1, 1) + points.shape)
        return result.item() if points.ndim == 0 else result

    def poles(self):
        return np.roots(self._den)

    def zeros(self):
        return np.roots(self._num)

    def dcgain(self):
        point = 1 if self.dt not in (0, None) else 0
        return np.real_if_close(self(point)).item()

    def frequency_response(self, omega=None, squeeze=None):
        return frequency_response(self, omega=omega, squeeze=squeeze)

    def feedback(self, other=1, sign=-1):
        other = _as_tf(other, self.dt)
        return TransferFunction(
            np.polymul(self._num, other._den),
            np.polyadd(
                np.polymul(self._den, other._den),
                -sign * np.polymul(self._num, other._num),
            ),
            _common_dt(self.dt, other.dt),
        )

    def __neg__(self):
        return TransferFunction(-self._num, self._den, self.dt)

    def __add__(self, other):
        other = _as_tf(other, self.dt)
        return TransferFunction(
            np.polyadd(
                np.polymul(self._num, other._den),
                np.polymul(other._num, self._den),
            ),
            np.polymul(self._den, other._den),
            _common_dt(self.dt, other.dt),
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-_as_tf(other, self.dt))

    def __rsub__(self, other):
        return _as_tf(other, self.dt) - self

    def __mul__(self, other):
        if isinstance(other, StateSpace):
            return ss(self) * other
        other = _as_tf(other, self.dt)
        return TransferFunction(
            np.polymul(self._num, other._num),
            np.polymul(self._den, other._den),
            _common_dt(self.dt, other.dt),
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = _as_tf(other, self.dt)
        return TransferFunction(
            np.polymul(self._num, other._den),
            np.polymul(self._den, other._num),
            _common_dt(self.dt, other.dt),
        )

    def __rtruediv__(self, other):
        return _as_tf(other, self.dt) / self

    def __pow__(self, exponent):
        try:
            exponent = index(exponent)
        except TypeError as error:
            raise TypeError("transfer-function powers require an integer exponent") from error
        if exponent == 0:
            return TransferFunction([1], [1], self.dt)
        if exponent < 0:
            return (TransferFunction(self._den, self._num, self.dt)) ** (-exponent)
        result = TransferFunction([1], [1], self.dt)
        base = self
        while exponent:
            if exponent & 1:
                result = result * base
            base = base * base
            exponent //= 2
        return result


class StateSpace:
    """Dense continuous- or discrete-time state-space model."""

    def __init__(self, *args, **kwargs):
        dt_kw = kwargs.pop("dt", None)
        for key in ("name", "inputs", "outputs", "states"):
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(f"Unsupported keyword arguments: {', '.join(kwargs)}")
        if len(args) == 1 and isinstance(args[0], StateSpace):
            source = args[0]
            A, B, C, D = source.A, source.B, source.C, source.D
            dt = source.dt if dt_kw is None else dt_kw
        elif len(args) in (4, 5):
            A, B, C, D = args[:4]
            dt = args[4] if len(args) == 5 else (0 if dt_kw is None else dt_kw)
        else:
            raise TypeError("StateSpace expects (A, B, C, D[, dt])")
        self.A = f64(np.atleast_2d(A), copy=True)
        self.B = f64(np.atleast_2d(B), copy=True)
        self.C = f64(np.atleast_2d(C), copy=True)
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be square")
        n = self.A.shape[0]
        if self.B.shape[0] != n or self.C.shape[1] != n:
            raise ValueError("State-space matrix dimensions are incompatible")
        p, m = self.C.shape[0], self.B.shape[1]
        D = f64(D)
        if D.ndim == 0:
            D = np.full((p, m), D)
        self.D = f64(np.atleast_2d(D), copy=True)
        if self.D.shape != (p, m):
            raise ValueError("D must have shape (outputs, inputs)")
        self.nstates, self.ninputs, self.noutputs = n, m, p
        self.shape = (p, m)
        self.dt = _dt(dt)

    def issiso(self):
        return self.ninputs == self.noutputs == 1

    def _evaluate_parts(self, xr, xi):
        xr = f64(xr).ravel()
        xi = f64(xi).ravel()
        if xr.size != xi.size:
            raise ValueError("real and imaginary point buffers must have equal length")
        count = xr.size
        if count == 0:
            return np.empty((self.noutputs, self.ninputs, 0), dtype=np.complex128)
        n, m, p = self.nstates, self.ninputs, self.noutputs
        real = np.empty((p, m, count))
        imag = np.empty_like(real)
        stride = 2 * n * n + 2 * n * m
        scratch = np.empty(scratch_chunks() * stride)
        status = np.zeros(count, dtype=np.int64)
        lib().mctl_ss_eval(
            addr(self.A),
            addr(self.B),
            addr(self.C),
            addr(self.D),
            n,
            m,
            p,
            addr(xr),
            addr(xi),
            addr(real),
            addr(imag),
            addr(scratch),
            addr(status),
            count,
        )
        result = np.empty((p, m, count), dtype=np.complex128)
        result.real = real
        result.imag = imag
        if np.any(status):
            eye = np.eye(n)
            for index in np.flatnonzero(status):
                point = complex(xr[index], xi[index])
                result[..., index] = self.C @ np.linalg.solve(
                    point * eye - self.A, self.B
                ) + self.D
        return result

    def __call__(self, x, squeeze=None):
        points = np.asarray(x, dtype=np.complex128)
        if points.size == 0:
            result = np.empty(
                (self.noutputs, self.ninputs) + points.shape, dtype=np.complex128
            )
            return result if squeeze is False else np.squeeze(result)
        if self.nstates == 0:
            result = np.broadcast_to(
                self.D[(...,) + (None,) * points.ndim],
                (self.noutputs, self.ninputs) + points.shape,
            ).copy()
            if squeeze is False:
                return result
            result = np.squeeze(result)
            return result.item() if result.ndim == 0 else result
        flat = np.ascontiguousarray(points.ravel())
        result = self._evaluate_parts(flat.real, flat.imag)
        result = result.reshape((self.noutputs, self.ninputs) + points.shape)
        if squeeze is False:
            return result
        result = np.squeeze(result)
        return result.item() if result.ndim == 0 else result

    def poles(self):
        return np.linalg.eigvals(self.A)

    def zeros(self):
        if not self.issiso():
            raise NotImplementedError("MIMO transmission zeros are not covered")
        return signal.ss2zpk(self.A, self.B, self.C, self.D)[0]

    def dcgain(self):
        matrix = (
            np.eye(self.nstates) - self.A
            if self.dt not in (0, None)
            else -self.A
        )
        gain = self.C @ np.linalg.solve(matrix, self.B) + self.D
        return gain.item() if self.issiso() else gain

    def frequency_response(self, omega=None, squeeze=None):
        return frequency_response(self, omega=omega, squeeze=squeeze)

    def feedback(self, other=1, sign=-1):
        if isinstance(other, Number):
            other = StateSpace(
                np.empty((0, 0)),
                np.empty((0, self.noutputs)),
                np.empty((self.ninputs, 0)),
                np.eye(self.ninputs, self.noutputs) * other,
                self.dt,
            )
        else:
            other = ss(other)
        if self.ninputs != other.noutputs or self.noutputs != other.ninputs:
            raise ValueError("State-space systems have incompatible feedback dimensions")
        dt = _common_dt(self.dt, other.dt)
        F = np.eye(self.ninputs) - sign * other.D @ self.D
        solved = np.linalg.solve(F, np.c_[other.D, other.C])
        ED = solved[:, : other.ninputs]
        EC = solved[:, other.ninputs :]
        T1 = np.eye(self.noutputs) + sign * self.D @ ED
        T2 = np.eye(self.ninputs) + sign * ED @ self.D
        A = np.block(
            [
                [
                    self.A + sign * self.B @ ED @ self.C,
                    sign * self.B @ EC,
                ],
                [
                    other.B @ T1 @ self.C,
                    other.A + sign * other.B @ self.D @ EC,
                ],
            ]
        )
        B = np.vstack([self.B @ T2, other.B @ self.D @ T2])
        C = np.hstack([T1 @ self.C, sign * self.D @ EC])
        D = self.D @ T2
        return StateSpace(A, B, C, D, dt)

    def append(self, other):
        return append(self, other)

    def __neg__(self):
        return StateSpace(self.A, self.B, -self.C, -self.D, self.dt)

    def __add__(self, other):
        if isinstance(other, Number):
            return StateSpace(self.A, self.B, self.C, self.D + other, self.dt)
        other = ss(other)
        if self.shape != other.shape:
            raise ValueError("Systems have incompatible input/output dimensions")
        return StateSpace(
            linalg.block_diag(self.A, other.A),
            np.vstack([self.B, other.B]),
            np.hstack([self.C, other.C]),
            self.D + other.D,
            _common_dt(self.dt, other.dt),
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-ss(other) if not isinstance(other, Number) else -other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if isinstance(other, Number):
            return StateSpace(self.A, self.B * other, self.C, self.D * other, self.dt)
        other = ss(other)
        if self.ninputs != other.noutputs:
            raise ValueError("Systems have incompatible series dimensions")
        A = np.block(
            [
                [other.A, np.zeros((other.nstates, self.nstates))],
                [self.B @ other.C, self.A],
            ]
        )
        B = np.vstack([other.B, self.B @ other.D])
        C = np.hstack([self.D @ other.C, self.C])
        return StateSpace(
            A, B, C, self.D @ other.D, _common_dt(self.dt, other.dt)
        )

    def __rmul__(self, other):
        if isinstance(other, Number):
            return StateSpace(self.A, self.B, other * self.C, other * self.D, self.dt)
        return ss(other) * self


def _as_tf(value, dt=0):
    if isinstance(value, TransferFunction):
        return value
    if isinstance(value, Number):
        return TransferFunction([value], [1], dt)
    return tf(value)


def ss(*args, **kwargs):
    if len(args) == 1:
        source = args[0]
        if isinstance(source, StateSpace):
            return StateSpace(source, **kwargs)
        if isinstance(source, TransferFunction):
            A, B, C, D = signal.tf2ss(source._num, source._den)
            return StateSpace(A, B, C, D, kwargs.pop("dt", source.dt), **kwargs)
        if isinstance(source, Number):
            return StateSpace(
                np.empty((0, 0)),
                np.empty((0, 1)),
                np.empty((1, 0)),
                [[source]],
                kwargs.pop("dt", 0),
                **kwargs,
            )
    return StateSpace(*args, **kwargs)


def tf(*args, **kwargs):
    if len(args) == 1:
        source = args[0]
        if isinstance(source, str):
            if source not in ("s", "z"):
                raise ValueError("Only 's' and 'z' are valid transfer-function variables")
            return TransferFunction(
                [1, 0], [1], kwargs.pop("dt", 0 if source == "s" else True), **kwargs
            )
        if isinstance(source, TransferFunction):
            return TransferFunction(source, **kwargs)
        if isinstance(source, StateSpace):
            if not source.issiso():
                raise NotImplementedError("MIMO state-space to transfer-function conversion")
            num, den = signal.ss2tf(source.A, source.B, source.C, source.D)
            return TransferFunction(num[0], den, kwargs.pop("dt", source.dt), **kwargs)
        if isinstance(source, Number):
            return TransferFunction([source], [1], kwargs.pop("dt", 0), **kwargs)
    return TransferFunction(*args, **kwargs)


def ssdata(sys):
    model = ss(sys)
    return model.A, model.B, model.C, model.D


def tfdata(sys):
    model = tf(sys)
    return model.num, model.den


def evalfr(sys, x, squeeze=None):
    return sys(x, squeeze=squeeze)


def _omega(sys, omega, omega_limits, omega_num):
    if omega is not None:
        return f64(omega).ravel()
    count = 1000 if omega_num is None else int(omega_num)
    if omega_limits is not None:
        return np.logspace(
            np.log10(omega_limits[0]), np.log10(omega_limits[1]), count
        )
    try:
        system_zeros = zeros(sys)
    except NotImplementedError:
        system_zeros = np.array([])
    roots = np.r_[np.abs(poles(sys)), np.abs(system_zeros)]
    roots = roots[np.isfinite(roots) & (roots > 1e-8)]
    if roots.size:
        return np.logspace(
            np.floor(np.log10(roots.min())) - 1,
            np.ceil(np.log10(roots.max())) + 1,
            count,
        )
    return np.logspace(-1, 1, count)


def frequency_response(
    sysdata,
    omega=None,
    omega_limits=None,
    omega_num=None,
    Hz=None,
    squeeze=None,
):
    if Hz not in (None, False):
        raise NotImplementedError("Hz-scaled automatic frequency grids are not covered")
    model = sysdata
    values = _omega(model, omega, omega_limits, omega_num)
    if isinstance(model, TransferFunction):
        if model.dt not in (0, None):
            angles = values * float(model.dt if model.dt is not True else 1.0)
            xr, xi = np.cos(angles), np.sin(angles)
        else:
            xr, xi = np.zeros_like(values), values
        response = model._evaluate_parts(xr, xi, (1, 1, values.size))
    else:
        if model.dt not in (0, None):
            angles = values * float(model.dt if model.dt is not True else 1.0)
            xr, xi = np.cos(angles), np.sin(angles)
        else:
            xr, xi = np.zeros_like(values), values
        response = model._evaluate_parts(xr, xi)
    return FrequencyResponseData(response, values, squeeze=squeeze)


def poles(sys):
    return sys.poles()


def zeros(sys):
    return sys.zeros()


def dcgain(sys):
    return sys.dcgain()


def append(*systems):
    models = [ss(system) for system in systems]
    if not models:
        raise ValueError("append requires at least one system")
    dt = reduce(_common_dt, (model.dt for model in models))
    A = linalg.block_diag(*(model.A for model in models))
    n = sum(model.nstates for model in models)
    m = sum(model.ninputs for model in models)
    p = sum(model.noutputs for model in models)
    B, C, D = np.zeros((n, m)), np.zeros((p, n)), np.zeros((p, m))
    ns = ms = ps = 0
    for model in models:
        B[ns : ns + model.nstates, ms : ms + model.ninputs] = model.B
        C[ps : ps + model.noutputs, ns : ns + model.nstates] = model.C
        D[ps : ps + model.noutputs, ms : ms + model.ninputs] = model.D
        ns += model.nstates
        ms += model.ninputs
        ps += model.noutputs
    return StateSpace(A, B, C, D, dt)


def series(*systems, **kwargs):
    if kwargs:
        raise TypeError("Named signal connections are not covered")
    if not systems:
        raise ValueError("series requires at least one system")
    return reduce(lambda result, nxt: nxt * result, systems[1:], systems[0])


def parallel(*systems, **kwargs):
    if kwargs:
        raise TypeError("Named signal connections are not covered")
    if not systems:
        raise ValueError("parallel requires at least one system")
    return reduce(lambda result, nxt: result + nxt, systems[1:], systems[0])


def feedback(sys1, sys2=1, sign=-1, **kwargs):
    if kwargs:
        raise TypeError("Named signal connections are not covered")
    return sys1.feedback(sys2, sign)


def _time_and_input(model, timepts, inputs, transpose):
    if timepts is None:
        raise ValueError("timepts is required by this covered subset")
    T = f64(timepts).ravel()
    if T.size < 2:
        raise ValueError("timepts must contain at least two points")
    delta = np.diff(T)
    if not np.allclose(delta, delta[0]):
        raise ValueError("timepts must be equally spaced")
    U = f64(inputs)
    if U.ndim == 0:
        U = np.full((model.ninputs, T.size), U)
    elif U.ndim == 1:
        if model.ninputs != 1 or U.size != T.size:
            raise ValueError("SISO input must have one value per time point")
        U = U.reshape(1, -1)
    elif transpose:
        U = U.T
    if U.shape != (model.ninputs, T.size):
        raise ValueError("inputs must have shape (ninputs, len(timepts))")
    return T, f64(U.T)


def forced_response(
    sysdata,
    timepts=None,
    inputs=0.0,
    initial_state=0.0,
    transpose=False,
    params=None,
    interpolate=False,
    return_states=None,
    squeeze=None,
    **kwargs,
):
    if params is not None:
        raise NotImplementedError("parameterized nonlinear systems are not covered")
    if interpolate:
        raise NotImplementedError("discrete input interpolation is not covered")
    if kwargs:
        timepts = kwargs.pop("T", timepts)
        inputs = kwargs.pop("U", inputs)
        initial_state = kwargs.pop("X0", initial_state)
        if kwargs:
            raise TypeError(f"Unsupported keyword arguments: {', '.join(kwargs)}")
    model = ss(sysdata)
    T, U = _time_and_input(model, timepts, inputs, transpose)
    dt = float(T[1] - T[0])
    if model.dt in (0, None):
        n, m = model.nstates, model.ninputs
        block = np.block(
            [
                [model.A * dt, model.B * dt, np.zeros((n, m))],
                [np.zeros((m, n + m)), np.eye(m)],
                [np.zeros((m, n + 2 * m))],
            ]
        )
        exp_block = linalg.expm(block)
        Ad = exp_block[:n, :n]
        B1 = exp_block[:n, n + m :]
        B0 = exp_block[:n, n : n + m] - B1
    else:
        sample = dt if model.dt is True else float(model.dt)
        if not np.isclose(dt, sample):
            raise ValueError("time step must match the discrete system sampling time")
        Ad, B0 = model.A, model.B
        B1 = np.zeros_like(B0)
    x0 = f64(initial_state)
    if x0.ndim == 0:
        x0 = np.full(model.nstates, x0)
    x = f64(x0.ravel(), copy=True)
    if x.size != model.nstates:
        raise ValueError("initial_state has the wrong length")
    if model.nstates == 0:
        outputs = (U @ model.D.T).T
        if squeeze is not False:
            outputs = np.squeeze(outputs)
        if transpose:
            outputs = np.asarray(outputs).T
        states = np.empty((T.size, 0)) if transpose else np.empty((0, T.size))
        return TimeResponseData(T, outputs, states, return_states=bool(return_states))
    next_x = np.empty(max(1, model.nstates))
    y = np.empty((T.size, model.noutputs))
    states = np.empty((T.size, model.nstates))
    Ad = f64(Ad)
    B0 = f64(B0)
    B1 = f64(B1)
    lib().mctl_ss_simulate(
        addr(Ad),
        addr(B0),
        addr(B1),
        addr(model.C),
        addr(model.D),
        addr(U),
        addr(x),
        addr(next_x),
        addr(y),
        addr(states),
        T.size,
        model.nstates,
        model.ninputs,
        model.noutputs,
        1 if model.dt in (0, None) else 0,
    )
    outputs = y.T
    state_data = states.T
    if squeeze is not False:
        outputs = np.squeeze(outputs)
    if transpose:
        outputs = np.asarray(outputs).T
        state_data = state_data.T
    return TimeResponseData(
        T,
        outputs,
        state_data,
        return_states=bool(return_states),
    )


def step_response(
    sysdata,
    timepts=None,
    initial_state=0.0,
    input_indices=None,
    output_indices=None,
    timepts_num=None,
    transpose=False,
    return_states=False,
    squeeze=None,
    params=None,
    **kwargs,
):
    if params is not None:
        raise NotImplementedError("parameterized nonlinear systems are not covered")
    if kwargs:
        raise TypeError(f"Unsupported keyword arguments: {', '.join(kwargs)}")
    model = ss(sysdata)
    if timepts is None:
        count = 100 if timepts_num is None else int(timepts_num)
        timepts = (
            np.arange(count) * float(model.dt if model.dt is not True else 1.0)
            if model.dt not in (0, None)
            else np.linspace(0, 10, count)
        )
    selected_inputs = (
        range(model.ninputs)
        if input_indices is None
        else [int(input_indices)]
    )
    selected_outputs = (
        np.arange(model.noutputs)
        if output_indices is None
        else np.atleast_1d(output_indices)
    )
    ys, xs = [], []
    for index in selected_inputs:
        U = np.zeros((model.ninputs, len(timepts)))
        U[index] = 1.0
        response = forced_response(
            model,
            timepts,
            U,
            initial_state,
            transpose=False,
            return_states=True,
            squeeze=False,
        )
        ys.append(response.outputs[selected_outputs])
        xs.append(response.states)
    outputs = np.stack(ys, axis=1)
    states = np.stack(xs, axis=1)
    if squeeze is not False:
        outputs = np.squeeze(outputs)
        states = np.squeeze(states)
    if transpose:
        outputs = np.moveaxis(outputs, -1, 0)
        states = np.moveaxis(states, -1, 0)
    return TimeResponseData(
        np.asarray(timepts), outputs, states, return_states=return_states
    )


def impulse_response(
    sysdata,
    timepts=None,
    input_indices=None,
    output_indices=None,
    timepts_num=None,
    transpose=False,
    return_states=False,
    squeeze=None,
    **kwargs,
):
    if kwargs:
        raise TypeError(f"Unsupported keyword arguments: {', '.join(kwargs)}")
    model = ss(sysdata)
    if timepts is None:
        count = 100 if timepts_num is None else int(timepts_num)
        timepts = (
            np.arange(count) * float(model.dt if model.dt is not True else 1.0)
            if model.dt not in (0, None)
            else np.linspace(0, 10, count)
        )
    selected_inputs = (
        range(model.ninputs)
        if input_indices is None
        else [int(input_indices)]
    )
    selected_outputs = (
        np.arange(model.noutputs)
        if output_indices is None
        else np.atleast_1d(output_indices)
    )
    ys, xs = [], []
    for index in selected_inputs:
        U = np.zeros((model.ninputs, len(timepts)))
        if model.dt not in (0, None):
            sample = float(model.dt if model.dt is not True else timepts[1] - timepts[0])
            U[index, 0] = 1.0 / sample
            x0 = np.zeros(model.nstates)
        else:
            x0 = model.B[:, index]
        response = forced_response(
            model,
            timepts,
            U,
            x0,
            transpose=False,
            return_states=True,
            squeeze=False,
        )
        ys.append(response.outputs[selected_outputs])
        xs.append(response.states)
    outputs = np.stack(ys, axis=1)
    states = np.stack(xs, axis=1)
    if squeeze is not False:
        outputs = np.squeeze(outputs)
        states = np.squeeze(states)
    if transpose:
        outputs = np.moveaxis(outputs, -1, 0)
        states = np.moveaxis(states, -1, 0)
    return TimeResponseData(
        np.asarray(timepts), outputs, states, return_states=return_states
    )


def c2d(
    sysc,
    Ts,
    method="zoh",
    alpha=None,
    prewarp_frequency=None,
    name=None,
    copy_names=True,
    **kwargs,
):
    if prewarp_frequency is not None:
        raise NotImplementedError("prewarped discretization is not covered")
    if kwargs:
        raise TypeError(f"Unsupported keyword arguments: {', '.join(kwargs)}")
    if isinstance(sysc, TransferFunction):
        result = signal.cont2discrete(
            (sysc._num, sysc._den), Ts, method=method, alpha=alpha
        )
        return TransferFunction(np.squeeze(result[0]), result[1], Ts)
    model = ss(sysc)
    result = signal.cont2discrete(
        (model.A, model.B, model.C, model.D),
        Ts,
        method=method,
        alpha=alpha,
    )
    return StateSpace(*result[:4], Ts)
