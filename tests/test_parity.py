"""Numerical and behavioral parity with python-control 0.10.2."""

import control as ct
import numpy as np
import pytest
import warnings

import mojo_control as mc


def assert_response_equal(ours, theirs, points, *, atol=1e-10):
    actual = mc.evalfr(ours, points, squeeze=False)
    expected = ct.evalfr(theirs, points, squeeze=False)
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, rtol=1e-10, atol=atol)


def assert_roots_equal(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.size == expected.size
    for root in actual:
        index = np.argmin(np.abs(expected - root))
        assert root == pytest.approx(expected[index], abs=1e-10)
        expected = np.delete(expected, index)


@pytest.fixture
def siso_pair():
    matrices = (
        [[0.0, 1.0], [-2.0, -3.0]],
        [[0.0], [1.0]],
        [[1.0, 0.4]],
        [[0.2]],
    )
    return mc.ss(*matrices), ct.ss(*matrices)


@pytest.fixture
def mimo_pair():
    matrices = (
        [[-1.0, 0.2, 0.0], [0.0, -2.0, 0.3], [0.1, 0.0, -3.0]],
        [[1.0, 0.0], [0.2, 1.0], [0.0, -0.3]],
        [[1.0, 0.0, 0.5], [0.0, 1.0, -0.2]],
        [[0.1, 0.0], [0.0, -0.2]],
    )
    return mc.ss(*matrices), ct.ss(*matrices)


def test_transfer_function_constructor_and_data():
    ours = mc.TransferFunction([2, 4], [1, 3, 2])
    theirs = ct.TransferFunction([2, 4], [1, 3, 2])
    assert ours.shape == theirs.shape == (1, 1)
    assert ours.dt == theirs.dt == 0
    onum, oden = mc.tfdata(ours)
    tnum, tden = ct.tfdata(theirs)
    assert np.array_equal(onum[0][0], tnum[0][0])
    assert np.array_equal(oden[0][0], tden[0][0])


@pytest.mark.parametrize("variable", ["s", "z"])
def test_transfer_function_variable(variable):
    ours = mc.tf(variable)
    theirs = ct.tf(variable)
    assert ours.dt == theirs.dt
    points = np.array([0.2 + 0.5j, -0.7 + 1.3j])
    assert_response_equal(ours, theirs, points)


def test_transfer_function_evalfr_and_frequency_response():
    ours = mc.tf([1.0, 2.0], [1.0, 0.5, 3.0])
    theirs = ct.tf([1.0, 2.0], [1.0, 0.5, 3.0])
    points = np.array([-1.5 + 0.2j, 0.7j, 2.0 - 0.4j])
    assert_response_equal(ours, theirs, points)
    omega = np.logspace(-3, 3, 701)
    actual = mc.frequency_response(ours, omega)
    expected = ct.frequency_response(theirs, omega)
    assert np.array_equal(actual.omega, expected.omega)
    assert np.allclose(actual.complex, expected.complex, rtol=1e-12, atol=1e-12)
    magnitude, phase, returned_omega = actual
    assert np.allclose(magnitude, expected.magnitude)
    assert np.allclose(phase, expected.phase)
    assert np.array_equal(returned_omega, omega)


def test_discrete_transfer_function_frequency_response():
    ours = mc.tf([0.1, 0.05], [1.0, -1.2, 0.4], 0.2)
    theirs = ct.tf([0.1, 0.05], [1.0, -1.2, 0.4], 0.2)
    omega = np.linspace(0.0, 12.0, 513)
    assert np.allclose(
        mc.frequency_response(ours, omega).complex,
        ct.frequency_response(theirs, omega).complex,
        rtol=1e-11,
        atol=1e-11,
    )


@pytest.mark.parametrize("count", [4097, 131_073])
def test_transfer_function_simd_tail_and_parallel_threshold(count):
    ours = mc.tf([0.7, -1.2, 0.3], [1.0, 0.4, 2.0, -0.1])
    theirs = ct.tf([0.7, -1.2, 0.3], [1.0, 0.4, 2.0, -0.1])
    omega = np.linspace(0.01, 20.0, count)
    assert np.allclose(
        mc.frequency_response(ours, omega).complex,
        ct.frequency_response(theirs, omega).complex,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "operation",
    [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a / b,
        lambda a, b: a**3,
        lambda a, b: a.feedback(b),
    ],
)
def test_transfer_function_arithmetic(operation):
    ours_a, ours_b = mc.tf([1, 2], [1, 3, 4]), mc.tf([2, -1], [1, 5])
    theirs_a, theirs_b = ct.tf([1, 2], [1, 3, 4]), ct.tf([2, -1], [1, 5])
    points = np.array([0.2j, 1.3j, 5.0j])
    assert_response_equal(
        operation(ours_a, ours_b), operation(theirs_a, theirs_b), points
    )


def test_transfer_function_analysis():
    ours = mc.tf([2, 6], [1, 6, 8])
    theirs = ct.tf([2, 6], [1, 6, 8])
    assert_roots_equal(mc.poles(ours), ct.poles(theirs))
    assert_roots_equal(mc.zeros(ours), ct.zeros(theirs))
    assert mc.dcgain(ours) == pytest.approx(ct.dcgain(theirs))


def test_state_space_constructor_and_data(siso_pair):
    ours, theirs = siso_pair
    for actual, expected in zip(mc.ssdata(ours), ct.ssdata(theirs)):
        assert np.array_equal(actual, expected)
    assert ours.nstates == theirs.nstates
    assert ours.ninputs == theirs.ninputs
    assert ours.noutputs == theirs.noutputs


def test_state_space_evalfr_siso(siso_pair):
    ours, theirs = siso_pair
    points = np.array([-2.0 + 0.3j, 0.1j, 3.2j])
    assert_response_equal(ours, theirs, points)
    assert np.isscalar(mc.evalfr(ours, 1j))


def test_state_space_evalfr_mimo(mimo_pair):
    ours, theirs = mimo_pair
    points = np.array([0.0 + 0.2j, 1.7j, -0.5 + 2.0j, 10j])
    assert_response_equal(ours, theirs, points)
    actual = mc.evalfr(ours, 1j, squeeze=False)
    assert actual.shape == (2, 2)


def test_state_space_frequency_response_mimo(mimo_pair):
    ours, theirs = mimo_pair
    omega = np.logspace(-2, 2, 301)
    actual = mc.frequency_response(ours, omega, squeeze=False)
    expected = ct.frequency_response(theirs, omega, squeeze=False)
    assert actual.frdata.shape == (2, 2, omega.size)
    assert np.allclose(actual.frdata, expected.frdata, rtol=1e-10, atol=1e-11)


def test_state_space_analysis(siso_pair):
    ours, theirs = siso_pair
    assert_roots_equal(ours.poles(), theirs.poles())
    assert_roots_equal(ours.zeros(), theirs.zeros())
    assert ours.dcgain() == pytest.approx(theirs.dcgain())


@pytest.mark.parametrize("kind", ["add", "multiply", "feedback"])
def test_state_space_interconnections(kind):
    o1 = mc.ss([[-1.0]], [[1.0]], [[2.0]], [[0.1]])
    o2 = mc.ss([[-2.0]], [[0.5]], [[-1.0]], [[0.2]])
    t1 = ct.ss([[-1.0]], [[1.0]], [[2.0]], [[0.1]])
    t2 = ct.ss([[-2.0]], [[0.5]], [[-1.0]], [[0.2]])
    if kind == "add":
        ours, theirs = o1 + o2, t1 + t2
    elif kind == "multiply":
        ours, theirs = o1 * o2, t1 * t2
    else:
        ours, theirs = o1.feedback(o2), t1.feedback(t2)
    assert_response_equal(ours, theirs, np.array([0.1j, 1j, 7j]))


def test_series_parallel_and_feedback_functions():
    systems_o = [mc.tf([1], [1, 1]), mc.tf([2], [1, 2]), mc.tf([1, 1], [1, 3])]
    systems_t = [ct.tf([1], [1, 1]), ct.tf([2], [1, 2]), ct.tf([1, 1], [1, 3])]
    points = np.array([0.2j, 2j])
    assert_response_equal(mc.series(*systems_o), ct.series(*systems_t), points)
    assert_response_equal(mc.parallel(*systems_o), ct.parallel(*systems_t), points)
    assert_response_equal(
        mc.feedback(systems_o[0], systems_o[1]),
        ct.feedback(systems_t[0], systems_t[1]),
        points,
    )


def test_append(mimo_pair, siso_pair):
    ours = mc.append(mimo_pair[0], siso_pair[0])
    theirs = ct.append(mimo_pair[1], siso_pair[1])
    points = np.array([0.5j, 2j])
    assert ours.shape == theirs.shape == (3, 3)
    assert_response_equal(ours, theirs, points)


def test_representation_conversion(siso_pair):
    ours_ss, theirs_ss = siso_pair
    ours_tf = mc.tf(ours_ss)
    theirs_tf = ct.tf(theirs_ss)
    points = np.array([0.1j, 1j, 10j])
    assert_response_equal(ours_tf, theirs_tf, points)
    assert_response_equal(mc.ss(ours_tf), ct.ss(theirs_tf), points)


def test_forced_response_continuous_siso(siso_pair):
    ours, theirs = siso_pair
    time = np.linspace(0, 8, 1601)
    inputs = np.sin(1.7 * time) + 0.2 * np.cos(0.3 * time)
    actual = mc.forced_response(
        ours, time, inputs, initial_state=[0.2, -0.1], return_states=True
    )
    expected = ct.forced_response(
        theirs, time, inputs, initial_state=[0.2, -0.1], return_states=True
    )
    assert np.allclose(actual.outputs, expected.outputs, atol=2e-12)
    assert np.allclose(actual.states, expected.states, atol=2e-12)
    assert len(tuple(actual)) == len(tuple(expected)) == 3


def test_forced_response_continuous_mimo(mimo_pair):
    ours, theirs = mimo_pair
    time = np.linspace(0, 4, 1001)
    inputs = np.vstack([np.sin(time), np.cos(2 * time)])
    x0 = [0.1, -0.2, 0.3]
    actual = mc.forced_response(ours, time, inputs, x0, return_states=True, squeeze=False)
    expected = ct.forced_response(theirs, time, inputs, x0, return_states=True, squeeze=False)
    assert actual.outputs.shape == expected.outputs.shape == (2, time.size)
    assert np.allclose(actual.outputs, expected.outputs, atol=2e-12)
    assert np.allclose(actual.states, expected.states, atol=2e-12)


def test_forced_response_discrete():
    matrices = (
        [[0.8, 0.1], [0.0, 0.7]],
        [[1.0], [0.2]],
        [[1.0, -0.4]],
        [[0.1]],
        0.05,
    )
    ours, theirs = mc.ss(*matrices), ct.ss(*matrices)
    time = np.arange(2001) * 0.05
    inputs = np.sin(0.4 * time)
    actual = mc.forced_response(ours, time, inputs, [0.1, -0.2], return_states=True)
    expected = ct.forced_response(theirs, time, inputs, [0.1, -0.2], return_states=True)
    assert np.allclose(actual.outputs, expected.outputs, atol=1e-12)
    assert np.allclose(actual.states, expected.states, atol=1e-12)


def test_step_response_siso(siso_pair):
    ours, theirs = siso_pair
    time = np.linspace(0, 7, 701)
    actual = mc.step_response(ours, time)
    expected = ct.step_response(theirs, time)
    assert actual.outputs.shape == expected.outputs.shape == (time.size,)
    assert np.allclose(actual.outputs, expected.outputs, atol=2e-12)


def test_step_response_mimo(mimo_pair):
    ours, theirs = mimo_pair
    time = np.linspace(0, 3, 401)
    actual = mc.step_response(ours, time, return_states=True, squeeze=False)
    expected = ct.step_response(theirs, time, return_states=True, squeeze=False)
    assert actual.outputs.shape == expected.outputs.shape == (2, 2, time.size)
    assert actual.states.shape == expected.states.shape == (3, 2, time.size)
    assert np.allclose(actual.outputs, expected.outputs, atol=2e-12)
    assert np.allclose(actual.states, expected.states, atol=2e-12)


def test_impulse_response_continuous(siso_pair):
    ours, theirs = siso_pair
    ours.D[:] = 0
    theirs.D[:] = 0
    time = np.linspace(0, 6, 601)
    actual = mc.impulse_response(ours, time, return_states=True)
    expected = ct.impulse_response(theirs, time, return_states=True)
    assert np.allclose(actual.outputs, expected.outputs, atol=2e-12)
    assert np.allclose(actual.states, expected.states, atol=2e-12)


def test_impulse_response_discrete():
    ours = mc.ss([[0.8]], [[0.5]], [[2.0]], [[0.1]], 0.2)
    theirs = ct.ss([[0.8]], [[0.5]], [[2.0]], [[0.1]], 0.2)
    time = np.arange(100) * 0.2
    assert np.allclose(
        mc.impulse_response(ours, time).outputs,
        ct.impulse_response(theirs, time).outputs,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("method", "alpha"),
    [
        ("zoh", None),
        ("foh", None),
        ("impulse", None),
        ("bilinear", None),
        ("euler", None),
        ("backward_diff", None),
        ("gbt", 0.3),
    ],
)
def test_c2d_state_space(siso_pair, method, alpha):
    ours, theirs = siso_pair
    if method == "impulse":
        ours.D[:] = 0
        theirs.D[:] = 0
    ours_d = mc.c2d(ours, 0.1, method=method, alpha=alpha)
    theirs_d = ct.c2d(theirs, 0.1, method=method, alpha=alpha)
    points = np.exp(1j * np.array([0.1, 0.7, 2.0]) * 0.1)
    assert ours_d.dt == theirs_d.dt
    assert_response_equal(ours_d, theirs_d, points)


def test_c2d_transfer_function():
    ours = mc.c2d(mc.tf([1, 2], [1, 3, 4]), 0.05)
    theirs = ct.c2d(ct.tf([1, 2], [1, 3, 4]), 0.05)
    points = np.exp(1j * np.array([0.1, 1.0, 5.0]) * 0.05)
    assert_response_equal(ours, theirs, points, atol=1e-9)


def test_validation_errors():
    with pytest.raises(ValueError, match="equally spaced"):
        mc.forced_response(mc.tf([1], [1, 1]), [0, 1, 3], [0, 1, 0])
    with pytest.raises(NotImplementedError, match="MIMO"):
        mc.tf(mc.ss([[-1]], [[1, 2]], [[1], [2]], np.zeros((2, 2))))
    discrete = mc.ss([[0.8]], [[1]], [[1]], [[0]], 0.1)
    reference = ct.ss([[0.8]], [[1]], [[1]], [[0]], 0.1)
    points = np.exp(1j * np.array([0.2, 1.0]) * 0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        reference_feedback = reference.feedback()
    assert_response_equal(discrete.feedback(), reference_feedback, points)


def test_no_silent_complex_narrowing():
    with pytest.raises(TypeError, match="complex"):
        mc.tf([1 + 1j], [1, 2])
    with pytest.raises(TypeError, match="complex"):
        mc.ss([[-1 + 1j]], [[1]], [[1]], [[0]])
    with pytest.raises(TypeError, match="complex"):
        mc.forced_response(mc.tf([1], [1, 1]), [0, 1], [0, 1j])
    with pytest.raises(TypeError, match="integer"):
        mc.tf([1], [1, 1]) ** 1.5


def test_empty_frequency_grid_and_static_gain():
    gain = mc.ss(3)
    assert mc.evalfr(gain, np.array([]), squeeze=False).shape == (1, 1, 0)
    time = np.linspace(0, 1, 11)
    inputs = np.arange(time.size, dtype=float)
    response = mc.forced_response(gain, time, inputs, return_states=True)
    assert np.array_equal(response.outputs, 3 * inputs)
    assert response.states.shape == (0, time.size)


def test_documented_arithmetic_variants(siso_pair):
    ours, theirs = siso_pair
    points = np.array([0.2j, 2j])
    assert_response_equal(ours - ours, theirs - theirs, points)
    assert_response_equal(2 * ours, 2 * theirs, points)
    tf_ours, tf_theirs = mc.tf([1, 1], [1, 2]), ct.tf([1, 1], [1, 2])
    for exponent in (-2, 0, 2):
        assert_response_equal(tf_ours**exponent, tf_theirs**exponent, points)
