import numpy as np
import pytest

from ezpit.core.smoothing import (
    bandwidth_to_lambda,
    batch_smooth_whittaker,
    make_dT_d,
    noise_gain_to_lambda,
    smooth_like_savitzky_golay,
    smooth_whittaker,
    smooth_with_noise_gain,
)


def test_smooth_whittaker_constant_signal_is_invariant():
    # a constant signal has zero 1st-order difference, so the penalty term is
    # zero at z=y for ANY lambda -- z=y is the exact minimiser.
    y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    smoothed = smooth_whittaker(y, lambda_=100.0, order=1)
    assert np.allclose(smoothed, y)


def test_smooth_whittaker_linear_signal_is_invariant():
    # a linear signal has zero 2nd-order difference, so it passes through
    # order=2 smoothing unchanged for ANY lambda, by the same argument.
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    smoothed = smooth_whittaker(y, lambda_=100.0, order=2)
    assert np.allclose(smoothed, y)


def test_smooth_whittaker_rejects_bad_order():
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="too large"):
        smooth_whittaker(y, lambda_=1.0, order=3)


def test_smooth_whittaker_rejects_nonpositive_lambda():
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="must be positive"):
        smooth_whittaker(y, lambda_=0.0, order=1)


def test_make_dT_d_annihilates_constant_vector():
    DTD = make_dT_d(5, 1)
    assert DTD.shape == (5, 5)
    assert np.allclose(DTD @ np.ones(5), np.zeros(5))


def test_bandwidth_to_lambda():
    # (2 * 2!)^2 / bandwidth^4 = 16 / 16 = 1.0 for bandwidth=2, order=2
    assert np.isclose(bandwidth_to_lambda(bandwidth=2.0, order=2), 1.0)


def test_noise_gain_to_lambda():
    # (2 * 2!)^2 / gain = 16 / 16 = 1.0 for gain=16, order=2
    assert np.isclose(noise_gain_to_lambda(gain=16.0, order=2), 1.0)


def test_smooth_like_savitzky_golay_linear_signal_is_invariant():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    smoothed = smooth_like_savitzky_golay(y, bandwidth=2.0, order=2)
    assert np.allclose(smoothed, y)


def test_smooth_with_noise_gain_constant_signal_is_invariant():
    y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    smoothed = smooth_with_noise_gain(y, gain=1e-2, order=1)
    assert np.allclose(smoothed, y)


def test_batch_smooth_whittaker_linear_rows_are_invariant():
    y_2d = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 8.0, 6.0, 4.0, 2.0]])
    smoothed = batch_smooth_whittaker(y_2d, lambda_=100.0, order=2)
    assert np.allclose(smoothed, y_2d)
