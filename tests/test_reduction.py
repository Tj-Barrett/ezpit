import numpy as np
import pytest

from ezpit.core.reduction import (
    apply_lorch_function,
    cal_expGr_fft,
    cal_expGr_fft_from_Fq,
    cal_expSq,
    cal_fq,
    lorch_function_sinc,
)


@pytest.mark.parametrize(
    ("Q", "Q_max", "expected"),
    [
        (np.array([1.0, 2.0, 3.0]), 0.0, np.array([1.0, 1.0, 1.0])),  # Q_max=0 -> ones
        (0.0, 10.0, 1.0),  # sinc(0) = 1
        (5.0, 5.0, 0.0),  # sinc(1) = 0
    ],
)
def test_lorch_function_sinc(Q, Q_max, expected):
    assert np.allclose(lorch_function_sinc(Q, Q_max), expected)


def test_apply_lorch_function():
    Q = np.array([0.0, 1.0, 2.0, 4.0])
    FQ = np.array([1.0, 2.0, 3.0, 4.0])
    expected = FQ * np.sinc(Q / np.max(Q))
    assert np.allclose(apply_lorch_function(Q, FQ), expected)


def test_cal_fq():
    Sq = np.array([2.0, 3.0])
    Fq, q_range = cal_fq(qmin=0, qmax=2, Sq=Sq)
    assert np.allclose(q_range, [0.0, 1.0])
    assert np.allclose(Fq, [0.0, 2.0])


def test_cal_expGr_fft_zero_structure_factor():
    q = np.linspace(0.1, 5.0, 50)
    Sq = np.ones_like(q)
    r_list, gr = cal_expGr_fft(q, Sq, rmin=0, rmax=2, rstep=0.5)
    assert np.allclose(gr, np.zeros_like(r_list), atol=1e-8)


def test_cal_expGr_fft_short_q_returns_none():
    assert cal_expGr_fft(np.array([1.0]), np.array([1.0])) == (None, None)
    assert cal_expGr_fft(np.array([1.0]), np.array([1.0]), return_padding=True) == (None, None, None)


def test_cal_expGr_fft_return_padding_keys():
    q = np.linspace(0.1, 5.0, 50)
    Sq = np.ones_like(q)
    r_list, gr, padding_info = cal_expGr_fft(q, Sq, rmin=0, rmax=2, rstep=0.5, return_padding=True)
    assert set(padding_info) == {
        "q_low", "F_low", "q_exp", "F_exp", "q_pad", "F_pad",
        "q_full_padded", "F_full_padded", "qstep", "total_N",
    }


def test_cal_expGr_fft_pad_mode_variants():
    # Sq != 1 everywhere so Fq = (Sq-1)*q is nonzero at the last q point,
    # making the three pad_mode behaviors distinguishable.
    q = np.linspace(0.1, 5.0, 50)
    Sq = 2.0 * np.ones_like(q)
    common = {"rmin": 0, "rmax": 2, "rstep": 0.5, "return_padding": True}

    _, _, info_zero = cal_expGr_fft(q, Sq, pad_mode="zero", **common)
    _, _, info_decay = cal_expGr_fft(q, Sq, pad_mode="decay", **common)
    _, _, info_constant = cal_expGr_fft(q, Sq, pad_mode="constant", **common)

    last_val = info_zero["F_exp"][-1]
    assert info_zero["F_pad"].size > 0
    assert np.allclose(info_zero["F_pad"], 0.0)
    assert np.allclose(info_decay["F_pad"], np.linspace(last_val, 0, info_decay["F_pad"].size))
    assert np.allclose(info_constant["F_pad"], last_val)


def test_cal_expGr_fft_pad_len_non_positive_skips_padding():
    # Large rstep -> few IFFT points needed -> no high-Q padding required.
    q = np.linspace(0.1, 5.0, 50)
    Sq = 2.0 * np.ones_like(q)
    _, gr, info = cal_expGr_fft(q, Sq, rmin=0, rmax=2, rstep=5.0, return_padding=True)
    assert info["F_pad"].size == 0
    assert np.all(np.isfinite(gr))


def test_cal_expGr_fft_low_q_mode_linear():
    q = np.linspace(0.1, 5.0, 50)
    Sq = 2.0 * np.ones_like(q)
    r_list, gr = cal_expGr_fft(q, Sq, rmin=0, rmax=2, rstep=0.5, low_q_mode="linear")
    assert len(gr) == len(r_list)
    assert np.all(np.isfinite(gr))


def test_cal_expGr_fft_nyquist_rescale():
    # rmax > pi/qstep forces the finer-grid Nyquist rescue path.
    q = np.linspace(0.1, 5.0, 50)
    Sq = 2.0 * np.ones_like(q)
    r_list, gr, info = cal_expGr_fft(q, Sq, rmin=0, rmax=50, rstep=0.5, return_padding=True)
    assert info["qstep"] < q[1] - q[0]
    assert len(gr) == len(r_list)
    assert np.all(np.isfinite(gr))


def test_cal_expGr_fft_from_Fq_zero():
    q = np.linspace(0.1, 5.0, 50)
    Fq = np.zeros_like(q)
    r_list, gr = cal_expGr_fft_from_Fq(q, Fq, rmin=0, rmax=2, rstep=0.5)
    assert np.allclose(gr, np.zeros_like(r_list), atol=1e-8)


def test_cal_expGr_fft_from_Fq_pad_mode_variants_differ():
    q = np.linspace(0.1, 5.0, 50)
    Fq = q.copy()  # nonzero at the last q point, so pad_mode changes the result
    common = {"rmin": 0, "rmax": 2, "rstep": 0.5}

    _, gr_zero = cal_expGr_fft_from_Fq(q, Fq, pad_mode="zero", **common)
    _, gr_decay = cal_expGr_fft_from_Fq(q, Fq, pad_mode="decay", **common)
    _, gr_constant = cal_expGr_fft_from_Fq(q, Fq, pad_mode="constant", **common)

    assert not np.allclose(gr_zero, gr_decay)
    assert not np.allclose(gr_zero, gr_constant)
    assert not np.allclose(gr_decay, gr_constant)


def test_cal_expGr_fft_from_Fq_pad_len_non_positive_runs():
    q = np.linspace(0.1, 5.0, 50)
    Fq = q.copy()
    r_list, gr = cal_expGr_fft_from_Fq(q, Fq, rmin=0, rmax=2, rstep=5.0)
    assert len(gr) == len(r_list)
    assert np.all(np.isfinite(gr))


def test_cal_expGr_fft_from_Fq_low_q_mode_linear():
    q = np.linspace(0.1, 5.0, 50)
    Fq = q.copy()
    r_list, gr = cal_expGr_fft_from_Fq(q, Fq, rmin=0, rmax=2, rstep=0.5, low_q_mode="linear")
    assert len(gr) == len(r_list)
    assert np.all(np.isfinite(gr))


def test_cal_expGr_fft_from_Fq_nyquist_rescale():
    q = np.linspace(0.1, 5.0, 50)
    Fq = q.copy()
    r_list, gr = cal_expGr_fft_from_Fq(q, Fq, rmin=0, rmax=50, rstep=0.5)
    assert len(gr) == len(r_list)
    assert np.all(np.isfinite(gr))


def test_cal_expSq_shape_and_sanity():
    # cal_expSq's PDFgetX3 least-squares normalisation and polynomial
    # correction make an exact closed-form expectation impractical to hand
    # derive, so this is a shape/sanity check rather than an exact-value test.
    scattering_factors = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.0]])
    exp_q = np.linspace(0.0, 10.0, 200)
    exp_Iq = 9.0 * np.ones_like(exp_q)

    result = cal_expSq(
        atom_indices=[0],
        scattering_factors=scattering_factors,
        expqiq=(exp_q, exp_Iq),
        bkgqiq=None,
        qmin=0,
        qmax=10,
        qstep=0.1,
        return_Iq=False,
    )
    assert len(result) == 13
    q_range = result[0]
    assert q_range[0] == 0
    assert q_range[-1] < 10
    assert len(q_range) == 100  # arange(0, 10, 0.1)
    for arr in result[:-1]:
        assert len(arr) == len(q_range)
        assert np.all(np.isfinite(arr))
    assert np.isfinite(result[-1])  # normalization_scale is a scalar


def test_cal_expSq_return_Iq_true_matches_false():
    # Unlike cal_Sq (which genuinely returns a shorter tuple when return_Iq is
    # False), cal_expSq's two branches build and return the identical 13-value
    # tuple in the identical order regardless of return_Iq -- the parameter is
    # currently a no-op here. This test documents that actual behavior rather
    # than an assumed one; if that's unintended, the fix belongs in cal_expSq
    # itself, not in this test.
    scattering_factors = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.0]])
    exp_q = np.linspace(0.0, 10.0, 200)
    exp_Iq = 9.0 * np.ones_like(exp_q)
    kwargs = {
        "atom_indices": [0],
        "scattering_factors": scattering_factors,
        "expqiq": (exp_q, exp_Iq),
        "bkgqiq": None,
        "qmin": 0,
        "qmax": 10,
        "qstep": 0.1,
    }

    result_true = cal_expSq(**kwargs, return_Iq=True)
    result_false = cal_expSq(**kwargs, return_Iq=False)

    assert len(result_true) == len(result_false) == 13
    for a, b in zip(result_true, result_false, strict=True):
        assert np.array_equal(a, b)
