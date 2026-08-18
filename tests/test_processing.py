import numpy as np
import pytest

import ezpit.processing as proc
from ezpit.processing import (
    apply_lorch_function,
    bandwidth_to_lambda,
    batch_smooth_whittaker,
    cal_expGr_fft,
    cal_expGr_fft_from_Fq,
    cal_expSq,
    cal_fq,
    cal_Gr_fft,
    cal_Gr_integral,
    cal_Iq,
    cal_Sq,
    compton_cal_exp,
    create_atom_distance_matrix,
    detect_header_lines,
    load_qiq_file,
    lorch_function_sinc,
    make_dT_d,
    noise_gain_to_lambda,
    smooth_like_savitzky_golay,
    smooth_whittaker,
    smooth_with_noise_gain,
)


@pytest.mark.parametrize(
    ("_2D_atom_positions", "expected"),
    [([[0, 1], [1, 0]], np.array([[0, 1.41421356], [1.41421356, 0]]))],
)
def test_create_atom_distance_2D_matrix(_2D_atom_positions, expected):
    dist = create_atom_distance_matrix(_2D_atom_positions)
    assert dist.shape == (2, 2)
    assert np.allclose(dist, expected)


@pytest.mark.parametrize(
    ("_3D_atom_positions", "expected"),
    [([[0, 0, 1], [0, 1, 0], [1, 0, 0]],
    np.array([[0, 1.41421356, 1.41421356],
              [1.41421356, 0, 1.41421356],
              [1.41421356, 1.41421356, 0]]))],
)
def test_create_atom_distance_3D_matrix(_3D_atom_positions, expected):
    dist = create_atom_distance_matrix(_3D_atom_positions)
    assert dist.shape == (3, 3)
    assert np.allclose(dist, expected)


@pytest.mark.parametrize(
    ("scat_values", "q", "expected"),
    [
        # all a_i = 0, c = 5 -> constant regardless of q
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.0], 2.0, 5.0),
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.0],
            np.array([0.0, 1.0, 2.0]), np.array([5.0, 5.0, 5.0])),
        # single term isolated: a1=2, b1=1, rest 0, c=0, q=4pi -> (q/4pi)^2 = 1
        ([2.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0], 4 * np.pi, 2.0 * np.exp(-1.0)),
    ],
)
def test_cal_fi(scat_values, q, expected):
    fi = proc.__cal_fi(scat_values, q)
    assert np.allclose(fi, expected)


@pytest.mark.parametrize(
    ("csf", "q", "expected"),
    [
        # all a_i = 0, fic = 7 -> constant regardless of q
        ([0, 0, 0, 0, 0, 7.0, 0, 0, 0, 0, 0], 3.0, 7.0),
        ([0, 0, 0, 0, 0, 7.0, 0, 0, 0, 0, 0],
            np.array([0.0, 1.0, 2.0]), np.array([7.0, 7.0, 7.0])),
        # single term isolated: csf[0]=2, csf[6]=1, rest 0, q=4pi -> (q/4pi)^2 = 1
        ([2.0, 0, 0, 0, 0, 0.0, 1.0, 0, 0, 0, 0], 4 * np.pi, 2.0 * np.exp(-1.0)),
    ],
)
def test_cal_compton_fi(csf, q, expected):
    fi = proc.__cal_compton_fi(csf, q)
    assert np.allclose(fi, expected)


def test_compton_cal_exp_single_element():
    # wavelength=0 makes the Breit-Dirac recoil factor exactly 1 for any alpha,
    # and an all-zero-amplitude compton_scat_parms row (fic=4.0) makes the form
    # factor constant across q, so the result reduces to a closed form:
    #   compton_scat = Z - fic^2 / Z  (constant across the whole q range)
    Z = 2.0
    fic = 4.0
    compton_scat_parms = np.array(
        [
            [0.0] * 11,
            [0, 0, 0, 0, 0, fic, 0, 0, 0, 0, 0],
        ]
    )
    q_range, list_compton_scat = compton_cal_exp(
        atom_indices=np.array([0]),
        compton_scat_parms=compton_scat_parms,
        compton_scattering_factors=np.array([0.0]),  # only its length (1) is used
        atomic_number=np.array([Z]),
        qmin=0,
        qmax=2,
        qstep=1,
        wavelength=0.0,
        alpha=2,
        weights=np.array([1.0]),
    )
    expected = Z - fic**2 / Z
    assert np.allclose(q_range, [0.0, 1.0])
    assert np.allclose(list_compton_scat, [expected, expected])


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


def test_cal_Iq_single_atom():
    # single atom -> self distance 0, so I(q) = f(q)^2 for every q
    scattering_factors = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.0]])
    q_range, list_Iq = cal_Iq(
        atom_indices=[0],
        scattering_factors=scattering_factors,
        atom_distance_matrix=np.array([[0.0]]),
        qmin=1.0,
        qmax=3.0,
        qstep=1.0,
    )
    assert np.allclose(q_range, [1.0, 2.0])
    assert np.allclose(list_Iq, [9.0, 9.0])


def test_cal_Sq_single_atom():
    # single atom -> S(q) is identically 1 regardless of the form factor value
    scattering_factors = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.0]])
    atom_distance_matrix = np.array([[0.0]])

    q_range, list_Sq = cal_Sq(
        atom_indices=[0],
        scattering_factors=scattering_factors,
        atom_distance_matrix=atom_distance_matrix,
        qmin=1.0,
        qmax=3.0,
        qstep=1.0,
        return_Iq=False,
    )
    assert np.allclose(q_range, [1.0, 2.0])
    assert np.allclose(list_Sq, [1.0, 1.0])

    q_range, list_Iq, list_Sq, list_Fq, mean_sq_fi, sq_mean_fi = cal_Sq(
        atom_indices=[0],
        scattering_factors=scattering_factors,
        atom_distance_matrix=atom_distance_matrix,
        qmin=1.0,
        qmax=3.0,
        qstep=1.0,
        return_Iq=True,
    )
    assert np.allclose(list_Iq, [9.0, 9.0])
    assert np.allclose(list_Sq, [1.0, 1.0])
    assert np.allclose(list_Fq, [0.0, 0.0])
    assert np.allclose(mean_sq_fi, [9.0, 9.0])
    assert np.allclose(sq_mean_fi, [9.0, 9.0])


def test_cal_Gr_integral():
    # two-point q axis, chosen so F(q) = (Sq-1)*q is nonzero only at q=1,
    # collapsing the Riemann-sum loop to a single closed-form sine term.
    q = np.array([0.0, 1.0])
    Sq = np.array([1.0, 2.0])
    r_list, list_Gr = cal_Gr_integral(q, Sq, rmin=0.0, rmax=2.0, rstep=1.0)
    expected_r = np.array([0.0, 1.0, 2.0])
    expected_Gr = np.sin(expected_r) * 2.0 / np.pi
    assert np.allclose(r_list, expected_r)
    assert np.allclose(list_Gr, expected_Gr)


def test_cal_Gr_integral_qdamp():
    # same closed form as test_cal_Gr_integral, with the qdamp Gaussian
    # envelope applied on top: G_damped(r) = exp(-0.5*(r*qdamp)^2) * G(r).
    q = np.array([0.0, 1.0])
    Sq = np.array([1.0, 2.0])
    qdamp = 0.5
    r_list, list_Gr = cal_Gr_integral(q, Sq, rmin=0.0, rmax=2.0, rstep=1.0, qdamp=qdamp)
    expected_r = np.array([0.0, 1.0, 2.0])
    expected_Gr = (np.sin(expected_r) * 2.0 / np.pi) * np.exp(-0.5 * (expected_r * qdamp) ** 2)
    assert np.allclose(r_list, expected_r)
    assert np.allclose(list_Gr, expected_Gr)


def test_cal_Gr_integral_integer_rbounds_do_not_truncate():
    # regression: integer rmin/rmax/rstep must not floor G(r) to 0 via an
    # int-dtype output array.
    q = np.array([0.0, 1.0])
    Sq = np.array([1.0, 2.0])
    r_list, list_Gr = cal_Gr_integral(q, Sq, rmin=0, rmax=2, rstep=1)
    expected_r = np.array([0.0, 1.0, 2.0])
    expected_Gr = np.sin(expected_r) * 2.0 / np.pi
    assert list_Gr.dtype == np.float64
    assert np.allclose(r_list, expected_r)
    assert np.allclose(list_Gr, expected_Gr)


def test_cal_Gr_fft_zero_structure_factor():
    # S(q) = 1 everywhere -> F(q) = 0 everywhere -> G(r) = 0 everywhere
    q = np.linspace(0.1, 5.0, 50)
    Sq = np.ones_like(q)
    r_list, gr = cal_Gr_fft(q, Sq, rmin=0, rmax=2, rstep=0.5)
    assert np.allclose(gr, np.zeros_like(r_list), atol=1e-8)


def test_cal_Gr_fft_qdamp_matches_undamped_envelope():
    # cal_Gr_fft applies qdamp as a post-multiplication by exp(-0.5*(r*qdamp)^2)
    # on the internal fine r-grid (spacing = rstep, starting at 0) before
    # interpolating onto r_list. With rmin=0 and the same rstep, r_list's
    # points coincide exactly with that fine grid, so np.interp introduces no
    # error and the damped/undamped outputs must satisfy this relationship
    # exactly -- letting us verify the qdamp branch without hand-deriving the
    # underlying IFFT values.
    q = np.linspace(0.1, 5.0, 50)
    Sq = 1.0 + 0.1 * np.sin(q)  # nontrivial, nonzero G(r)
    rmin, rmax, rstep = 0.0, 2.0, 0.5
    qdamp = 0.5

    r_list, gr_undamped = cal_Gr_fft(q, Sq, rmin=rmin, rmax=rmax, rstep=rstep, qdamp=0.0)
    _, gr_damped = cal_Gr_fft(q, Sq, rmin=rmin, rmax=rmax, rstep=rstep, qdamp=qdamp)

    assert not np.allclose(gr_undamped, 0.0)  # sanity: the baseline isn't trivially zero
    expected_damped = gr_undamped * np.exp(-0.5 * (r_list * qdamp) ** 2)
    assert np.allclose(gr_damped, expected_damped)


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


def test_cal_expGr_fft_from_Fq_zero():
    q = np.linspace(0.1, 5.0, 50)
    Fq = np.zeros_like(q)
    r_list, gr = cal_expGr_fft_from_Fq(q, Fq, rmin=0, rmax=2, rstep=0.5)
    assert np.allclose(gr, np.zeros_like(r_list), atol=1e-8)


def test_detect_header_lines_and_load_qiq_file(tmp_path):
    data_file = tmp_path / "sample.iq"
    data_file.write_text("# comment header\nsome text header\n1.0 2.0\n2.0 3.0\n3.0 4.0\n")

    assert detect_header_lines(str(data_file)) == 2

    data = load_qiq_file(str(data_file))
    assert np.allclose(data, [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])


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
