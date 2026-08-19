import numpy as np
import pytest

import ezpit.core.processing as proc
from ezpit.core.processing import (
    cal_Gr_fft,
    cal_Gr_integral,
    cal_Iq,
    cal_Sq,
    compton_cal_exp,
    create_atom_distance_matrix,
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
    fi = proc._cal_fi(scat_values, q)
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
