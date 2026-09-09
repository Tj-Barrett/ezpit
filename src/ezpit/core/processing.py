import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from scipy import interpolate
from scipy.spatial.distance import cdist


def __cal_compton_fi(
    compton_scattering_factors: list[float] | NDArray[np.float64], q: float | NDArray[np.float64]
) -> float | NDArray[np.float64]:
    """
    [EN] Calculate Compton scattering form factor (Inelastic scattering).

    [KR] 콤프턴 산란 형상 인자를 계산합니다 (비탄성 산란).
    """
    k_sq = (0.25 * q / np.pi) ** 2
    fi1 = compton_scattering_factors[0] * np.exp(-compton_scattering_factors[6] * k_sq)
    fi2 = compton_scattering_factors[1] * np.exp(-compton_scattering_factors[7] * k_sq)
    fi3 = compton_scattering_factors[2] * np.exp(-compton_scattering_factors[8] * k_sq)
    fi4 = compton_scattering_factors[3] * np.exp(-compton_scattering_factors[9] * k_sq)
    fi5 = compton_scattering_factors[4] * np.exp(-compton_scattering_factors[10] * k_sq)
    fic = compton_scattering_factors[5]
    fi = fi1 + fi2 + fi3 + fi4 + fi5 + fic
    return fi


def _cal_fi(scat_values: list[float], q: float | np.ndarray[tuple[int], np.dtype[np.float32]]
) -> float | np.ndarray[tuple[int], np.dtype[np.float32]]:
    """
    [EN] Calculate atomic form factor f(q) using 5-Gaussian approximation.

    [KR] 5-가우시안 근사를 사용하여 원자 형상 인자 f(q)를 계산합니다.
    Formula: f(q) = sum(ai * exp(-bi * (q/4pi)^2)) + c.

    Args:
        scat_values (list): [EN] Coefficients [a1, b1, ..., c] / [KR] 계수 리스트
        q (float/array): [EN] Momentum transfer / [KR] 운동량 변화량 Q

    Returns
    -------
        fi (float/array): [EN] Calculated form factor / [KR] 계산된 형상 인자
    """
    k_sq = (0.25 * q / np.pi) ** 2
    fi1 = scat_values[0] * np.exp(-scat_values[1] * k_sq)
    fi2 = scat_values[2] * np.exp(-scat_values[3] * k_sq)
    fi3 = scat_values[4] * np.exp(-scat_values[5] * k_sq)
    fi4 = scat_values[6] * np.exp(-scat_values[7] * k_sq)
    fi5 = scat_values[8] * np.exp(-scat_values[9] * k_sq)
    fic = scat_values[10]
    fi = fi1 + fi2 + fi3 + fi4 + fi5 + fic
    return fi


def cal_Gr_integral(
    q: NDArray[np.float64],
    Sq: NDArray[np.float64],
    rmin: float = 0,
    rmax: float = 100,
    rstep: float = 0.02,
    qdamp: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    [EN] Calculate the Pair Distribution Function G(r) from Structure Factor S(q) using direct sine integral transform.

         This method performs a discrete integration (Riemann sum) for each r value.
         It is generally slower than FFT but free from grid artifacts and padding issues.
         Formula: G(r) = (2/pi) * integral [ Q * (S(Q)-1) * sin(Q*r) ] dQ.

    [KR] 구조 인자 S(q)로부터 직접 사인 적분 변환을 사용하여 쌍 분포 함수 G(r)을 계산합니다.
         각 r 값에 대해 이산 적분(리만 합)을 수행합니다.
         FFT 방식보다 속도는 느리지만 그리드 아티팩트나 패딩 문제 없이 정확한 값을 얻을 수 있습니다.
         공식: G(r) = (2/pi) * integral [ Q * (S(Q)-1) * sin(Q*r) ] dQ

    Args:
        q (numpy.ndarray):
            [EN] The momentum transfer axis Q (inverse Angstrom). 1D array of floats.
            [KR] 운동량 전달 축 Q (단위: 1/A). 실수형 1차원 배열.

        Sq (numpy.ndarray):
            [EN] The Structure Factor S(q) corresponding to the q axis. 1D array of floats.
            [KR] Q 축에 해당하는 구조 인자 S(q). 실수형 1차원 배열.

        rmin (float, optional):
            [EN] Minimum value for the real-space distance r (Angstrom). Default is 0.
            [KR] 실공간 거리 r의 최소값 (단위: A). 기본값은 0.

        rmax (float, optional):
            [EN] Maximum value for the real-space distance r (Angstrom). Default is 100.
            [KR] 실공간 거리 r의 최대값 (단위: A). 기본값은 100.

        rstep (float, optional):
            [EN] Step size for the r axis. Default is 0.02.
            [KR] r 축의 간격. 기본값은 0.02.

        qdamp (float, optional):
            [EN] Gaussian damping factor to simulate limited instrumental resolution.
                 Applied as envelope: exp(-0.5 * (r * qdamp)^2). Default is 0.0 (no damping).
            [KR] 제한된 기기 해상도를 시뮬레이션하기 위한 가우시안 감쇠 인자.
                 적용 식: exp(-0.5 * (r * qdamp)^2). 기본값은 0.0 (감쇠 없음).

    Returns
    -------
        tuple (numpy.ndarray, numpy.ndarray):
            [EN] Returns a tuple containing:
                 - list_r: The generated r-axis (Angstrom).
                 - list_Gr: The calculated G(r).
            [KR] 다음을 포함하는 튜플을 반환합니다:
                 - list_r: 생성된 r 축 (단위: A).
                 - list_Gr: 계산된 G(r) 값.
    """
    # dtype=float guards against integer rmin/rmax/rstep silently truncating
    # list_Gr (via zeros_like below) to an int array, which would floor every
    # computed G(r) value to 0.
    list_r = np.arange(rmin, rmax + rstep, rstep, dtype=np.float64)
    Fq = (Sq - 1) * q
    qstep = q[1] - q[0]
    list_Gr = np.zeros_like(list_r)
    list_qFq = Fq * qstep
    for i, r in enumerate(list_r):
        list_Gr[i] = np.sum(list_qFq * np.sin(q * r)) * 2.0 / np.pi
    if qdamp != 0.0:
        list_Gr = np.exp(-0.5 * (list_r * qdamp) ** 2) * list_Gr
    return list_r, list_Gr


def cal_Gr_fft(
    q: NDArray[np.float64],
    Sq: NDArray[np.float64],
    rmin: float = 0,
    rmax: float = 100,
    rstep: float = 0.02,
    qdamp: float = 0.0,
    extrapolate_type: str = "linear",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    [EN] Calculate the Pair Distribution Function G(r) via Inverse Fast Fourier Transform.

         This function handles low-Q extrapolation and zero-padding to achieve
         the desired real-space resolution (rstep).
         Formula: G(r) = (2/pi) * FFT_imag[ Q * (S(Q)-1) ].

    [KR] 고속 푸리에 변환(IFFT)을 사용하여 구조 인자 S(q)로부터 쌍 분포 함수 G(r)을 계산합니다.
         Low-Q 영역의 보간(extrapolation)과 제로 패딩(zero-padding)을 처리하여 원하는 실공간 해상도(rstep)를 얻습니다.
         공식: G(r) = (2/pi) * FFT_imag[ Q * (S(Q)-1) ]

    Args:
        q (numpy.ndarray):
            [EN] The momentum transfer axis Q (inverse Angstrom). 1D array of floats.
            [KR] 운동량 전달 축 Q (단위: 1/A). 실수형 1차원 배열.

        Sq (numpy.ndarray):
            [EN] The Structure Factor S(q) corresponding to the q axis.
            [KR] Q 축에 해당하는 구조 인자 S(q).

        rmin (float, optional):
            [EN] Minimum value for the real-space distance r (Angstrom). Default is 0.
            [KR] 실공간 거리 r의 최소값 (단위: A). 기본값은 0.

        rmax (float, optional):
            [EN] Maximum value for the real-space distance r (Angstrom). Default is 100.
            [KR] 실공간 거리 r의 최대값 (단위: A). 기본값은 100.

        rstep (float, optional):
            [EN] Step size for the r axis. Smaller steps require larger zero-padding in Q-space. Default is 0.02.
            [KR] r 축의 간격. 간격이 작을수록 Q 공간에서 더 많은 제로 패딩이 필요합니다. 기본값은 0.02.

        qdamp (float, optional):
            [EN] Gaussian damping factor for instrument resolution correction.
                 Applied in real space as: G(r) * exp(-0.5 * (r * qdamp)^2). Default is 0.0.
            [KR] 기기 해상도 보정을 위한 가우시안 감쇠 인자.
                 실공간에서 G(r) * exp(-0.5 * (r * qdamp)^2) 형태로 적용됩니다. 기본값은 0.0.

        extrapolate_type (str, optional):
            [EN] Interpolation method for filling the low-Q region (0 to Qmin).
                 Options: 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'. Default is 'linear'.
            [KR] Low-Q 영역 (0부터 Qmin까지)을 채우기 위한 보간 방법.
                 옵션: 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'. 기본값은 'linear'.

    Returns
    -------
        tuple (numpy.ndarray, numpy.ndarray):
            [EN] Returns a tuple containing:
                 - r_list: The generated r-axis (Angstrom).
                 - gr: The calculated G(r) interpolated onto r_list.
            [KR] 다음을 포함하는 튜플을 반환합니다:
                 - r_list: 생성된 r 축 (단위: A).
                 - gr: r_list에 맞춰 보간된 G(r) 값.
    """
    qstep = q[1] - q[0]
    num_point = int(np.ceil(q[0] / qstep))
    Fq = (Sq - 1) * q
    pad_Fq = np.zeros(num_point)
    if q[0] > 0.0:
        f_inter = interpolate.interp1d(q, Fq, fill_value="extrapolate", kind=extrapolate_type)
        pad_Fq[:num_point] = f_inter(np.arange(num_point) * qstep)
    Fq_vals = np.append(pad_Fq, Fq)
    r_list = np.arange(rmin, rmax + rstep, rstep)

    # ------------------------------------------------------------------
    # Nyquist check (same as cal_expGr_fft):
    #   The IFFT produces total_point values; only the first half
    #   (indices 0 .. total_point//2) corresponds to physically valid
    #   positive-r values, giving r_nyquist = pi / qstep.
    #   If rmax exceeds this limit, interpolate F(q) onto a finer q grid.
    # ------------------------------------------------------------------
    r_nyquist = np.pi / qstep
    if rmax > r_nyquist:
        qstep_needed = np.pi / rmax
        q_orig = np.arange(len(Fq_vals)) * qstep
        q_fine = np.arange(q_orig[0], q_orig[-1] + qstep_needed, qstep_needed)
        Fq_vals = np.interp(q_fine, q_orig, Fq_vals)
        qstep = qstep_needed

    num_point = len(Fq_vals)
    total_point = int(2 * np.pi / (rstep * qstep))
    Fq_pad = np.pad(Fq_vals, (0, total_point - num_point), mode="constant")
    norm = total_point * qstep * 2 / np.pi
    gr_full = norm * np.imag(np.fft.ifft(Fq_pad))

    # Use only the alias-free first half of the IFFT output
    half = total_point // 2
    rfine = np.arange(half) * rstep
    gr_fine = gr_full[:half]
    if qdamp != 0.0:
        gr_fine = gr_fine * np.exp(-0.5 * (rfine * qdamp) ** 2)
    gr = np.interp(r_list, rfine, gr_fine)
    return r_list, gr


def cal_Iq(
    atom_indices: list[int] | NDArray[np.int64],
    scattering_factors: NDArray[np.float64],
    atom_distance_matrix: NDArray[np.float64],
    qmin: float = 0.5,
    qmax: float = 20,
    qstep: float = 0.05,
) -> tuple[NDArray[np.float64], list[float]]:
    """
    [EN] Calculate the total Scattering Intensity I(q) based on the Debye scattering equation.

         This function computes the sum of coherent scattering from all atom pairs.
         Formula: I(q) = sum_i sum_j [ f_i(q) * f_j(q) * sin(q*r_ij)/(q*r_ij) ].

    [KR] Debye 산란 공식을 기반으로 총 산란 강도 I(q)를 계산합니다.
         모든 원자 쌍 사이의 간섭성 산란 합을 구합니다.
         공식: I(q) = sum_i sum_j [ f_i(q) * f_j(q) * sin(q*r_ij)/(q*r_ij) ]

    Args:
        atom_indices (list or numpy.ndarray):
            [EN] A 1D array of integers (length N) mapping each atom to its element type index.
            [KR] 각 원자가 어떤 원소 종류인지 매핑하는 정수형 1차원 배열 (길이 N).

        scattering_factors (numpy.ndarray):
            [EN] A 2D array of atomic scattering factor coefficients (shape: [num_elements, 11]).
                 Used to calculate f(q).
            [KR] 원자 산란 인자 계수를 담은 2차원 배열. f(q) 계산에 사용됩니다.

        atom_distance_matrix (numpy.ndarray):
            [EN] A symmetric matrix (N x N) representing the Euclidean distance between atom i and atom j.
            [KR] i번째 원자와 j번째 원자 사이의 유클리드 거리를 나타내는 대칭 행렬 (N x N).

        qmin (float, optional):
            [EN] Minimum Q value (inverse Angstrom). Default is 0.5.
            [KR] Q 값의 최소값 (1/A). 기본값은 0.5.

        qmax (float, optional):
            [EN] Maximum Q value (inverse Angstrom). Default is 20.
            [KR] Q 값의 최대값 (1/A). 기본값은 20.

        qstep (float, optional):
            [EN] Step size for Q. Default is 0.05.
            [KR] Q 계산 간격. 기본값은 0.05.

    Returns
    -------
        tuple (numpy.ndarray, list):
            [EN] Returns a tuple containing:
                 - q_range: Array of Q values.
                 - list_Iq: List of calculated Intensity values I(q).
            [KR] 다음을 포함하는 튜플을 반환합니다:
                 - q_range: Q 값 배열.
                 - list_Iq: 계산된 산란 강도 I(q) 리스트.
    """
    # Theoretical calculation loop (kept as loop to prevent memory overflow for large N)
    num_atom = len(atom_indices)
    num_fact = len(scattering_factors)
    diag_idx = np.diag_indices(num_atom)
    distance_matrix_non_zero = np.copy(atom_distance_matrix)
    distance_matrix_non_zero[diag_idx] = 1.0
    list_Iq = []
    q_range = np.arange(qmin, qmax, qstep)
    for q in q_range:
        fi_mat = np.zeros((num_atom, num_atom))
        list_fi = []
        for k in range(num_fact):
            list_fi.append(_cal_fi(scattering_factors[k], q))
        list_fi = np.asarray(list_fi)
        for i, idx in enumerate(atom_indices):
            fi = list_fi[idx]
            fi_mat[i, :] = fi
        if q == 0:
            sin_mat = np.ones(np.shape(atom_distance_matrix))
        else:
            sin_mat = np.sin(q * atom_distance_matrix) / (q * distance_matrix_non_zero)
        sin_mat[diag_idx] = 1.0
        Iq = fi_mat * np.transpose(fi_mat) * sin_mat
        list_Iq.append(np.sum(Iq))
    return q_range, list_Iq


@njit(parallel=True, cache=True)
def _debye_sum(
    fi_by_atom: NDArray[np.float64], atom_distance_matrix: NDArray[np.float64], q_range: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    [EN] Numba-jitted Debye double sum I(q) = sum_i sum_j f_i(q) f_j(q) sin(q r_ij)/(q r_ij).

         Parallelized over q (prange), and never materializes an N x N matrix, so it stays
         cheap on memory for large atom counts. `fi_by_atom` is (num_atom, len(q_range)).
    [KR] Debye 이중합 계산을 위한 numba JIT 함수입니다. q에 대해 병렬화되어 있으며
         N x N 행렬을 만들지 않아 원자 수가 많아도 메모리 사용량이 적습니다.
    """
    num_atom = atom_distance_matrix.shape[0]
    n_q = q_range.shape[0]
    list_Iq = np.empty(n_q, dtype=np.float64)
    for qi in prange(n_q):
        q = q_range[qi]
        total = 0.0
        for i in range(num_atom):
            fi_i = fi_by_atom[i, qi]
            for j in range(num_atom):
                if i == j:
                    total += fi_i * fi_i
                elif q == 0.0:
                    total += fi_i * fi_by_atom[j, qi]
                else:
                    qr = q * atom_distance_matrix[i, j]
                    total += fi_i * fi_by_atom[j, qi] * (np.sin(qr) / qr)
        list_Iq[qi] = total
    return list_Iq


def cal_Sq(
    atom_indices: list[int] | NDArray[np.int64],
    scattering_factors: NDArray[np.float64],
    atom_distance_matrix: NDArray[np.float64],
    qmin: float = 0.5,
    qmax: float = 20,
    qstep: float = 0.05,
    return_Iq: bool = False,
) -> (
    tuple[NDArray[np.float64], NDArray[np.float64]]
    | tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]
):
    """
    [EN] Calculate theoretical Structure Factor S(q) using the Debye scattering equation.

         This function computes I(q), S(q), and F(q) from atomic coordinates and scattering factors.
    [KR] Debye 산란 공식을 사용하여 이론적인 구조 인자 S(q)를 계산합니다.
         원자 좌표와 산란 인자를 바탕으로 I(q), S(q), F(q)를 계산합니다.

    Args:
        atom_indices (list or numpy.ndarray):
            [EN] A 1D array of integers mapping each atom to its element type index.
                 (e.g., [0, 0, 1] means Atom1=Type0, Atom2=Type0, Atom3=Type1)
            [KR] 각 원자가 어떤 원소 종류인지 나타내는 정수형 인덱스 배열입니다.
                 (예: [0, 0, 1]은 원자1=타입0, 원자2=타입0, 원자3=타입1을 의미)

        scattering_factors (numpy.ndarray):
            [EN] A 2D array of scattering coefficients (shape: [num_unique_elements, 11]).
                 Contains Gaussian parameters (a, b, c) for f(q) calculation.
            [KR] 산란 계수들을 포함하는 2차원 배열입니다 (크기: [고유원소수, 11]).
                 f(q) 계산을 위한 가우시안 파라미터(a, b, c)를 포함합니다.

        atom_distance_matrix (numpy.ndarray):
            [EN] A symmetric matrix (N x N) of pairwise Euclidean distances between atoms.
                 Calculated by 'create_atom_distance_matrix'.
            [KR] 원자들 사이의 쌍(pairwise) 유클리드 거리를 담은 대칭 행렬 (N x N)입니다.
                 'create_atom_distance_matrix' 함수로 계산된 값입니다.

        qmin (float):
            [EN] Minimum Q value (inverse Angstrom). Default is 0.5.
            [KR] Q 값의 최소범위 (1/A). 기본값은 0.5입니다.

        qmax (float):
            [EN] Maximum Q value (inverse Angstrom). Default is 20.
            [KR] Q 값의 최대범위 (1/A). 기본값은 20입니다.

        qstep (float):
            [EN] Step size for Q iteration. Default is 0.05.
            [KR] Q 계산 간격. 기본값은 0.05입니다.

        return_Iq (bool):
            [EN] If True, returns detailed data including I(q), F(q), and form factor averages.
            [KR] True일 경우, I(q), F(q), 형상 인자 평균값 등 상세 데이터를 함께 반환합니다.

    Returns
    -------
        If return_Iq is False:
            (q_range, list_Sq)
            - q_range (numpy.ndarray): [EN] Generated Q axis / [KR] 생성된 Q 축 데이터
            - list_Sq (numpy.ndarray): [EN] Calculated Structure Factor S(q) / [KR] 계산된 구조 인자 S(q)

        If return_Iq is True:
            (q_range, list_Iq, list_Sq, list_Fq, mean_sq_fi, sq_mean_fi)
            - list_Iq (numpy.ndarray): [EN] Total Scattering Intensity I(q) / [KR] 총 산란 강도 I(q)
            - list_Fq (numpy.ndarray): [EN] Reduced Structure Function F(q) = Q * (S(q) - 1) / [KR] 환원 구조 함수
            - mean_sq_fi (numpy.ndarray): [EN] Average of squared form factors <f^2> / [KR] 형상 인자 제곱의 평균
            - sq_mean_fi (numpy.ndarray): [EN] Squared average of form factors <f>^2 / [KR] 평균 형상 인자의 제곱
    """
    num_atom = len(atom_indices)
    atom_indices = np.asarray(atom_indices)
    q_range = np.arange(qmin, qmax, qstep)

    # Form factor f(q) for every element type at every q, computed in one vectorized shot
    # (was a `for k in range(num_fact)` Python loop re-run per q).
    fi_by_type = _cal_fi(scattering_factors.T[:, :, None], q_range)  # (num_fact, len(q_range))
    fi_by_atom = fi_by_type[atom_indices, :]  # (num_atom, len(q_range))

    fi_sum = fi_by_atom.sum(axis=0)
    fi2_sum = (fi_by_atom**2).sum(axis=0)
    mean_sq_fi = fi2_sum / num_atom  # <f^2>
    sq_mean_fi = (fi_sum / num_atom) ** 2  # <f>^2

    list_Iq = np.asarray(_debye_sum(fi_by_atom, atom_distance_matrix, q_range))
    list_Sq = (list_Iq - num_atom * mean_sq_fi) / (num_atom * sq_mean_fi) + 1
    list_Fq = q_range * (list_Sq - 1)

    if return_Iq:
        return q_range, list_Iq, list_Sq, list_Fq, mean_sq_fi, sq_mean_fi
    else:
        return q_range, list_Sq


def compton_cal_exp(
    atom_indices: list[int] | NDArray[np.int64],
    compton_scat_parms: NDArray[np.float64],
    compton_scattering_factors: NDArray[np.float64],
    atomic_number: NDArray[np.float64],
    qmin: float,
    qmax: float,
    qstep: float,
    wavelength: float,
    alpha: float,
    weights: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], list[float]]:
    """
    [EN] Calculate total experimental Compton scattering intensity.

         Includes Breit-Dirac recoil factor correction.
         Supports BOTH integer and fractional compositions:
           - Integer (default): pass 'atom_indices' (one entry per atom); each
             unique element is counted with weight 1 per atom.
           - Fractional: pass 'weights' (per-unique-element amounts from
             composition_weights). The Compton average is normalised by the total
             weight, so fractional amounts (e.g. Li0.2Co0.36...) are used directly
             and give the same result as their integer-scaled form.
    [KR] 전체 실험적 콤프턴 산란 강도를 계산합니다.
         Breit-Dirac 반동(Recoil) 보정이 포함됩니다.
         정수/소수 조성 모두 지원합니다:
           - 정수(기본): 'atom_indices' 전달 (원자당 항목, 원자별 weight=1).
           - 소수: composition_weights의 'weights' 전달 (고유 원소별 양).
             Compton 평균은 총 weight로 정규화되므로 소수 조성(예: Li0.2Co0.36...)을
             그대로 사용해도 정수배 형태와 동일한 결과를 줍니다.

    Args:
        atom_indices (array): [EN] Per-atom element index (integer composition)
                              [KR] 원자별 원소 인덱스 (정수 조성)
        weights (array, optional): [EN] Per-unique-element amounts (fractional
                                   composition). If given, used instead of
                                   atom_indices for the averaging.
                                   [KR] 고유 원소별 양 (소수 조성). 주어지면
                                   평균 계산에서 atom_indices 대신 사용됨.
        wavelength (float): [EN] X-ray wavelength (Angstrom) / [KR] X선 파장
        alpha (float): [EN] Recoil parameter (usually 2 or 3) / [KR] 반동 파라미터

    Returns
    -------
        q_range (numpy.ndarray): [EN] Q axis / [KR] Q 축 데이터
        list_compton_scat (list): [EN] Compton intensity / [KR] 콤프턴 산란 강도
    """
    me = 9.109534e-31  # [EN] Electron mass (kg) / [KR] 전자 질량
    C = 2.99792458e18  # [EN] Speed of light (Angstrom/s) / [KR] 빛의 속도
    h = 6.62607015e-14  # [EN] Planck constant / [KR] 플랑크 상수
    part_A = 2.0 * h * wavelength / me / C

    num_fact = len(compton_scattering_factors)
    q_range = np.arange(qmin, qmax, qstep)
    list_compton_scat = []

    # [EN] Determine per-unique-element weights c_k and the total weight.
    #      - weights given  → fractional composition, use as-is.
    #      - else           → integer composition, count atoms per element from
    #                         atom_indices (bincount over 0..num_fact-1).
    # [KR] 고유 원소별 weight c_k와 총합을 결정.
    #      - weights 있음 → 소수 조성, 그대로 사용.
    #      - 없음         → 정수 조성, atom_indices에서 원소별 원자 수를 셈.
    if weights is not None:
        c_k = np.asarray(weights, dtype=float)  # (num_fact,)
    else:
        c_k = np.bincount(np.asarray(atom_indices), minlength=num_fact).astype(float)  # (num_fact,)
    total_c = np.sum(c_k)  # Σ c_k

    # [EN] Pre-fetch atomic numbers per unique element as a float array.
    # [KR] 고유 원소별 원자번호를 실수 배열로 준비.
    Z_k = np.asarray(atomic_number, dtype=float)  # (num_fact,)

    for q in q_range:
        part_B = (q / (4.0 * np.pi)) ** 2.0
        # [EN] Breit-Dirac recoil factor / [KR] Breit-Dirac 반동 인자
        BD_recoil_fact = (part_A * part_B + 1) ** (-alpha)

        # [EN] Compton form factors for all unique elements at this q.
        # [KR] 이 q에서 모든 고유 원소의 Compton form factor.
        list_fi = np.array(
            [__cal_compton_fi(compton_scat_parms[int(atomic_number[k]) - 1], q) for k in range(num_fact)]
        )

        # [EN] Weighted sums over unique elements (c_k as the amount of each element).
        #      Σ c_k·Z_k  and  Σ c_k·(f_k²/Z_k), both normalised by Σ c_k.
        # [KR] 고유 원소에 대한 가중 합 (c_k = 각 원소의 양).
        #      Σ c_k·Z_k, Σ c_k·(f_k²/Z_k)를 Σ c_k로 정규화.
        atomic_number_sum = np.sum(c_k * Z_k)
        fi2_sum = np.sum(c_k * (list_fi**2) / Z_k)

        compton_scat = BD_recoil_fact * (atomic_number_sum / total_c - fi2_sum / total_c)
        list_compton_scat.append(compton_scat)
    return q_range, list_compton_scat


def create_atom_distance_matrix(atom_positions: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    [EN] Calculate the pairwise Euclidean distance matrix for all atoms.

         This matrix is essential for the Debye scattering equation (sin(qr)/qr).
    [KR] 모든 원자 쌍 사이의 유클리드 거리 행렬을 계산합니다.
         이 행렬은 Debye 산란 공식 (sin(qr)/qr) 계산에 필수적입니다.

    Args:
        atom_positions (numpy.ndarray):
            [EN] Input array of atomic coordinates with shape (N, 3).
                 N is the number of atoms, and 3 represents (x, y, z) coordinates.
                 (Data Type: float)
            [KR] (N, 3) 크기의 원자 좌표 입력 배열입니다.
                 N은 원자의 개수이며, 3은 (x, y, z) 좌표를 나타냅니다.
                 (데이터 타입: float)

    Returns
    -------
        distance_matrix (numpy.ndarray):
            [EN] Output symmetric distance matrix with shape (N, N).
                 Element (i, j) represents the Euclidean distance between atom i and atom j.
                 (Data Type: float)
            [KR] (N, N) 크기의 대칭 거리 행렬 결과입니다.
                 (i, j) 요소는 i번째 원자와 j번째 원자 사이의 유클리드 거리를 나타냅니다.
                 (데이터 타입: float)
    """
    # [EN] Calculate distance between every pair of points using scipy.spatial.distance.cdist
    # [KR] scipy.spatial.distance.cdist를 사용하여 모든 점 쌍 사이의 거리를 계산합니다.
    return cdist(atom_positions, atom_positions)
