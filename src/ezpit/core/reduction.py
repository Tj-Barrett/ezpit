from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import interpolate

from ezpit.core.io import load_qiq_file
from ezpit.core.processing import _cal_fi


# ----------------------------------------------------------------------------------
# [Cal_expSq] Core Analysis Function / 핵심 분석 함수
# ----------------------------------------------------------------------------------
def cal_expSq(
    atom_indices: list[int] | NDArray[np.int64],
    scattering_factors: NDArray[np.float64],
    expqiq: str | NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]],
    bkgqiq: str | NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]] | None,
    qmin: float = 0,
    qmax: float = 25,
    qstep: float = 0.01,
    background_scale: float = 1.1,
    poly_order: float = 11.0,
    return_Iq: bool = False,
    weights: NDArray[np.float64] | None = None,
) -> (
    tuple[
        NDArray[np.float64],  # q_range
        NDArray[np.float64],  # list_Iq
        NDArray[np.float64],  # scaled_expIq
        NDArray[np.float64],  # list_scaled_bkgIq
        NDArray[np.float64],  # list_Sq
        NDArray[np.float64],  # norm_list_Sq
        NDArray[np.float64],  # list_Fq
        NDArray[np.float64],  # mean_sq_fi
        NDArray[np.float64],  # sq_mean_fi
        NDArray[np.float64],  # polynomial_for_sq
        NDArray[np.float64],  # normalized_intensity
        NDArray[np.float64],  # normal_scattering_factor
        float,  # normalization_scale (scalar)
    ]
):
    """
    [EN] Core function to calculate Structure Factor S(q) and F(q) from experimental I(q).

         Supports BOTH integer and fractional compositions:
           - Integer (default): pass 'atom_indices' (one entry per atom, from
             group_atoms), and each element's form factor is counted with weight 1.
           - Fractional: pass 'weights' (per-unique-element amounts from
             composition_weights). 'scattering_factors' must then be ordered to
             match the unique-element order of 'weights', and 'atom_indices' is
             ignored for the averaging. Fractional amounts (e.g. Li0.2Co0.36...)
             are used directly since <f>, <f^2> are normalised by the total weight.
    [KR] 실험 I(q)로부터 S(q), F(q)를 계산하는 핵심 함수. 정수/소수 조성 모두 지원.
           - 정수(기본): group_atoms의 'atom_indices' 전달 (원자당 항목, weight=1).
           - 소수: composition_weights의 'weights' 전달 (고유 원소별 양).
             이때 'scattering_factors'는 weights의 고유 원소 순서와 일치해야 하며,
             평균 계산에서 atom_indices는 무시됩니다. 소수 조성(예: Li0.2Co0.36...)은
             <f>, <f^2>가 총 weight로 정규화되므로 그대로 사용됩니다.

    Steps (단계):
    1. Load Data / 2. Interpolate / 3. Background Subtraction
    4. Form Factors (weighted) / 5. S(q) (PDFgetX3) / 6. Polynomial Correction

    Args:
        atom_indices (array): [EN] Per-atom element index (integer composition)
                              [KR] 원자별 원소 인덱스 (정수 조성)
        scattering_factors (ndarray): [EN] Per-unique-element scattering params
                                      [KR] 고유 원소별 산란 파라미터
        weights (array, optional): [EN] Per-unique-element amounts (fractional
                                   composition). If given, overrides atom_indices
                                   for the <f>, <f^2> averaging.
                                   [KR] 고유 원소별 양 (소수 조성). 주어지면 평균
                                   계산에서 atom_indices 대신 사용됨.
        poly_order (float): [EN] Polynomial order for correction / [KR] 보정 다항식 차수
        background_scale (float): [EN] Background scale factor / [KR] 배경 스케일

    Returns
    -------
        Tuple of 13 values / 13개 값의 튜플:
            q_range, list_Iq, scaled_expIq, list_scaled_bkgIq, list_Sq, norm_list_Sq,
            list_Fq, mean_sq_fi, sq_mean_fi, polynomial_for_sq,
            normalized_intensity, normal_scattering_factor, normalization_scale
          - normalized_intensity     : I / <f>^2           (array)
          - normal_scattering_factor : <f^2> / <f>^2        (array)
          - normalization_scale      : PDFgetX3 least-squares scale (scalar)
    """
    # 1. Load Data
    if isinstance(expqiq, str):
        # [EN] If input is a file path string, load from file
        # [KR] 입력이 파일 경로 문자열이면 파일에서 로드
        data_exp = load_qiq_file(expqiq, min_cols=2, usecols=(0, 1))
        exp_q = data_exp[:, 0]
        exp_Iq = data_exp[:, 1]
    else:
        # [EN] If input is already an array
        # [KR] 입력이 이미 배열인 경우
        exp_q = expqiq[0]
        exp_Iq = expqiq[1]

    if isinstance(bkgqiq, str):
        data_bkg = load_qiq_file(bkgqiq, min_cols=2, usecols=(0, 1))
        bkg_Iq = data_bkg[:, 1]
    elif bkgqiq is not None:
        if isinstance(bkgqiq, np.ndarray) and bkgqiq.ndim > 1:
            bkg_Iq = bkgqiq[1]
        elif isinstance(bkgqiq, (list, tuple)) and len(bkgqiq) == 2 and hasattr(bkgqiq[0], "__len__"):
            bkg_Iq = bkgqiq[1]
        else:
            bkg_Iq = bkgqiq
    else:
        bkg_Iq = np.zeros_like(exp_Iq)  # [EN] Fill with zeros if no background / [KR] 배경 데이터가 없으면 0으로 채움

    # 2. Interpolate
    q_range = np.arange(qmin, qmax, qstep)
    scaled_expIq = np.interp(q_range, exp_q, exp_Iq)  # [EN] Interpolate exp data / [KR] 실험 데이터 보간
    scaled_bkgIq = np.interp(q_range, exp_q, bkg_Iq)  # [EN] Interpolate bkg data / [KR] 배경 데이터 보간

    # 3. Background Subtraction
    # [EN] Calculate Net Intensity: I_net = I_exp - scale * I_bkg
    # [KR] 순수 강도 계산: 실험값 - (스케일 * 배경값)
    list_scaled_bkgIq = background_scale * scaled_bkgIq
    list_Iq = scaled_expIq - list_scaled_bkgIq

    # 4. Calculate Form Factors (VECTORIZED, weight-aware)
    # [EN] Vectorized form factor calculation. Supports both:
    #        - integer composition via atom_indices (weight 1 per atom), and
    #        - fractional composition via 'weights' (per-unique-element amounts).
    #      Both reduce to the SAME normalised averages:
    #        <f>   = Σ_k(c_k · f_k)   / Σ_k(c_k)
    #        <f^2> = Σ_k(c_k · f_k^2) / Σ_k(c_k)
    #      For the integer path, c_k is the number of atoms of element k
    #      (obtained by counting atom_indices), which is exactly equivalent to
    #      the previous per-atom sum divided by num_atom.
    # [KR] 벡터화된 형상 인자 계산. 정수(atom_indices) / 소수(weights) 모두 지원.
    #      둘 다 동일한 정규화 평균으로 귀결됩니다.
    #      정수 경로에서 c_k는 원소 k의 원자 수(atom_indices를 세어 구함)이며,
    #      이는 기존의 원자별 합을 num_atom으로 나눈 것과 정확히 동일합니다.

    # [EN] (1) Form factors for ALL unique elements over the whole q_range.
    #          Result shape: (num_unique_elements, len(q_range))
    # [KR] (1) 모든 고유 원소의 전체 q_range form factor. shape: (원소수, q수)
    list_fi_all_q = np.array([_cal_fi(sf, q_range) for sf in scattering_factors])

    # [EN] (2) Determine per-unique-element weights c_k.
    # [KR] (2) 고유 원소별 weight c_k 결정.
    if weights is not None:
        # [EN] Fractional composition: use given weights directly.
        # [KR] 소수 조성: 주어진 weight 직접 사용.
        c_k = np.asarray(weights, dtype=float)  # shape: (num_unique_elements,)
    else:
        # [EN] Integer composition: count how many atoms per unique element
        #      from atom_indices (0,1,2,... referring to scattering_factors rows).
        # [KR] 정수 조성: atom_indices에서 고유 원소별 원자 수를 셈.
        num_unique = len(scattering_factors)
        c_k = np.bincount(np.asarray(atom_indices), minlength=num_unique).astype(float)

    total_c = np.sum(c_k)  # Σ c_k (= num_atom for integer)

    # [EN] (3) Weighted averages over unique elements (broadcast c_k over q axis).
    #          list_fi_all_q shape: (num_unique, num_q); c_k shape: (num_unique,)
    # [KR] (3) 고유 원소에 대한 가중 평균 (c_k를 q축으로 broadcast).
    weighted_fi = c_k[:, None] * list_fi_all_q  # c_k · f_k
    weighted_fi_sq = c_k[:, None] * (list_fi_all_q**2)  # c_k · f_k²

    sum_fi = np.sum(weighted_fi, axis=0)  # Σ c_k · f_k
    sum_fi_sq = np.sum(weighted_fi_sq, axis=0)  # Σ c_k · f_k²

    sq_mean_fi = (sum_fi / total_c) ** 2  # <f>²
    mean_sq_fi = sum_fi_sq / total_c  # <f²>

    # [EN] num_atom kept for backward compatibility (integer path only).
    # [KR] num_atom은 하위 호환성을 위해 유지 (정수 경로).
    int(round(total_c))

    # 6. Calculate S(q) — PDFgetX3 normalization (default)
    # ------------------------------------------------------------------
    # [EN] X-ray scattering-factor normalization following the ad hoc
    #      data-reduction approach of PDFgetX3. The experimental I(q) is in
    #      arbitrary units (not normalized per incident flux nor per number of
    #      scatterers), so a least-squares scale factor is required to bring the
    #      normalized intensity onto the normal scattering-factor curve before
    #      computing S(q). Without this scale the S(q) amplitude is wrong
    #      (the deviation from 1 is off by a large factor).
    #        normalized_intensity     = I / <f>^2
    #        normal_scattering_factor = <f^2> / <f>^2
    #        normalization_scale = dot(normalized_intensity, normal_scattering_factor)
    #                              / dot(normalized_intensity, normalized_intensity)
    #        S(q) = normalization_scale * normalized_intensity
    #               - normal_scattering_factor + 1
    # [KR] PDFgetX3의 ad hoc 데이터 처리 방식을 따르는 X-ray 산란인자 정규화.
    #      실험 I(q)는 임의 단위(입사 강도/산란체 수로 정규화되지 않음)이므로,
    #      정규화 강도를 normal scattering-factor 곡선에 맞추는 최소자승 스케일
    #      상수가 필요하다. 이 스케일이 없으면 S(q) 진폭이(1로부터의 편차가)
    #      큰 배수로 어긋난다.
    #
    # Reference:
    #   Juhas, P., Davis, T., Farrow, C. L. & Billinge, S. J. L. (2013).
    #   PDFgetX3: a rapid and highly automatable program for processing powder
    #   diffraction data into total scattering pair distribution functions.
    #   J. Appl. Cryst. 46, 560-566.
    # ------------------------------------------------------------------
    normalized_intensity = list_Iq / sq_mean_fi  # I / <f>^2   (sq_mean_fi = <f>^2)
    normal_scattering_factor = mean_sq_fi / sq_mean_fi  # <f^2> / <f>^2 (mean_sq_fi = <f^2>)
    normalization_scale = np.dot(normalized_intensity, normal_scattering_factor) / np.dot(
        normalized_intensity, normalized_intensity
    )
    list_Sq = normalization_scale * normalized_intensity - normal_scattering_factor + 1

    # ------------------------------------------------------------------
    # [EN] Standard formula (kept for reference, NOT used by default).
    #      To revert to the standard formula, comment out the PDFgetX3 block
    #      above and uncomment the line below.
    # [KR] 표준 공식 (참고용으로 보존, 기본 사용 안 함).
    #      표준 공식으로 되돌리려면 위 PDFgetX3 블록을 주석 처리하고
    #      아래 줄의 주석을 해제하세요.
    #
    #   list_Sq = (list_Iq - num_atom * mean_sq_fi) / (num_atom * sq_mean_fi) + 1
    # ------------------------------------------------------------------

    # 7. Polynomial Correction for F(q)
    qmax_for_sq = float(np.max(q_range))

    # [EN] Helper function to calculate polynomial fit
    # [KR] 다항식 적합(Fit)을 계산하는 내부 함수
    def _poly_for_sq_at_order(order_int: int):
        order_int = max(int(order_int), 1)
        qscaled = q_range / qmax_for_sq
        V = np.vander(qscaled, order_int + 1)
        A = V[:, :-1]
        b = q_range * (list_Sq - 1.0)
        p, _, _, _ = np.linalg.lstsq(A, b, rcond=None)  # [EN] Least squares fit / [KR] 최소자승법
        vand2 = np.vander([qmax_for_sq], order_int + 1)[0, :-1]
        new_p = p / vand2
        final_p = np.poly1d(new_p)
        return final_p(q_range)

    # [EN] Handle non-integer polynomial orders
    # [KR] 소수점 차수(Non-integer) 다항식 처리 (두 정수 차수의 가중 평균)
    po = float(poly_order)
    if po.is_integer():
        polynomial_for_sq = _poly_for_sq_at_order(int(round(po)))
    else:
        lo = int(np.floor(po))
        hi = int(np.ceil(po))
        w_hi = po - lo
        w_lo = 1.0 - w_hi
        poly_lo = _poly_for_sq_at_order(lo)
        poly_hi = _poly_for_sq_at_order(hi)
        polynomial_for_sq = w_lo * poly_lo + w_hi * poly_hi

        # print('w_hi = ', w_hi)
        # print('w_lo = ', w_lo)
        # print('lo = ', lo)
        # print('hi = ', hi)

    # [EN] Normalize S(q) and calculate F(q)
    # [KR] S(q) 정규화 및 F(q) 계산
    norm_list_Sq = list_Sq - polynomial_for_sq  # [EN] Remove polynomial background / [KR] 다항식 배경 제거
    list_Fq = q_range * (norm_list_Sq - 1.0)

    # NOTE: both branches currently return the identical 13-value tuple in the
    # identical order - return_Iq has no effect on the result here
    # tests/test_processing.py::test_cal_expSq_return_Iq_true_matches_false.
    if return_Iq:
        return (
            q_range,
            list_Iq,
            scaled_expIq,
            list_scaled_bkgIq,
            list_Sq,
            norm_list_Sq,
            list_Fq,
            mean_sq_fi,
            sq_mean_fi,
            polynomial_for_sq,
            normalized_intensity,
            normal_scattering_factor,
            normalization_scale,
        )
    else:
        return (
            q_range,
            list_Iq,
            scaled_expIq,
            list_scaled_bkgIq,
            list_Sq,
            norm_list_Sq,
            list_Fq,
            mean_sq_fi,
            sq_mean_fi,
            polynomial_for_sq,
            normalized_intensity,
            normal_scattering_factor,
            normalization_scale,
        )


def cal_fq(qmin: float, qmax: float, Sq: NDArray[np.float64], qstep: float = 0.01
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    [EN] Calculate the Reduced Structure Function F(q) from Structure Factor S(q).

         This function generates the Q-axis based on the length of Sq and computes F(q) = Q * (S(q) - 1).
    [KR] 구조 인자 S(q)로부터 환원 구조 함수 F(q)를 계산합니다.
         Sq의 길이에 맞춰 Q축을 생성하고 F(q) = Q * (S(q) - 1) 공식을 적용합니다.

    Args:
        qmin (float):
            [EN] Minimum Q value (inverse Angstrom).
            [KR] Q 값의 최소값 (단위: 1/A).

        qmax (float):
            [EN] Maximum Q value (inverse Angstrom).
            [KR] Q 값의 최대값 (단위: 1/A).

        Sq (numpy.ndarray or list):
            [EN] 1D array of Structure Factor S(q) data.
            [KR] 구조 인자 S(q) 데이터가 담긴 1차원 배열입니다.

        qstep (float, optional):
            [EN] Step size for Q. Default is 0.01.
                 (Note: In this specific implementation using np.linspace, qstep is
                 not directly used to determine the grid size, but is included for
                 parameter consistency).
            [KR] Q 값의 간격. 기본값은 0.01입니다.
                 (참고: np.linspace를 사용하는 이 구현에서는 그리드 크기 결정에 직접 사용되지
                 않지만, 파라미터 일관성을 위해 포함되었습니다).

    Returns
    -------
        tuple (numpy.ndarray, numpy.ndarray):
            [EN] A tuple containing (Fq, q_range).
                 - Fq: Calculated Reduced Structure Function F(q).
                 - q_range: The generated Q-axis values.
            [KR] (Fq, q_range) 튜플을 반환합니다.
                 - Fq: 계산된 환원 구조 함수 F(q).
                 - q_range: 생성된 Q축 값들.
    """
    # [EN] Generate Q axis evenly spaced between qmin and qmax matching the length of Sq
    # [KR] Sq의 데이터 길이에 맞춰 qmin과 qmax 사이의 등간격 Q축을 생성합니다.
    q_range = np.linspace(qmin, qmax, len(Sq), endpoint=False)

    # [EN] Calculate F(q) = Q * (S(q) - 1) and return with Q axis
    # [KR] F(q) = Q * (S(q) - 1) 계산 후 Q축과 함께 반환
    return (Sq - 1) * q_range, q_range


def cal_expGr_fft(
    q: NDArray[np.float64],
    Sq_or_Fq: NDArray[np.float64],
    rmin: float = 0,
    rmax: float = 100,
    rstep: float = 0.01,
    is_Fq: bool = False,
    pad_mode: str = "zero",
    low_q_mode: str = "anchor",
    extrapolate_type: str = "linear",
    return_padding: bool = False,
) -> (
    tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    | tuple[
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
        dict[str, NDArray[np.floating[Any]] | float | int],
    ]
    | tuple[None, None]
    | tuple[None, None, None]
):
    """
    [EN] Calculate G(r) from S(q) or F(q) using IFFT with low-q extrapolation.

         and high-q padding.

         Low-Q extrapolation modes (low_q_mode):
           - "anchor" (default): F(q=0)=0 enforced (physically correct),
                                 straight line from (0,0) to (qmin, F(qmin)).
           - "linear": scipy interp1d extrapolation (legacy method),
                      uses the slope of the first two data points; may yield F(0)≠0.

         Nyquist check: if rmax > pi/qstep, F(q) is interpolated onto a finer q grid
                        and only the alias-free first half of the IFFT output is used.

         Visualization mode (return_padding=True):
           Additionally returns the three segments of the padded F(q) array
           used internally by the IFFT — useful for reproducing Figure 3 of the
           EZPDF paper (low-Q extrapolation in red, measured F(q) in black,
           high-Q zero padding in gray).

    [KR] IFFT를 사용하여 S(q) 또는 F(q)로부터 G(r)을 계산합니다.

         Low-Q 외삽 방식 (low_q_mode):
           - "anchor" (기본): F(q=0)=0을 강제 (물리적으로 정확),
                              (0,0)에서 (qmin, F(qmin))까지 직선.
           - "linear": scipy interp1d 외삽 (구버전 방식),
                      처음 두 점의 기울기 사용; F(0) ≠ 0 가능.

         Nyquist 체크: rmax > pi/qstep이면 F(q)를 더 촘촘한 q 그리드로 보간하고,
                       IFFT 출력의 alias-free 앞 절반만 사용합니다.

         시각화 모드 (return_padding=True):
           IFFT 내부에서 사용된 padded F(q)의 3개 구간을 추가로 반환합니다.
           EZPDF 논문 Figure 3 재현용 (Low-Q 외삽=빨강, 실험 F(q)=검정,
           High-Q zero padding=회색).

    Args:
        q (numpy.ndarray)       : Q-axis (1/Å)
        Sq_or_Fq (numpy.ndarray): S(q) or F(q) data
        rmin, rmax (float)      : real-space r range (Å)
        rstep (float)           : r step size (Å)
        is_Fq (bool)            : True → input is F(q), False → input is S(q)
        pad_mode (str)          : high-Q padding method
                                  'zero'     → fill with 0 (default)
                                  'decay'    → linear decay from last value to 0
                                  'constant' → extend last value
        low_q_mode (str)        : low-Q extrapolation method
                                  'anchor' → F(0)=0 enforced (default, physically correct)
                                  'linear' → scipy interp1d extrapolation (legacy)
        extrapolate_type (str)  : interpolation kind when low_q_mode='linear'
                                  ('linear', 'cubic', 'quadratic', etc.)
        return_padding (bool)   : if True, also return the three F(q) segments
                                  used for IFFT (for Figure 3 visualization)

    Returns
    -------
        if return_padding=False (default):
            (r_list, gr) or (None, None) on error.

        if return_padding=True:
            (r_list, gr, padding_info)  or  (None, None, None) on error
            padding_info is a dict with keys:
                'q_low'         : q values of low-Q extrapolated region (q = 0 → qmin)
                'F_low'         : F(q) values of low-Q extrapolated region (RED in Figure 3)
                'q_exp'         : q values of measured experimental region (qmin → qmax)
                'F_exp'         : F(q) values of measured experimental region (BLACK)
                'q_pad'         : q values of high-Q padding region (qmax →)
                'F_pad'         : F(q) values of high-Q padding region (GRAY)
                'q_full_padded' : full Q axis fed to IFFT (0 → N × qstep)
                'F_full_padded' : full padded F(q) array fed to IFFT (concatenation of all 3 segments)
                'qstep'         : q-step actually used for IFFT (may differ from input
                                  if Nyquist interpolation was applied)
                'total_N'       : total number of points fed to IFFT
    """
    try:
        if len(q) < 2:
            print(f"Error in cal_expGr_fft: q array is too short (len={len(q)}) to calculate qstep.")
            return (None, None, None) if return_padding else (None, None)

        qstep_calc = q[1] - q[0]

        if qstep_calc == 0.0:
            print("Error in cal_expGr_fft: qstep is zero (q[1]==q[0]). Cannot divide by zero.")
            return (None, None, None) if return_padding else (None, None)

        num_point = int(np.ceil(q[0] / qstep_calc))

        if is_Fq:
            Fq = Sq_or_Fq
        else:
            Fq = (Sq_or_Fq - 1) * q

        pad_Fq = np.zeros(num_point)

        if q[0] > 0.0:
            if low_q_mode == "anchor":
                # [EN] Low-q extrapolation with F(q=0) = 0 enforced (anchor method).
                #      F(q) = q*(S(q)-1), so F(0) = 0 by definition regardless of S(0).
                #      A straight line from (0, 0) to (q[0], Fq[0]) fills the gap,
                #      which is mathematically consistent with the definition of F(q).
                # [KR] anchor 방식: F(0)=0을 강제하고 (qmin, F(qmin))까지 직선 연결.
                q_low = np.arange(num_point) * qstep_calc
                pad_Fq[:num_point] = (Fq[0] / q[0]) * q_low
            else:
                # [EN] Linear extrapolation (legacy method via scipy interp1d).
                #      Uses the slope of the first two data points to extrapolate down to q=0.
                #      May result in F(0) ≠ 0 (not physically guaranteed).
                # [KR] Linear 외삽 (구버전, scipy interp1d 사용).
                #      처음 두 점의 기울기로 q=0까지 외삽. F(0) ≠ 0 가능.
                f_inter = interpolate.interp1d(q, Fq, fill_value="extrapolate", kind=extrapolate_type)
                pad_Fq[:num_point] = f_inter(np.arange(num_point) * qstep_calc)

        # [EN] Save the three segments for visualization (BEFORE Nyquist interpolation)
        # [KR] Nyquist 보간 전 3구간을 시각화용으로 저장
        q_low_vis = np.arange(num_point) * qstep_calc
        F_low_vis = pad_Fq.copy()
        q_exp_vis = q.copy()
        F_exp_vis = Fq.copy()

        Fq_vals = np.append(pad_Fq, Fq)
        r_list = np.arange(rmin, rmax + rstep, rstep)

        # ------------------------------------------------------------------
        # Nyquist check:
        #   The IFFT produces total_point values; only the first half
        #   (indices 0 .. total_point//2) corresponds to physically valid
        #   positive-r values, giving r_nyquist = pi / qstep.
        #   If the requested rmax exceeds this limit, the second (aliased)
        #   half of the IFFT output would be used, introducing spurious
        #   oscillations at high r.
        #   Fix: interpolate F(q) onto a finer q grid so that
        #   pi / qstep_fine >= rmax before running the IFFT.
        # ------------------------------------------------------------------
        r_nyquist = np.pi / qstep_calc
        if rmax > r_nyquist:
            qstep_needed = np.pi / rmax
            q_orig = np.arange(len(Fq_vals)) * qstep_calc
            q_fine = np.arange(q_orig[0], q_orig[-1] + qstep_needed, qstep_needed)
            Fq_vals = np.interp(q_fine, q_orig, Fq_vals)
            qstep_calc = qstep_needed

        # --- High-Q padding (pad_mode: 'zero' / 'decay' / 'constant') ---
        num_point = len(Fq_vals)
        total_point = int(2 * np.pi / (rstep * qstep_calc))
        pad_len = total_point - num_point

        if pad_len > 0:
            last_val = Fq_vals[-1]
            if pad_mode == "decay":
                high_q_padding = np.linspace(last_val, 0, pad_len)
            elif pad_mode == "constant":
                high_q_padding = np.full(pad_len, last_val)
            else:  # "zero" (default)
                high_q_padding = np.zeros(pad_len)
            Fq_pad = np.concatenate((Fq_vals, high_q_padding))
        else:
            Fq_pad = Fq_vals
            high_q_padding = np.array([])

        # [EN] Save high-Q padding segment for visualization
        # [KR] High-Q 패딩 구간을 시각화용으로 저장
        q_pad_vis = q[-1] + qstep_calc + np.arange(len(high_q_padding)) * qstep_calc
        F_pad_vis = high_q_padding.copy() if len(high_q_padding) > 0 else np.array([])

        norm = total_point * qstep_calc * 2 / np.pi
        gr_full = norm * np.imag(np.fft.ifft(Fq_pad))

        # Use only the alias-free first half of the IFFT output
        half = total_point // 2
        rfine = np.arange(half) * rstep
        gr_fine = gr_full[:half]
        gr = np.interp(r_list, rfine, gr_fine)

        if return_padding:
            # [EN] Full padded F(q) array fed to IFFT (concatenation of all 3 segments)
            # [KR] IFFT에 실제로 들어간 전체 padded F(q) 배열 (3개 구간 합친 것)
            q_full_padded = np.arange(total_point) * qstep_calc
            F_full_padded = Fq_pad.copy()

            padding_info = {
                "q_low": q_low_vis,
                "F_low": F_low_vis,
                "q_exp": q_exp_vis,
                "F_exp": F_exp_vis,
                "q_pad": q_pad_vis,
                "F_pad": F_pad_vis,
                "q_full_padded": q_full_padded,  # [EN/KR] Full Q axis (0 → N×qstep)
                "F_full_padded": F_full_padded,  # [EN/KR] Full padded F(q) fed to IFFT
                "qstep": qstep_calc,
                "total_N": total_point,
            }
            return r_list, gr, padding_info
        return r_list, gr

    except Exception as e:
        print(f"Error in cal_expGr_fft: {e}")
        return (None, None, None) if return_padding else (None, None)


def cal_expGr_fft_from_Fq(
    q: NDArray[np.float64],
    Fq: NDArray[np.float64],
    rmin: float = 0,
    rmax: float = 100,
    rstep: float = 0.01,
    pad_mode: str = "zero",
    low_q_mode: str = "anchor",
    extrapolate_type: str = "linear",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    [EN] Calculate G(r) directly from F(q) using IFFT.

         Same approach as cal_expGr_fft (anchor/linear low-q extrapolation, Nyquist check).
    [KR] F(q)로부터 직접 IFFT를 사용하여 G(r)을 계산합니다.
         cal_expGr_fft와 동일한 방식 (anchor/linear low-q 외삽, Nyquist 체크).

    Args:
        pad_mode (str)        : 'zero' (default) / 'decay' / 'constant'
        low_q_mode (str)      : 'anchor' (default, F(0)=0 enforced) / 'linear' (legacy)
        extrapolate_type (str): interpolation kind when low_q_mode='linear'
    """
    qstep = q[1] - q[0]
    num_point = int(np.ceil(q[0] / qstep))
    pad_Fq = np.zeros(num_point)

    if q[0] > 0.0:
        if low_q_mode == "anchor":
            # Low-q extrapolation with F(q=0) = 0 enforced (anchor method).
            q_low = np.arange(num_point) * qstep
            pad_Fq[:num_point] = (Fq[0] / q[0]) * q_low
        else:
            # Linear extrapolation (legacy method via scipy interp1d).
            f_inter = interpolate.interp1d(q, Fq, fill_value="extrapolate", kind=extrapolate_type)
            pad_Fq[:num_point] = f_inter(np.arange(num_point) * qstep)

    Fq_vals = np.append(pad_Fq, Fq)
    r_list = np.arange(rmin, rmax + rstep, rstep)

    # Nyquist check: interpolate onto finer q grid if rmax > pi/qstep
    r_nyquist = np.pi / qstep
    if rmax > r_nyquist:
        qstep_needed = np.pi / rmax
        q_orig = np.arange(len(Fq_vals)) * qstep
        q_fine = np.arange(q_orig[0], q_orig[-1] + qstep_needed, qstep_needed)
        Fq_vals = np.interp(q_fine, q_orig, Fq_vals)
        qstep = qstep_needed

    # --- High-Q padding (pad_mode: 'zero' / 'decay' / 'constant') ---
    num_point = len(Fq_vals)
    total_point = int(2 * np.pi / (rstep * qstep))
    pad_len = total_point - num_point

    if pad_len > 0:
        last_val = Fq_vals[-1]
        if pad_mode == "decay":
            high_q_padding = np.linspace(last_val, 0, pad_len)
        elif pad_mode == "constant":
            high_q_padding = np.full(pad_len, last_val)
        else:  # "zero" (default)
            high_q_padding = np.zeros(pad_len)
        Fq_pad = np.concatenate((Fq_vals, high_q_padding))
    else:
        Fq_pad = Fq_vals

    norm = total_point * qstep * 2 / np.pi
    gr_full = norm * np.imag(np.fft.ifft(Fq_pad))

    # Use only alias-free first half
    half = total_point // 2
    rfine = np.arange(half) * rstep
    gr_fine = gr_full[:half]
    gr = np.interp(r_list, rfine, gr_fine)

    return r_list, gr


# ----------------------------------------------------------------------------------
# Lorch Function Added (Requested 12/28/2025)
# ----------------------------------------------------------------------------------
def apply_lorch_function(Q: NDArray[np.float64], FQ: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    [EN] Apply the Lorch function to F(Q) data.

    [KR] F(Q) 데이터에 Lorch 함수를 적용(곱하기)합니다.

    Args:
        Q (numpy.ndarray): [EN] Q values / [KR] Q 값 배열
        FQ (numpy.ndarray): [EN] F(Q) intensity values / [KR] F(Q) 강도 배열

    Returns
    -------
        modified_FQ (numpy.ndarray): [EN] Lorch-modified F(Q) / [KR] Lorch 보정된 F(Q)
    """
    Q = np.asarray(Q)
    FQ = np.asarray(FQ)
    Q_max = np.max(Q)
    lorch_values = lorch_function_sinc(Q, Q_max)
    return FQ * lorch_values


def lorch_function_sinc(Q: NDArray[np.float64] | float, Q_max: float | np.float64
) -> NDArray[np.float64]:
    """
    [EN] Calculate the Lorch modification function M(Q).

         Used to reduce termination ripples in FFT.
    [KR] Lorch 수정 함수 M(Q)를 계산합니다.
         FFT 시 발생하는 종료 리플(Termination Ripple, 물결무늬 노이즈)을 줄이는 데 사용됩니다.
    Formula: M(Q) = sin(pi * Q / Q_max) / (pi * Q / Q_max).
    """
    Q = np.asarray(Q)
    if Q_max == 0:
        return np.ones_like(Q)
    # [EN] numpy.sinc(x) is mathematically sin(pi*x)/(pi*x)
    # [KR] numpy.sinc(x) 함수는 수학적으로 sin(pi*x)/(pi*x) 입니다.
    return np.sinc(Q / Q_max)
