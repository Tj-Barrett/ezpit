from math import factorial

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.linalg import LinAlgError, cho_factor, cho_solve


def bandwidth_to_lambda(bandwidth: float, order: int) -> float:
    """Convert a Savitzky-Golay-like bandwidth to Whittaker lambda."""
    return (2 * factorial(order)) ** 2 / (bandwidth ** (2 * order))


def make_dT_d(n: int, d: int):
    """
    [EN] Constructs the penalty matrix (D.T * D) for Whittaker smoothing.

    [KR] 스무딩을 위한 차분 행렬의 곱(D.T * D)을 생성합니다.
    """

    def diff_matrix(k: int, n: int):
        D = sp.eye(n, format="csc")
        for _ in range(k):
            D = D[1:] - D[:-1]  # [EN] Difference operator / [KR] 차분 연산자
        return D

    D = diff_matrix(d, n)
    return D.T @ D


def noise_gain_to_lambda(gain: float, order: int) -> float:
    """Convert noise gain parameter to Whittaker lambda."""
    return (2 * factorial(order)) ** 2 / gain


def smooth_like_savitzky_golay(y: NDArray[np.float64], bandwidth: float, order: int):
    """Whittaker smoothing configured to mimic Savitzky-Golay behavior."""
    lambda_ = bandwidth_to_lambda(bandwidth, order)
    return smooth_whittaker(y, lambda_, order)


def smooth_with_noise_gain(y: NDArray[np.float64], gain: float = 1e-4, order: int = 2):
    """Whittaker smoothing using a noise gain parameter."""
    lambda_ = noise_gain_to_lambda(gain, order)
    return smooth_whittaker(y, lambda_, order)


# =============================================================================
# [Added Section] Whittaker-Henderson Smoothing Functions (12/2025 Added)
# =============================================================================

def batch_smooth_whittaker(y_2d: NDArray[np.float64], lambda_: float = 1600.0, order: int = 2) -> NDArray[np.float64]:
    """Apply Whittaker-Henderson smoothing to each row of a 2D array."""
    return np.array([smooth_whittaker(row, lambda_, order) for row in y_2d])


def smooth_whittaker(
    y: NDArray[np.float64] | list[float],
    lambda_: float = 1600.0,
    order: int = 2,
) -> NDArray[np.float64]:
    """
    [EN] Whittaker-Henderson smoothing algorithm.

         It minimizes: sum((y - z)^2) + lambda * sum((delta^order z)^2)
    [KR] Whittaker-Henderson 스무딩 알고리즘입니다.
         데이터 충실도(원본과의 차이)와 부드러움(미분값의 크기) 사이의 균형을 맞춥니다.

    Args:
        y (array): [EN] Input data to be smoothed / [KR] 스무딩할 입력 데이터
        lambda_ (float): [EN] Smoothing parameter (Larger = Smoother) / [KR] 스무딩 강도 (클수록 부드러워짐)
        order (int): [EN] Difference order (Usually 2) / [KR] 미분 차수 (보통 2 사용)
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    if order >= n:
        raise ValueError(f"Error: order={order} is too large for data length={n}. It must be less than {n}.")
    if lambda_ <= 0:
        raise ValueError(f"Error: lambda_ must be positive. Got {lambda_}.")

    try:
        DTD = make_dT_d(n, order)  # [EN] Create penalty matrix / [KR] 페널티 행렬 생성
        A = sp.eye(n, format="csc") + lambda_ * DTD  # A = I + lambda * D'D
        c, low = cho_factor(A.toarray())  # [EN] Cholesky factorization / [KR] 촐레스키 분해

        # [EN] Solve linear system Ay = z
        # [KR] 선형 연립방정식 Ay = z 를 풀어서 스무딩된 값 z를 구함
        return cho_solve((c, low), y)

    except LinAlgError as e:
        raise RuntimeError(
            f"Numerical error during smoothing: {e}. Consider lowering order or increasing lambda."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error during smoothing: {e}. Check your lambda and order values.") from e
