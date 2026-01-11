import numpy as np
from scipy.optimize import linear_sum_assignment

def match_angles(theta_ref: np.ndarray, theta_cur: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Сопоставляет текущий и предшествующий векторы углов.
    Переставляет элементы текущего вектора для того, чтобы
    минимизировать расстояние между текущим и предшествующим
    векторами.

    Parameters
    ---------------------------------------------------------------------------
    theta_ref: np.ndarray
        Вектор углов на предшествующей итерации.
    theta_cur: np.ndarray 
        Вектор углов на текущей итерации.

    Returns
    ---------------------------------------------------------------------------
    theta_cur_matched: np.ndarray
        Переупорядоченный вектор текущих углов.
    col_ind: np.ndarray
        Индексы, соответствующие перестановке.
    """
    # cost matrix |θ_i - θ_j|
    cost = np.abs(theta_ref[:, None] - theta_cur[None, :])

    row_ind, col_ind = linear_sum_assignment(cost)

    theta_cur_matched = theta_cur[col_ind]
    return theta_cur_matched, col_ind


#theta_ref = np.array([-0.4, 0.1, 0.6])
#theta_cur = np.array([0.58, -0.42, 0.12])

#theta_cur_matched, perm = match_angles(theta_ref, theta_cur)