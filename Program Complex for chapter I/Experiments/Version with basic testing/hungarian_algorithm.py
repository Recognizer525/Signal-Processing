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
    cost = np.abs(theta_ref[:, None] - theta_cur[None, :])

    row_ind, col_ind = linear_sum_assignment(cost)

    theta_cur_matched = theta_cur[col_ind]
    return theta_cur_matched, col_ind


def match_powers(power_ref: np.ndarray, power_cur: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Сопоставляет текущую и предшествующую ковариацию сигналов.
    Переставляет элементы диагонали текущей матрицы для того, чтобы
    минимизировать расстояние между текущей и предшествующей матрицей.

    Parameters
    ---------------------------------------------------------------------------
    power_ref: np.ndarray
        Ковариация сигналов на предшествующей итерации.
    power_cur: np.ndarray 
        Ковариация сигналов на текущей итерации.

    Returns
    ---------------------------------------------------------------------------
    power_cur_matched: np.ndarray
        Переупорядоченная текущая ковариация сигналов.
    col_ind: np.ndarray
        Индексы, соответствующие перестановке.
    """
    power_ref = np.diag(power_ref)
    power_cur = np.diag(power_cur)

    print(f"power_ref={power_ref}")
    print(f"power_cur={power_cur}")

    cost = np.abs(power_ref[:, None] - power_cur[None, :])

    row_ind, col_ind = linear_sum_assignment(cost)

    power_cur_matched = np.diag(power_cur[col_ind])
    return power_cur_matched, col_ind




