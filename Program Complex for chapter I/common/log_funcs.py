import numpy as np
from . import sensors as sn
from . import diff_sensor_structures as dss

def incomplete_lkhd(X: np.ndarray,
                    theta: np.ndarray, 
                    P: np.ndarray, 
                    Q: np.ndarray) -> np.float64:
    """
    Вычисляет неполное правдоподобие на основании доступных наблюдений 
    и текущей оценки параметров.

    Parameters
    ---------------------------------------------------------------------------
    X: np.ndarray
        Двумерный массив, соответствующий наблюдениям.
    theta: np.ndarray
        Одномерный массив размера (K,1). Соответствует оценке DoA.
    P: np.ndarray
        Оценка ковариационной матрицы исходных сигналов.
    Q: np.ndarray
        Ковариационная матрица шума.

    Returns
    ---------------------------------------------------------------------------
    res: np.float64
        Значение неполного правдоподобия.
    """
    A = dss.A_ULA(X.shape[1], theta)
    R = A @ P @ A.conj().T + Q

    sgn_R, log_det_R = np.linalg.slogdet(R)
    if sgn_R == 0:
        raise ValueError(f"Non-inversible R")

    #print(f"is_pd(R)={sn.is_pd(R)}")
    #print(f"is_pd(P)={sn.is_pd(P)}")
    #print(f"is_pd(Q)={sn.is_pd(Q)}")
    #print(f"Positive P? Ans is {np.all(np.diag(P) >= 0)}")

    inv_R = np.linalg.inv(R)
    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1]+1)
    O = col_numbers * (Indicator == False) - 1
    res = 0
    for i in range(X.shape[0]):
        if set(O[i, ]) != set(col_numbers - 1):
            O_i = O[i, ][O[i, ] > -1]
            R_o = R[np.ix_(O_i, O_i)]
            sgn_R_o, log_det_R_o = np.linalg.slogdet(R_o)
            if sgn_R_o == 0:
                raise ValueError(f"Non-inversible R_o")

            #R_o = R_o + 1e-6 * np.eye(R_o.shape[0])
            res += (- log_det_R_o - (X[i, O_i].T).conj().T @ 
                      np.linalg.inv(R_o) @ (X[i, O_i].T))
        else:
            res += (- log_det_R - (X[i].T).conj().T @ inv_R @ (X[i].T))
    return res.real