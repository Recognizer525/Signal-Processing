import numpy as np

def if_params_converged(angles:np.ndarray, 
                        new_angles: np.ndarray, 
                        P: np.ndarray, 
                        new_P: np.ndarray, 
                        rtol: float = 1e-3) -> bool:
    """
    Проверка достижения сходимости алгоритма на текущей итерации, 
    сравниваются отсортированные вектора/матрицы параметров 
    на текущей и предшествующей итерации.
    """


    idx1 = np.argsort(angles)
    idx2 = np.argsort(new_angles)
    angles[:] = angles[idx1]
    new_angles[:] = new_angles[idx2]
    P = P[np.ix_(idx1, idx1)]
    new_P[:] = new_P[np.ix_(idx2, idx2)]
    if (np.linalg.norm(angles - new_angles) / np.linalg.norm(new_angles) < rtol 
        and np.linalg.norm(P - new_P, ord = 2) / np.linalg.norm(new_P, ord = 2)  < rtol):
        return True
    return False


def if_lkhd_converged(old_lkhd: np.float64,
                      lkhd: np.float64,
                      rtol: float = 1e-6) -> bool:
    """
    Проверяет степень близости старого и нового значения 
    неполного правдоподобия.
    """
    if lkhd == 0:
        return np.abs(lkhd - old_lkhd) < rtol
    return np.abs(lkhd-old_lkhd)/np.abs(lkhd) < rtol