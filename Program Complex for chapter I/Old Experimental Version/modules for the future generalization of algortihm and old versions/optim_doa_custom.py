import numpy as np
import torch
from scipy.optimize import minimize

import diff_sensor_structures as dss

DIST_RATIO = 0.5

def cost_angles(angles: torch.Tensor,
                Sigma_XS: torch.Tensor, 
                P: torch.Tensor,
                coords: torch.Tensor, 
                Q_inv_sqrt: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет значение фробениусовой нормы ||Q^{-1/2}(Sigma_XS-AP)||^2_F, 
    которая подлежит минимизации.

    Parameters
    ---------------------------------------------------------------------------
    angles: torch.Tensor
        Оценка DoA.
    Sigma_XS: torch.Tensor
        Кросс-ковариация наблюдений и сигналов на текущем Е-шаге.
    P: torch.Tensor
        Оценка ковариации сигналов с предыдущей итерации.
    coords: torch.Tensor
        Координаты сенсоров, массив размера (L,3).
    Q_inv_sqrt: torch.Tensor
        Квадратный корень от матрицы, обратной к ковариационной матрице шума.

    Returns
    ---------------------------------------------------------------------------
    res: torch.Tensor
        Значение фробениусовой нормы ||Q^{-1/2}(Sigma_XS-AP)||^2_F.
    """
    A = dss.A_custom_torch(coords, angles)
    E = torch.matmul(Q_inv_sqrt, Sigma_XS - torch.matmul(A, P))  
    return torch.norm(E, 'fro')**2  # скалярный тензор


def find_angles_start(Sigma_XS: torch.Tensor, 
                      angles0_np: np.ndarray,
                      coords: torch.Tensor, 
                      P: torch.Tensor, 
                      Q_inv_sqrt: torch.Tensor,
                      bounds: list|None = None, 
                      method: str = 'SLSQP', 
                      tol: int = 1e-6) -> tuple[np.ndarray, float]:
    """
    Запускает оптимизацию нормы ||Q^{-1/2}(Sigma_XS - AP)||^2_F по вектору DoA.

    Parameters
    ---------------------------------------------------------------------------
    Sigma_XS: torch.Tensor
        Кросс-ковариация наблюдений и сигналов на текущем Е-шаге.
    angles0_np: np.ndarray
        Первое приближение для оценивания DoA. 
        Представляет собой одномерный массив.
    coords: torch.Tensor
        Координаты сенсоров, массив размера (L,3).
    P: torch.Tensor
        Оценка ковариации сигналов с предыдущей итерации.
    Q_inv_sqrt: torch.Tensor
        Квадратный корень от матрицы, обратной к ковариационной матрице шума. 
    bounds: list|None
        Границы, в пределах которых надо искать оптимальное значение вектора DoA.
    method: str = 'SLSQP'
        Метод оптимизации функции потерь для DoA.
    tol: int
        Порог для остановки оптимизационного процесса.

    Returns
    ---------------------------------------------------------------------------
    res.x: np.ndarray
        Оценка DoA, полученная в ходе оптимизационного процесса для
        заданного начального приближения.
    res.fun: float
        Значение минимизируемой фробениусовой нормы для полученной оценки DoA.
    """
    def fun(angles_np: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Возвращает значение функции потерь и значение градиента.
        """
        angles_t = torch.tensor(angles_np, 
                               dtype=torch.float32, 
                               requires_grad=True)
        
        loss = cost_angles(angles_t, Sigma_XS, P, coords, Q_inv_sqrt)
        loss.backward()
        grad = angles_t.grad.detach().numpy()
        return loss.item(), grad

    res = minimize(lambda th: fun(th)[0], angles0_np, 
                   jac=lambda th: fun(th)[1], bounds=bounds, method=method, tol=tol)
    #print(f"Optim.res={res.success}")
    return res.x, res.fun


def find_angles(Sigma_XS_np: np.ndarray, 
                angles0_np: np.ndarray, 
                coords_np: np.ndarray,
                P_np: np.ndarray, 
                Q_inv_sqrt_np: np.ndarray, 
                num_of_starts: int = 7,
                bounds: list|None = None) -> np.ndarray:
    """
    Функция предназначена для поиска оценки DoA, которая минимизирует норму
    ||Q^{-1/2}(Sigma_XS-AP)||^2_F.

    Parameters
    ---------------------------------------------------------------------------
    Sigma_XS_np: np.ndarray
        Кросс-ковариация наблюдений и сигналов на текущем Е-шаге.
    angles0_np: np.ndarray
        Оценка DoA, полученная на предыдущей итерации,
        либо же начальная оценка DoA.
    coords_np: np.ndarray
        Координаты сенсоров, массив размера (L,3).
    P_np: np.ndarray
        Оценка ковариации сигналов с предыдущей итерации.
    Q_inv_sqrt_np: np.ndarray
        Квадратный корень от матрицы, обратной к ковариационной матрице шума.
    num_of_starts: int
        Число начальных приближений относительно которых нужно проводить
        оптимизационный процесс.
    bounds: list|None
        Границы, в пределах которых надо искать оптимальное значение вектора DoA.
    method: str
        Метод оптимизации функции потерь для DoA.

    Returns
    ---------------------------------------------------------------------------
    best_theta: np.ndarray
        Полученная в ходе процесса оптимизации наилучшая оценка вектора DoA.
    """
    best_angles, best_fun = None, np.inf

    Sigma_XS_t = torch.tensor(Sigma_XS_np, dtype=torch.complex128)
    P_t = torch.tensor(P_np, dtype=torch.complex128)
    Q_inv_sqrt_t = torch.tensor(Q_inv_sqrt_np, dtype=torch.complex128)
    coords_t = torch.tensor(coords_np, dtype=torch.float64)


    for i in range(num_of_starts):
        if i == 0:
            est_angles, est_fun = find_angles_start(Sigma_XS_t, angles0_np,
                                                    coords_t, P_t, Q_inv_sqrt_t,
                                                    bounds=bounds)
        else:
            K = len(angles0_np)
            nu = np.random.RandomState(42+i).uniform(-np.pi, np.pi)
            angles = np.array([(nu + j * 2 * np.pi/K) % (2 * np.pi) 
                              for j in range(K)]) - np.pi
            est_angles, est_fun = find_angles_start(Sigma_XS_np, angles, 
                                                    coords_t, P_np, Q_inv_sqrt_np,
                                                    bounds=bounds)
        if est_fun < best_fun:
            best_fun, best_angles = est_fun, est_angles
    return best_angles