import numpy as np
import torch
from scipy.optimize import minimize

import diff_sensor_structures as dss

DIST_RATIO = 0.5

def cost_theta(theta: torch.Tensor,
               Sigma_XS: torch.Tensor, 
               P: torch.Tensor, 
               Q_inv_sqrt: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет значение фробениусовой нормы ||Q^{-1/2}(Sigma_XS-AP)||^2_F, 
    которая подлежит минимизации.

    Parameters
    ---------------------------------------------------------------------------
    theta: torch.Tensor
        Оценка DoA.
    Sigma_XS: torch.Tensor
        Кросс-ковариация наблюдений и сигналов с текущего Е-шага.
    P: torch.Tensor
        Оценка ковариации сигналов с предыдущей итерации.
    Q_inv_sqrt: torch.Tensor
        Квадратный корень от матрицы, обратной к ковариационной матрице шума.

    Returns
    ---------------------------------------------------------------------------
    res: torch.Tensor
        Значение фробениусовой нормы ||Q^{-1/2}(Sigma_XS-AP)||^2_F.
    """
    A = dss.A_ULA_torch(Q_inv_sqrt.shape[0], theta)
    E = torch.matmul(Q_inv_sqrt, Sigma_XS - torch.matmul(A, P))  
    return torch.norm(E, 'fro')**2  # скалярный тензор


def find_angles_start(Sigma_XS: torch.Tensor, 
                      theta0_np: np.ndarray, 
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
        Кросс-ковариация наблюдений и сигналов с текущего Е-шага.
    theta0_np: np.ndarray
        Первое приближение для оценивания DoA. 
        Представляет собой одномерный массив.
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
    if bounds is None:
        bounds = [(-np.pi/2, np.pi/2)] * len(theta0_np)
    def fun_and_grad(theta_np: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Возвращает значение функции потерь и значение градиента.
        """
        theta_t = torch.tensor(theta_np, 
                               dtype=torch.float64, 
                               requires_grad=True)
        
        loss = cost_theta(theta_t, Sigma_XS, P, Q_inv_sqrt)
        loss.backward()
        grad = theta_t.grad.detach().numpy()
        return loss.item(), grad

    res = minimize(fun_and_grad, theta0_np, 
                   jac=True, bounds=bounds, method=method, tol=tol)
    #print(f"Optim.res={res.success}")
    return res.x, res.fun


def find_angles(Sigma_XS_np: np.ndarray, 
                theta0_np: np.ndarray, 
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
        Кросс-ковариация наблюдений и сигналов с текущего Е-шага.
    theta0_np: np.ndarray
        Оценка DoA, полученная на предыдущей итерации,
        либо же начальная оценка DoA.
    P_np: np.ndarray
        Оценка ковариации сигналов с предыдущей итерации.
    Q_inv_sqrt_np: np.ndarray
        Квадратный корень от матрицы, обратной к ковариационной матрице шума.
    num_of_starts: int
        Число начальных приближений относительно которых нужно проводить
        оптимизационный процесс.
    bounds: list|None
        Границы, в пределах которых надо искать оптимальное значение вектора DoA.

    Returns
    ---------------------------------------------------------------------------
    best_theta: np.ndarray
        Полученная в ходе процесса оптимизации наилучшая оценка вектора DoA.
    """
    best_theta, best_fun = None, np.inf

    Sigma_XS_t = torch.tensor(Sigma_XS_np, dtype=torch.complex128)
    P_t = torch.tensor(P_np, dtype=torch.complex128)
    Q_inv_sqrt_t = torch.tensor(Q_inv_sqrt_np, dtype=torch.complex128)


    for i in range(num_of_starts):
        if i == 0:
            est_theta, est_fun = find_angles_start(Sigma_XS_t, theta0_np, 
                                                   P_t, Q_inv_sqrt_t,
                                                   bounds=bounds)
        else:
            K = len(theta0_np)
            nu = np.random.RandomState(42+i).uniform(-np.pi, np.pi)
            theta = np.array([(nu + j * 2 * np.pi/K) % (2 * np.pi) 
                              for j in range(K)]) - np.pi
            est_theta, est_fun = find_angles_start(Sigma_XS_t, theta, 
                                                   P_t, Q_inv_sqrt_t,
                                                   bounds=bounds)
        if est_fun < best_fun:
            best_fun, best_theta = est_fun, est_theta
    return best_theta