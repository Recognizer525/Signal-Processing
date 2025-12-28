import numpy as np
import torch
from scipy.optimize import minimize

import diff_sensor_structures as dss

DIST_RATIO = 0.5

def cost_u(u: torch.Tensor,
           X: torch.Tensor, 
           S: torch.Tensor, 
           Q_inv_sqrt: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет значение фробениусовой нормы ||Q^{-1/2}(X-AS)||^2_F, 
    которая подлежит минимизации.

    Parameters
    ---------------------------------------------------------------------------
    u: torch.Tensor
        Оценка направляющих косинусов (sin(DoA)).
    X: torch.Tensor
        Двумерный массив, соответствующий наблюдениям 
        (с учетом оценок пропущенных значений).
    S: torch.Tensor
        Текущая оценка детерминированных исходных сигналов.
    Q_inv_sqrt: torch.Tensor
        Квадратный корень от матрицы, обратной к ковариационной матрице шума.

    Returns
    ---------------------------------------------------------------------------
    res: torch.Tensor
        Значение фробениусовой нормы ||Q^{-1/2}(X-AS)||^2_F.
    """
    A = dss.A_ULA_u_torch(X.shape[0], u)
    E = torch.matmul(Q_inv_sqrt, X - torch.matmul(A, S))  
    return torch.norm(E, 'fro')**2  # скалярный тензор


def find_u_start(X: torch.Tensor, 
                 u0_np: np.ndarray, 
                 S: torch.Tensor, 
                 Q_inv_sqrt: torch.Tensor, 
                 method: str = 'SLSQP', 
                 tol: int = 1e-6) -> tuple[np.ndarray, float]:
    """
    Запускает оптимизацию нормы ||Q^{-1/2}(X-AS)||^2_F по вектору DoA.

    Parameters
    ---------------------------------------------------------------------------
    X: torch.Tensor
        Двумерный массив, соответствующий наблюдениям 
        (с учетом оценок пропущенных значений).
    u0_np: np.ndarray
        Первое приближение для оценивания sin(DoA). 
        Представляет собой одномерный массив.
    S: torch.Tensor
        Текущая оценка детерминированных исходных сигналов.
    Q_inv_sqrt: torch.Tensor
        Квадратный корень от матрицы, обратной к ковариационной матрице шума. 
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
    def fun_and_grad(u_np: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Возвращает значение функции потерь и значение градиента.
        """
        u_t = torch.tensor(u_np, dtype=torch.float64, requires_grad=True)
        
        loss = cost_u(u_t, X, S, Q_inv_sqrt)
        loss.backward()
        grad = u_t.grad.detach().numpy()
        return loss.item(), grad

    res = minimize(fun_and_grad, u0_np, 
                   jac=True, method=method, tol=tol)
    #print(f"Optim.res={res.success}")
    return res.x, res.fun


def CM_step_theta(X_np: np.ndarray, 
                  theta0_np: np.ndarray, 
                  S_np: np.ndarray, 
                  Q_inv_sqrt_np: np.ndarray, 
                  num_of_starts: int = 7) -> np.ndarray:
    """
    Функция предназначена для поиска оценки DoA, которая минимизирует норму
    ||Q^{-1/2}(X-AS)||^2_F.

    Parameters
    ---------------------------------------------------------------------------
    X: np.ndarray
        Двумерный массив, соответствующий наблюдениям 
        (с учетом оценок пропущенных значений).
    theta0_np: np.ndarray
        Оценка DoA, полученная на предыдущей итерации,
        либо же начальная оценка DoA.
    S_np: np.ndarray
        Текущая оценка детерминированных исходных сигналов.
    Q_inv_sqrt_np: np.ndarray
        Квадратный корень от матрицы, обратной к ковариационной матрице шума.
    num_of_starts: int
        Число начальных приближений относительно которых нужно проводить
        оптимизационный процесс.
    method: str
        Метод оптимизации функции потерь для DoA.

    Returns
    ---------------------------------------------------------------------------
    best_theta: np.ndarray
        Полученная в ходе процесса оптимизации наилучшая оценка вектора DoA.
    """
    best_u, best_fun_val = None, np.inf

    X_t = torch.tensor(X_np, dtype=torch.complex128)
    S_t = torch.tensor(S_np, dtype=torch.complex128)
    Q_inv_sqrt_t = torch.tensor(Q_inv_sqrt_np, dtype=torch.complex128)

    K = len(theta0_np)
    u0_np = np.sin(theta0_np)

    for i in range(num_of_starts):
        if i == 0:
            u_start = u0_np
        elif i < num_of_starts - 1:
            u_start = u0_np + 0.2 * np.random.randn(K)
        else:
            u_start = np.random.uniform(-1.0, 1.0, size=K)

        # Гарантирует корректность начальной точки старта
        u_start = np.clip(u_start, -1.0, 1.0)

        est_u, fun_val = find_u_start(X_t, u_start, S_t, Q_inv_sqrt_t)
        if fun_val < best_fun_val:
            best_fun_val, best_u = fun_val, est_u
    
    theta_hat = np.arcsin(np.clip(best_u, -1.0, 1.0))

    return theta_hat