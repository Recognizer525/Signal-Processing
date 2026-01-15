import numpy as np
import torch

import diff_sensor_structures as dss

DIST_RATIO = 0.5

def cost_u(u: torch.Tensor,
           Sigma_XS: torch.Tensor, 
           P: torch.Tensor, 
           Q_inv_sqrt: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет значение фробениусовой нормы ||Q^{-1/2}(Sigma_XS-AP)||^2_F, 
    которая подлежит минимизации.

    Parameters
    ---------------------------------------------------------------------------
    u: torch.Tensor
        Оценка направляющих косинусов (sin(DoA)).
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
    A = dss.A_ULA_torch(Q_inv_sqrt.shape[0], u)
    E = torch.matmul(Q_inv_sqrt, Sigma_XS - torch.matmul(A, P))  
    return torch.norm(E, 'fro')**2


def find_u_start(Sigma_XS: torch.Tensor, 
                 u0_np: np.ndarray, 
                 P: torch.Tensor, 
                 Q_inv_sqrt: torch.Tensor,
                 tol: float = 1e-6,
                 max_iter: int = 500,
                 alpha0: float = 1.0,
                 beta: float = 0.5,
                 c: float = 1e-4) -> tuple[np.ndarray, float]:
    """
    Запускает оптимизацию нормы ||Q^{-1/2}(Sigma_XS - AP)||^2_F по вектору DoA
    с использованием градиентного спуска с backtracking line search.

    Гарантируется невозрастание минимизируемой функции.
    Ограничение u ∈ [-1, 1]^K обеспечивается проекцией.

    Parameters
    ---------------------------------------------------------------------------
    Sigma_XS: torch.Tensor
        Кросс-ковариация наблюдений и сигналов с текущего Е-шага.
    u0_np: np.ndarray
        Начальное приближение для sin(DoA).
    P: torch.Tensor
        Оценка ковариации сигналов.
    Q_inv_sqrt: torch.Tensor
        Квадратный корень от обратной ковариации шума.
    tol: float
        Порог остановки по норме градиента.
    max_iter: int
        Максимальное число итераций.
    alpha0: float
        Начальный шаг градиентного спуска.
    beta: float
        Коэффициент уменьшения шага (0 < beta < 1).
    c: float
        Параметр условия Армихо.

    Returns
    ---------------------------------------------------------------------------
    u_np: np.ndarray
        Оценка sin(DoA).
    loss_val: float
        Значение минимизируемой функции.
    """

    u = torch.tensor(u0_np, dtype=torch.float64, requires_grad=True)

    for _ in range(max_iter):
        # Вычисление функции и градиента
        loss = cost_u(u, Sigma_XS, P, Q_inv_sqrt)
        loss.backward()

        grad = u.grad
        grad_norm_sq = torch.sum(grad**2)

        # Критерий остановки
        if torch.sqrt(grad_norm_sq) < tol:
            break

        alpha = alpha0

        # Backtracking line search (Armijo)
        with torch.no_grad():
            while True:
                u_new = u - alpha * grad
                u_new = torch.clamp(u_new, -1.0, 1.0)

                new_loss = cost_u(u_new, Sigma_XS, P, Q_inv_sqrt)

                if new_loss <= loss - c * alpha * grad_norm_sq:
                    break

                alpha *= beta

        # Обновление
        with torch.no_grad():
            u[:] = u_new

        u.grad.zero_()

    return u.detach().numpy(), loss.item()


def find_angles(Sigma_XS_np: np.ndarray, 
                theta0_np: np.ndarray, 
                P_np: np.ndarray, 
                Q_inv_sqrt_np: np.ndarray, 
                num_of_starts: int = 15) -> np.ndarray:
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

    Returns
    ---------------------------------------------------------------------------
    best_theta: np.ndarray
        Полученная в ходе процесса оптимизации наилучшая оценка вектора DoA.
    """
    best_u, best_fun_val = None, np.inf

    Sigma_XS_t = torch.tensor(Sigma_XS_np, dtype=torch.complex128)
    P_t = torch.tensor(P_np, dtype=torch.complex128)
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
        
        
        est_u, fun_val = find_u_start(Sigma_XS_t, u_start, P_t, Q_inv_sqrt_t)
        if fun_val < best_fun_val:
            best_fun_val, best_u = fun_val, est_u
    
    theta_hat = np.arcsin(np.clip(best_u, -1.0, 1.0))

    return theta_hat