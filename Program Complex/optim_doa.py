import numpy as np
import torch
from scipy.optimize import minimize

DIST_RATIO = 0.5

def A_ULA_torch(L: int, theta: torch.Tensor) -> torch.Tensor:
    """
    Создает матрицу векторов направленности для массива сенсоров типа ULA.

    Parameters
    ---------------------------------------------------------------------------
    L: int
        Число сенсоров.
    theta: torch.Tensor
        Оценка DoA.
    
    Returns
    ---------------------------------------------------------------------------
    A: torch.Tensor
        Матрица векторов направленности.
    """
    device = theta.device
    sensor_indices = torch.arange(L, device=device).reshape(-1, 1).float() 
    return torch.exp(-2j * torch.pi * DIST_RATIO * 
                     sensor_indices * torch.sin(theta))


def cost_theta_torch(theta: torch.Tensor,
                     X: torch.Tensor, 
                     S: torch.Tensor, 
                     Q_inv_sqrt: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет значение фробениусовой нормы ||Q^{-1/2}(X-AS)||^2_F, 
    которая подлежит минимизации.

    Parameters
    ---------------------------------------------------------------------------
    theta: torch.Tensor
        Оценка DoA.
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
    A = A_ULA_torch(X.shape[0], theta)
    E = torch.matmul(Q_inv_sqrt, X - torch.matmul(A, S))  
    return torch.norm(E, 'fro')**2  # скалярный тензор


def CM_step_theta_start(X_np: np.ndarray, 
                        theta0_np: np.ndarray, 
                        S_np: np.ndarray, 
                        Q_inv_sqrt_np: np.ndarray, 
                        method: str = 'SLSQP', 
                        tol: int = 1e-6) -> tuple[np.ndarray, float]:
    """
    Запускает оптимизацию нормы ||Q^{-1/2}(X-AS)||^2_F по вектору DoA.

    Parameters
    ---------------------------------------------------------------------------
    X_np: np.ndarray
        Двумерный массив, соответствующий наблюдениям 
        (с учетом оценок пропущенных значений).
    theta0_np: np.ndarray
        Первое приближение для оценивания DoA. 
        Представляет собой одномерный массив.
    S_np: np.ndarray
        Текущая оценка детерминированных исходных сигналов.
    Q_inv_sqrt_np: np.ndarray
        Квадратный корень от матрицы, обратной к ковариационной матрице шума. 
    method: str = 'SLSQP'
        Метод оптимизации.
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
    def fun(theta_np: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Возвращает значение функции потерь и значение градиента.
        """
        theta_t = torch.tensor(theta_np, 
                               dtype=torch.float32, 
                               requires_grad=True)
        X_t = torch.tensor(X_np, dtype=torch.cfloat)
        S_t = torch.tensor(S_np, dtype=torch.cfloat)
        Q_inv_sqrt_t = torch.tensor(Q_inv_sqrt_np, dtype=torch.cfloat)
        
        loss = cost_theta_torch(theta_t, X_t, S_t, Q_inv_sqrt_t)
        loss.backward()
        grad = theta_t.grad.detach().numpy().astype(np.float64)
        return loss.item(), grad

    res = minimize(lambda th: fun(th)[0], theta0_np, 
                   jac=lambda th: fun(th)[1], method=method, tol=tol)
    #print(f"Optim.res={res.success}")
    return res.x, res.fun


def CM_step_theta(X_np: np.ndarray, 
                  theta0_np: np.ndarray, 
                  S_np: np.ndarray, 
                  Q_inv_sqrt_np: np.ndarray, 
                  num_of_starts: int = 5) -> np.ndarray:
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

    Returns
    ---------------------------------------------------------------------------
    best_theta: np.ndarray
        Полученная в ходе процесса оптимизации наилучшая оценка вектора DoA.
    """
    best_theta, best_fun = None, np.inf
    for i in range(num_of_starts):
        if i == 0:
            est_theta, est_fun = CM_step_theta_start(X_np, theta0_np, 
                                                     S_np, Q_inv_sqrt_np)
        else:
            M = len(theta0_np)
            nu = np.random.RandomState(42+i).uniform(-np.pi, np.pi)
            theta = np.array([(nu + j * 2 * np.pi/M) % (2 * np.pi) 
                              for j in range(M)]) - np.pi
            est_theta, est_fun = CM_step_theta_start(X_np, theta, 
                                                     S_np, Q_inv_sqrt_np)
        if est_fun < best_fun:
            best_fun, best_theta = est_fun, est_theta
    return best_theta