import numpy as np
import torch
from scipy.optimize import minimize

DIST_RATIO = 0.5

def A_ULA_torch(L, theta):
    """
    Создает матрицу векторов направленности для массива сенсоров типа ULA.

    Параметры:
    L - число сенсоров,
    theta - тензор углов прибытия (размер [n_angles])
    """
    device = theta.device
    sensor_indices = torch.arange(L, device=device).reshape(-1, 1).float()  # (L,1)
    return torch.exp(-2j * torch.pi * DIST_RATIO * sensor_indices * torch.sin(theta))  # (L, n_angles)


def cost_theta_torch(theta, X, S, Q_inv_sqrt):
    """
    theta - тензор углов прибытия (requires_grad=True)
    X, S, Q_inv_sqrt - тоже тензоры PyTorch, dtype=torch.cfloat или torch.float
    """
    A = A_ULA_torch(X.shape[0], theta)  # (L, n_angles)
    E = torch.matmul(Q_inv_sqrt, X - torch.matmul(A, S))  
    return torch.norm(E, 'fro')**2  # скалярный тензор


def CM_step_theta_start(X_np: np.ndarray, 
                        theta0_np: np.ndarray, 
                        S_np: np.ndarray, 
                        Q_inv_sqrt_np: np.ndarray, 
                        method: str = 'SLSQP', 
                        tol: int = 1e-6):
    """
    X_np, theta0_np, S_np, Q_inv_sqrt_np - numpy массивы
    """
    # Объявляем функцию для scipy, которая принимает numpy theta, внутри переводим в torch и вычисляем
    def fun(theta_np):
        theta_t = torch.tensor(theta_np, dtype=torch.float32, requires_grad=True)
        X_t = torch.tensor(X_np, dtype=torch.cfloat)
        S_t = torch.tensor(S_np, dtype=torch.cfloat)
        Q_inv_sqrt_t = torch.tensor(Q_inv_sqrt_np, dtype=torch.cfloat)
        
        loss = cost_theta_torch(theta_t, X_t, S_t, Q_inv_sqrt_t)
        loss.backward()
        grad = theta_t.grad.detach().numpy().astype(np.float64)
        return loss.item(), grad

    res = minimize(lambda th: fun(th)[0], theta0_np, jac=lambda th: fun(th)[1], method=method, tol=tol)
    #print(f"Optim.res={res.success}")
    return res.x, res.fun


def CM_step_theta(X_np: np.ndarray, 
                  theta0_np: np.ndarray, 
                  S_np: np.ndarray, 
                  Q_inv_sqrt_np: np.ndarray, 
                  num_of_starts: int = 5):
    best_theta, best_fun = None, np.inf
    for i in range(num_of_starts):
        if i == 0:
            est_theta, est_fun = CM_step_theta_start(X_np, theta0_np, S_np, Q_inv_sqrt_np)
        else:
            M = len(theta0_np)
            nu = np.random.RandomState(42+i).uniform(-np.pi, np.pi)
            theta = np.array([(nu + j * 2 * np.pi/M)%(2 * np.pi) for j in range(M)]) - np.pi
            est_theta, est_fun = CM_step_theta_start(X_np, theta, S_np, Q_inv_sqrt_np)
        if est_fun < best_fun:
            best_fun, best_theta = est_fun, est_theta
    return best_theta