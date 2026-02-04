import numpy as np
import torch

from . import diff_sensor_structures as dss

DIST_RATIO = 0.5


def cost(u: torch.Tensor,
         Sigma_XS: torch.Tensor,
         Sigma_SS: torch.Tensor,
         Q_inv: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет значение целевой функции, минимизация 
    которой позволяет найти оптимальные нулевые координаты. 
    """
    A = dss.A_ULA_torch(Q_inv.shape[0], u)
    operand1 = -2 * torch.real(torch.trace(torch.conj(A).T @ Q_inv @ Sigma_XS))
    operand2 = torch.trace(Q_inv @ A @ Sigma_SS @ torch.conj(A).T)
    return torch.real(operand1 + operand2)


def gradient_descent_backtracking(u, Sigma_XS, Sigma_SS, Q_inv,
                                  alpha0=1.0, beta=0.5, c=1e-4, max_iters=100, max_ls=25):
    for _ in range(max_iters):
        # Сброс градиента
        if u.grad is not None:
            u.grad.zero_()
        
        # Вычисляем значение функции
        L = cost(u, Sigma_XS, Sigma_SS, Q_inv)
        
        # Вычисляем градиент
        L.backward()
        grad_u = u.grad.clone()
        
        # Backtracking line search
        alpha = alpha0
        ls_iter = 0
        while ls_iter < max_ls:
            u_new = u - alpha * grad_u
            L_new = cost(u_new, Sigma_XS, Sigma_SS, Q_inv)
            
            # Условие Armijo
            if L_new <= L - c * alpha * torch.norm(grad_u)**2:
                break
            alpha *= beta
            ls_iter += 1
        
        # Обновляем переменную
        with torch.no_grad():
            u -= alpha * grad_u
        
        # Проверка сходимости
        if torch.norm(grad_u) < 1e-6:
            break
            
    return u


def find_angles(Sigma_XS_np: np.ndarray, 
                theta0_np: np.ndarray, 
                Sigma_SS_np: np.ndarray, 
                Q_inv_np: np.ndarray, 
                num_of_starts: int = 10,
                base_seed: int = 42) -> np.ndarray:
    """
    Поиск DoA с использованием GEM/MM-алгоритма
    с гарантией неубывания EM-функционала.
    """
    Sigma_XS = torch.tensor(Sigma_XS_np, dtype=torch.complex128)
    Sigma_SS = torch.tensor(Sigma_SS_np, dtype=torch.complex128)
    Q_inv = torch.tensor(Q_inv_np, dtype=torch.complex128)

    u0 = torch.tensor(np.sin(theta0_np), dtype=torch.float64, requires_grad=True)
    best_u, best_val = u0, cost(u0, Sigma_XS, Sigma_SS, Q_inv)

    for i in range(num_of_starts):
        if i == 0:
            u_start = u0
        else:
            gen = torch.Generator(device=u0.device)
            gen.manual_seed(base_seed + i)

            noise = torch.randn(
                u0.shape,
                dtype=u0.dtype,
                device=u0.device,
                generator=gen
                )
            
            u_start = torch.clamp(
                u0 + 0.2 * noise,
                -1.0, 1.0
            ).detach().clone().requires_grad_(True)

        u_hat = gradient_descent_backtracking(u_start, Sigma_XS, Sigma_SS, Q_inv)
        val = cost(u_hat, Sigma_XS, Sigma_SS, Q_inv)
        old_val = cost(u_start, Sigma_XS, Sigma_SS, Q_inv)

        if val < best_val:
            best_val, best_u = val, u_hat
    if best_val - old_val > 0:
        raise ValueError('Cost function increases!')

    return np.arcsin(best_u.detach().numpy())