import numpy as np
import torch

from . import diff_sensor_structures as dss

def cost(u: torch.Tensor,
         Sigma_XS: torch.Tensor,
         Sigma_SS: torch.Tensor,
         Q_inv: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет значение целевой функции, минимизация 
    которой позволяет найти оптимальные угловые координаты. 
    """
    A = dss.A_ULA_torch(Q_inv.shape[0], u)
    operand1 = -2 * torch.real(torch.trace(torch.conj(A).T @ Q_inv @ Sigma_XS))
    operand2 = torch.trace(Q_inv @ A @ Sigma_SS @ torch.conj(A).T)
    return torch.real(operand1 + operand2)


def mm_step_u(u: torch.Tensor,
              Sigma_XS: torch.Tensor,
              P: torch.Tensor,
              Q_inv: torch.Tensor,
              step_init: float = 1.0,
              beta: float = 0.5,
              max_ls: int = 20) -> torch.Tensor:
    """
    Один MM / GEM шаг для минимизации целевой функцией
    с гарантией ее невозрастания.
    """
    u = u.detach().clone().requires_grad_(True)
    f0 = cost(u, Sigma_XS, P, Q_inv)
    f0.backward()
    grad = u.grad.detach()

    # Квази-Ньютоновская диагональная аппроксимация
    H_diag = torch.clamp(grad.abs(), min=1e-6)
    direction = -grad / H_diag

    # Backtracking line search
    step = step_init
    with torch.no_grad():
        for _ in range(max_ls):
            u_new = torch.clamp(u + step * direction, -1.0, 1.0)
            if cost(u_new, Sigma_XS, P, Q_inv) <= f0:
                return u_new
            step *= beta

    # Если backtracking не сработал — EM допускает "no-op" шаг
    return u.detach()


def optimize_u_mm(Sigma_XS: torch.Tensor,
                  u0: torch.Tensor,
                  P: torch.Tensor,
                  Q_inv_sqrt: torch.Tensor,
                  max_iter: int = 50,
                  tol: float = 1e-6) -> torch.Tensor:
    """
    GEM/MМ-оптимизация по u с гарантией невозрастания cost.
    """
    u = u0.clone()
    prev_val = cost(u, Sigma_XS, P, Q_inv_sqrt)

    for _ in range(max_iter):
        u_new = mm_step_u(u, Sigma_XS, P, Q_inv_sqrt)
        new_val = cost(u_new, Sigma_XS, P, Q_inv_sqrt)

        if torch.abs(prev_val - new_val) < tol:
            break

        u, prev_val = u_new, new_val

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

    u0 = torch.tensor(np.sin(theta0_np), dtype=torch.float64)
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
            )

        u_hat = optimize_u_mm(Sigma_XS, u_start, Sigma_SS, Q_inv)
        val = cost(u_hat, Sigma_XS, Sigma_SS, Q_inv)
        old_val = cost(u_start, Sigma_XS, Sigma_SS, Q_inv)
        if val - old_val > 0:
            raise ValueError('Cost function increases!')

        if val < best_val:
            best_val, best_u = val, u_hat
    

    return np.arcsin(best_u.numpy())