import numpy as np
import torch

import diff_sensor_structures as dss

DIST_RATIO = 0.5

def cost_u(u: torch.Tensor,
           Sigma_XS: torch.Tensor, 
           P: torch.Tensor, 
           Q_inv_sqrt: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет значение фробениусовой нормы ||Q^{-1/2}(Sigma_XS-AP)||^2_F.
    """
    A = dss.A_ULA_torch(Q_inv_sqrt.shape[0], u)
    E = Q_inv_sqrt @ (Sigma_XS - A @ P)
    return torch.norm(E, 'fro') ** 2


def mm_step_u(u: torch.Tensor,
              Sigma_XS: torch.Tensor,
              P: torch.Tensor,
              Q_inv_sqrt: torch.Tensor,
              step_init: float = 1.0,
              beta: float = 0.5,
              max_ls: int = 20) -> torch.Tensor:
    """
    Один MM / GEM шаг для минимизации ||Q^{-1/2}(Sigma_XS-AP)||^2_F
    с гарантией невозрастания функционала.
    """
    u = u.detach().clone().requires_grad_(True)
    f0 = cost_u(u, Sigma_XS, P, Q_inv_sqrt)
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
            if cost_u(u_new, Sigma_XS, P, Q_inv_sqrt) <= f0:
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
    prev_val = cost_u(u, Sigma_XS, P, Q_inv_sqrt)

    for _ in range(max_iter):
        u_new = mm_step_u(u, Sigma_XS, P, Q_inv_sqrt)
        new_val = cost_u(u_new, Sigma_XS, P, Q_inv_sqrt)

        if torch.abs(prev_val - new_val) < tol:
            break

        u, prev_val = u_new, new_val

    return u


def find_angles(Sigma_XS_np: np.ndarray, 
                theta0_np: np.ndarray, 
                P_np: np.ndarray, 
                Q_inv_sqrt_np: np.ndarray, 
                num_of_starts: int = 10,
                base_seed: int = 42) -> np.ndarray:
    """
    Поиск DoA с использованием GEM/MM-алгоритма
    с гарантией неубывания EM-функционала.
    """
    Sigma_XS = torch.tensor(Sigma_XS_np, dtype=torch.complex128)
    P = torch.tensor(P_np, dtype=torch.complex128)
    Q_inv_sqrt = torch.tensor(Q_inv_sqrt_np, dtype=torch.complex128)

    u0 = torch.tensor(np.sin(theta0_np), dtype=torch.float64)
    best_u, best_val = u0, cost_u(u0, Sigma_XS, P, Q_inv_sqrt)

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

        u_hat = optimize_u_mm(Sigma_XS, u_start, P, Q_inv_sqrt)
        val = cost_u(u_hat, Sigma_XS, P, Q_inv_sqrt)
        old_val = cost_u(u_start, Sigma_XS, P, Q_inv_sqrt)

        if val < best_val:
            best_val, best_u = val, u_hat
    if best_val - old_val > 0:
        raise ValueError('Cost function increases!')

    return np.arcsin(best_u.numpy())