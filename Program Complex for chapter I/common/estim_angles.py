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
    operand1 = -2 * torch.real(torch.trace(Q_inv @ A @ Sigma_XS))
    operand2 = torch.trace(Q_inv @ A @ Sigma_SS @ torch.conj(A).T)
    return torch.real(operand1 + operand2)



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
    best_u, best_val = u0, cost(u0, Sigma_XS, P, Q_inv_sqrt)

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
        val = cost(u_hat, Sigma_XS, P, Q_inv_sqrt)
        old_val = cost(u_start, Sigma_XS, P, Q_inv_sqrt)

        if val < best_val:
            best_val, best_u = val, u_hat
    if best_val - old_val > 0:
        raise ValueError('Cost function increases!')

    return np.arcsin(best_u.numpy())