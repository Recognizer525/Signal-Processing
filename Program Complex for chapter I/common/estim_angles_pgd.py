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


def gradient_descent_backtracking(
    u: torch.Tensor,
    Sigma_XS: torch.Tensor,
    Sigma_SS: torch.Tensor,
    Q_inv: torch.Tensor,
    alpha0: float = 1.0,
    beta: float = 0.5,
    max_iters: int = 100,
    max_ls: int = 25,
    grad_tol: float = 1e-6) -> torch.Tensor:

    """
    Проекционный градиетнный спуск для поиска оптимального u.
    """
    
    # Вычисляем начальное значение стоимости
    with torch.no_grad():
        prev_val = cost(u, Sigma_XS, Sigma_SS, Q_inv)

    for _ in range(max_iters):

        # Сброс градиента
        if u.grad is not None:
            u.grad.zero_()

        # Вычисляем значение функции
        L = cost(u, Sigma_XS, Sigma_SS, Q_inv)
        L.backward()

        # Вычисляем градиент
        grad_u = u.grad.detach().clone()

        # Проверка сходимости (проверка стационарной точки)
        if torch.norm(grad_u) < grad_tol:
            break

        # Armijo backtracking line search после проекции
        alpha = alpha0
        u_new = None
        L_new = None

        for _ in range(max_ls):
            # Пробный шаг
            u_tmp = u - alpha * grad_u

            # Проекция на [-1, 1]^k
            u_tmp = torch.clamp(u_tmp, -1.0, 1.0)

            # вычисление функции стоимости в проецированной точке
            with torch.no_grad():
                L_tmp = cost(u_tmp, Sigma_XS, Sigma_SS, Q_inv)

            # Проверка выполнимости невозрастания
            if L_tmp <= prev_val:
                u_new = u_tmp
                L_new = L_tmp
                break

            alpha *= beta

        # Если линейный поиск был неуспешным - прекращаем цикл
        if u_new is None:
            break

        # Копируем полученный результат
        with torch.no_grad():
            u.copy_(u_new)
            prev_val = L_new

    return u



def find_angles(Sigma_XS_np: np.ndarray, 
                theta0_np: np.ndarray, 
                Sigma_SS_np: np.ndarray, 
                Q_inv_np: np.ndarray, 
                num_of_starts: int = 10,
                base_seed: int = 42) -> np.ndarray:
    """
    Поиск DoA с использованием градиентного спуска и backtracking line search.
    """
    Sigma_XS = torch.tensor(Sigma_XS_np, dtype=torch.complex128)
    Sigma_SS = torch.tensor(Sigma_SS_np, dtype=torch.complex128)
    Q_inv = torch.tensor(Q_inv_np, dtype=torch.complex128)

    u0 = torch.tensor(np.sin(theta0_np), dtype=torch.float64, requires_grad=True)
    best_u, best_val = u0, cost(u0, Sigma_XS, Sigma_SS, Q_inv).detach()

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
        with torch.no_grad():
            val = cost(u_hat, Sigma_XS, Sigma_SS, Q_inv)
    

        if val < best_val:
            best_u = u_hat.detach().clone()
            best_val = val.detach()

    return np.arcsin(best_u.detach().numpy())