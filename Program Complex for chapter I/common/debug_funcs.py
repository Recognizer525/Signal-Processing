import numpy as np

from . import sensors as sn

def is_valid_result(data: np.ndarray,
                    data_name: str,
                    expected_shape: tuple|None = None,
                    check_psd: bool = False)-> None:
    """
    Проверяет пригодность данных. Проверяет наличие пропусков,
    бесконечных значений, PSD-свойство. печатает форму данных.
    """
    if np.isnan(data).any() or np.isinf(data).any():
        print(f"{data_name}.shape={data.shape}")
        raise ValueError(f'Infs or/and NaNs in {data_name}')

    if expected_shape is not None:
        if data.shape != expected_shape:
            raise ValueError(f"{data_name} has wrong shape!")
        pass

    if check_psd and data.ndim == 2:
        if not sn.is_pd(data):
            raise ValueError(f"{data_name} is not pd")

    if check_psd and data.ndim == 3:
        for i in range(data.shape[0]):
            if not sn.is_pd(data[i]):
                print(f"{data_name}[{i}]={data[i]}")
                raise ValueError(f"{data_name}[{i}] is not pd")
