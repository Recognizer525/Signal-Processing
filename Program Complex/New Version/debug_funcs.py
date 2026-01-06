import numpy as np

import sensors

def is_valid_result(data: np.ndarray,
                    data_name: str,
                    check_psd: bool = False)-> None:
    """
    Проверяет пригодность данных. Проверяет наличие пропусков,
    бесконечных значений, PSD-свойство. печатает форму данных.
    """
    if np.isnan(data).any() or np.isinf(data).any():
        print('Infs or/and NaNs in {data_name}')
    print(f"{data_name}.shape={data.shape}")

    if check_psd and data.ndim == 2:
        if not sensors.is_psd(data):
            print(f"{data_name} is not psd")

    if check_psd and data.ndim == 3:
        for i in range(data.shape[0]):
            if not sensors.is_psd(data[i]):
                print(f"{data_name}[{i}] is not psd")
