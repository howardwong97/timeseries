from typing import Optional, Union

import numpy as np
import pandas as pd

__all__ = ["ensure1d", "ensure2d"]


def ensure1d(
    x: Union[np.ndarray, pd.Series, pd.DataFrame],
    name: Optional[str] = None,
    series: bool = False,
) -> Union[np.ndarray, pd.Series]:
    if isinstance(x, pd.Series):
        if not series:
            return np.asarray(x)
        if not isinstance(x.name, str):
            x.name = str(x.name)
        return x

    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"{name} must be squeezable to 1D")
        if not series:
            return np.asarray(x.iloc[:, 0])
        x_series = pd.Series(x.iloc[:, 0], x.index)
        if not isinstance(x_series.name, str):
            x_series.name = str(x_series.name)
        return x_series

    x_arr = np.squeeze(np.asarray(x))
    if x_arr.ndim == 0:
        x_arr = x_arr[None]
    elif x_arr.ndim != 1:
        raise ValueError(f"{name} must be squeezable to 1D")

    return pd.Series(x_arr, name=name) if series else x_arr


def ensure2d(
    x: Union[np.ndarray, pd.Series, pd.DataFrame]
) -> Union[np.ndarray, pd.DataFrame]:
    if isinstance(x, pd.Series):
        return pd.DataFrame(x)
    elif isinstance(x, pd.DataFrame):
        return x
    elif isinstance(x, np.ndarray):
        if x.ndim == 0:
            return np.asarray([[x]])
        elif x.ndim == 1:
            return x[:, None]
        elif x.ndim == 2:
            return x
        else:
            raise ValueError("Input must be 2D or reshape-able to 2D")
    else:
        raise TypeError("Input must be a Series, DataFrame or ndarray")
