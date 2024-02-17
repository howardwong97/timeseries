import numpy as np
from timeseries.utils.array import ensure2d


def outer_product2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(ensure2d(x))
    return np.einsum("ij,ik->ijk", x, x)
