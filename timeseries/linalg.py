import numpy as np
from scipy.linalg import expm, logm, norm

from timeseries.utils.array import ensure1d, ensure2d

__all__ = [
    "outer_product2d",
    "matrix_power3d",
    "corr_transform",
    "corr_inverse_transform",
    "corr_vech",
    "corr_inverse_vech",
]


def outer_product2d(x: np.ndarray) -> np.ndarray:
    """
    Takes a 2D array and computes the outer product for each row to return a 3D array of
    outer product matrices

    Parameters
    ----------
    x : ndarray
        T x K matrix

    Returns
    -------
    xx : ndarray
        T x K x K array of outer product matrices
    """
    x = np.asarray(ensure2d(x))
    return np.einsum("ij,ik->ijk", x, x)


def matrix_power3d(x: np.ndarray, power: float) -> np.ndarray:
    """
    Takes an array of matrices (dimensions T x K x K) and takes the matrix power along
    each row (first axis)

    Parameters
    ----------
    x : ndarray
        T x K x K array of positive semi-definite matrices
    power : float
        Power to raise each matrix to

    Returns
    -------
    result : ndarray
        T x K x K array of matrices raised to power
    """
    if x.ndim != 3 or x.shape[1] != x.shape[2]:
        raise ValueError("Input must be 3D with dimensions of the form T x K x K")

    d, v = np.linalg.eig(x)
    d = np.einsum("ij,jk->ijk", d, np.eye(x.shape[1]))
    d = d ** abs(power)

    result = np.einsum("ijk,ikl,iml->ijm", v, d, v)

    return result if power >= 0 else np.linalg.inv(result)


def corr_transform(corr: np.ndarray) -> np.ndarray:
    """
    Transformation of a correlation matrix to a unique corresponding real vector

    Parameters
    ----------
    corr : ndarray
        Correlation matrix with dimensions K x K

    Returns
    -------
    z : ndarray
        Real vector with length K(K-1)/2
    """
    corr = np.asarray(ensure2d(corr), dtype=np.float64)
    if corr.shape[0] != corr.shape[1]:
        raise ValueError("corr must be a square matrix")
    if not np.array_equal(
        np.diag(corr), np.ones(corr.shape[0], dtype=np.float64)
    ) or not np.array_equal(corr, corr.T):
        raise ValueError("corr must be a correlation matrix")

    z = np.asarray(logm(corr), dtype=np.float64)[np.triu_indices(corr.shape[0], 1)]

    return z


def corr_inverse_transform(
    z: np.ndarray, tol: float = 1e-8, maxiter: int = 500
) -> np.ndarray:
    """
    Transform a real vector with appropriate dimensions to a correlation matrix

    Parameters
    ----------
    z : ndarray
        Real vector with length K(K-1)/2
    tol : float
        Tolerance used to terminate the iterative algorithm
    maxiter : int
        Maximum number of iterations

    Returns
    -------
    corr : ndarray
        Correlation matrix with dimensions K x K
    """
    # Ensure z has the correct number of elements (form of K(K-1)/2)
    z = ensure1d(z, "z", False)
    n = 0.5 * (1 + np.sqrt(1 + 8 * len(z)))
    assert n.is_integer()

    # Create n x n symmetric matrix
    n = int(n)
    a = np.zeros((n, n), dtype=np.float64)
    a[np.triu_indices(n, 1)] = z
    a = a + a.T

    # Read properties of the matrix
    diag_vec = np.diag(a)
    diag_ind = np.diag_indices_from(a)

    # Iterative algorithm to get the proper diagonal vector
    dist = np.sqrt(n)
    iter_num = 0
    while dist > np.sqrt(n) * tol and iter_num < maxiter:
        diag_delta = np.log(np.diag(expm(a)))
        diag_vec = diag_vec - diag_delta
        a[diag_ind] = diag_vec
        dist = norm(diag_delta)
        iter_num += 1

    # Get the unique reciprocal correlation matrix
    corr = np.asarray(expm(a), dtype=np.float64)
    np.fill_diagonal(corr, 1)
    corr = (corr + corr.T) / 2

    return corr


def corr_vech(corr: np.ndarray) -> np.ndarray:
    """
    Transformation of a symmetric matrix into a 1D vector of its lower/upper triangular
    elements

    Parameters
    ----------
    corr : ndarray
        Symmetric matrix (K x K)

    Returns
    -------
    z : ndarray
        Vector of length K(K-1)/2
    """
    corr = np.asarray(ensure2d(corr), dtype=np.float64)
    if corr.shape[0] != corr.shape[1] or not np.array_equal(corr, corr.T):
        raise ValueError("corr must be a 2D symmetric matrix")

    k = corr.shape[0]
    z = corr[~np.triu(np.ones((k, k), dtype=bool))]

    return z


def corr_inverse_vech(z: np.ndarray) -> np.ndarray:
    """
    Transform a vector of lower/upper triangular elements to a symmetric correlation
    matrix

    Parameters
    ----------
    z : ndarray
        Vector of length K(K-1)/2

    Returns
    -------
    corr : ndarray
        Correlation matrix with dimensions K x K
    """
    z = ensure1d(z, "z", False)
    n = (-1 + np.sqrt(1 + 8 * len(z))) / 2 + 1
    assert n.is_integer()

    n = int(n)
    corr = np.zeros((n, n), dtype=np.float64)
    corr[~np.triu(np.ones((n, n), dtype=bool))] = z
    corr = corr + corr.T + np.eye(n)

    return corr
