from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np

__all__ = [
    "combine_constraints",
    "format_constraints",
    "combine_and_format_constraints",
]


def combine_and_format_constraints(
    constraints: Iterable[Tuple[np.ndarray, np.ndarray]], offsets: np.ndarray
) -> Iterable[Dict[str, object]]:
    a, b = combine_constraints(constraints, offsets)
    constraints_dict = format_constraints(a, b)

    return constraints_dict


def combine_constraints(
    constraints: Iterable[Tuple[np.ndarray, np.ndarray]], offsets: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    num_cons = []
    for c in constraints:
        assert c is not None
        num_cons.append(c[0].shape[0])

    num_constraints = np.array(num_cons, dtype=int)
    num_params = offsets.sum()
    a = np.zeros((int(num_constraints.sum()), int(num_params)))
    b = np.zeros(int(num_constraints.sum()))

    for i, c in enumerate(constraints):
        r_en = num_constraints[: i + 1].sum()
        c_en = offsets[: i + 1].sum()
        r_st = r_en - num_constraints[i]
        c_st = c_en - offsets[i]
        if r_en - r_st > 0:
            a[r_st:r_en, c_st:c_en] = c[0]
            b[r_st:r_en] = c[1]

    return a, b


def format_constraints(a: np.ndarray, b: np.ndarray) -> List[Dict[str, object]]:
    """
    Generate constraints from arrays

    Parameters
    ----------
    a : ndarray
        Parameter loadings
    b : ndarray
        Constraint bounds

    Returns
    -------
    constraints : list of dicts
        Dictionary of inequality constraints, one for each row of `a`

    Notes
    -----
    Parameter constraints satisfy a.dot(parameters) - b >= 0
    """

    def factory(coef: np.ndarray, val: float) -> Callable[..., float]:
        def f(params: np.ndarray, *_: Any) -> float:
            return np.dot(coef, params) - val

        return f

    constraints = []
    for i in range(a.shape[0]):
        con = {"type": "ineq", "fun": factory(a[i], float(b[i]))}
        constraints.append(con)

    return constraints
