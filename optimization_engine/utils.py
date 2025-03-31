"""Contains, well..., utils, that are utilised by other functions. Specific operations on numpy ndarrays."""

import numpy as np
from typing import Literal, Any


def symmetrize_numpy_matrix(matrix: np.ndarray, copied_side: Literal["upper", "lower"] = "upper") -> np.ndarray | None:
    """
    this method takes one matrix and returns symmetric matrix - when copied_side is 'upper' then upper side is
    copied to the lower side, analogically with copied_side = 'lower'. If provided array has more than 2 dimensions
    the operation is applied to the final two axes. If provided array has less than 2D None is returned
    """
    if matrix.ndim < 2:
        raise ValueError(f"An array to symmetrize needs to have at least 2 dimensions, got {matrix.ndim} instead")
    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(
            f"Final two dimensions need to have the same size, got ({matrix.shape[-1]}, {matrix.shape[-2]}) instead")
    if copied_side == "upper":
        return np.triu(matrix, k=1).swapaxes(-1, -2) + np.triu(matrix, k=0)
    if copied_side == "lower":
        return np.tril(matrix, k=-1).swapaxes(-1, -2) + np.tril(matrix, k=0)

def random_choose_2_arrays(a: np.ndarray, b: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
    """Takes two ndarrays of the same shape and returns another ndarray of the same shape that has
    its elements randomly chosen from `a` and `b`"""
    if a.shape != b.shape:
        raise ValueError(f"Random choose from 2 arrays expected the arrays to have the same shape, got {a.shape} and {b.shape} instead")
    rng = rng if rng is not None else np.random.default_rng()
    child_indexes = rng.integers(0, 1, size=a.shape, endpoint=True)
    child = np.where(child_indexes == 0, a, b)
    return child