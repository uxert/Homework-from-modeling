"""
this file contains computational backbone of the whole project - here the given optimization problem is actually
being solved
"""

import numpy as np
from typing import Tuple

def generate_random_city_values(cities_amount:int, max_city_size: np.uint32 = 2**32-1, max_distance: np.uint32 = 2**32-1,
                                seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function generates random city parameters (i.e. distances between cities and their sizes) for the optimization
    problem. A seed can be provided if one has the need for reproducibility. Both max_city_size and max_distance
    should not be bigger than the default2**32 - 1 - any higher values will be truncated to 2**32 - 1
    :return: a tuple: (distances_matrix, city_sizes_vector). Distance matrix is a 2D np ndarray where element with
        index [i][j] is the distance between city i and city j. This matrix by definition is symmetric but unfortunately
        numpy does not provide a built-in method for efficiently storing symmetric matrices so the whole matrix will be
        returned anyway. City sizes vector represents exactly what it's name implies: element with index [i] is the size
        of i-th city.
    """
    rng = np.random.default_rng(seed)

    # distances matrix has to be symmetric by definition, but of course random generation does not provide that
    # symmetry is achieved by taking only upper half of the matrix, transposing it and adding the two pieces together
    # in one of the pieces k=1 so that the 'main' diagonal is taken into the account only once
    distances_matrix: np.ndarray = rng.integers(low=1, high=max_distance, size=(cities_amount, cities_amount),
                                                dtype=np.uint32)
    distances_matrix = np.triu(distances_matrix, k=1) + (np.triu(distances_matrix, k=0)).T
    city_sizes_vector: np.ndarray = rng.integers(low=1, high=max_city_size, size=(cities_amount,), dtype=np.uint32)
    # int64 used in numpy arrays by default is a drastic overkill, uint32 will be way more than enough
    return distances_matrix, city_sizes_vector


def function_F(size1: np.uint32, size2: np.uint32, distance: np.uint32 ) -> np.uint64:
    """
    this function takes two city sizes and a distance between the cities and returns the amount of money earned
    for connecting those two together. By definition F(sizeA, sizeB, dist) = F(sizeB, sizeA, dist). This function
    expects input values as np.uint32 and returns np.uint64, since the resulting number might be greater by many orders
    of magnitude. With uint64 there is no risk of an overflow, even with max (2**32 - 1) possible values as inputs.
    If inputs are larger (they never should be btw) they will be truncated to numpy.iinfo(np.uint32).max
    """
    max_val = np.iinfo(np.uint32).max
    size1 = size1 if size1 < max_val else max_val
    size2 = size2 if size2 < max_val else max_val
    distance = distance if distance < max_val else max_val

    # when performing np.log the result is automatically cast to np.float64
    money = (np.log(size1) + np.log(size2)) * np.power(distance, 5/4, dtype=np.float64) * 100
    # multiplying by a 100 effectively gets rid of two decimals - then with a clear conscience this number can be
    # rounded to an int afterward.
    return money.astype(np.uint64)  # there is absolutely no point in being more precise than an
    # int value, especially since in this case "money" usually is in magnitude of millions or larger

