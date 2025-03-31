"""
Contains functions that pseudo-randomly (you can provide seed) generate random data, like random coordinates or
random connections between cities
"""

import numpy as np
from typing import Tuple, Literal, Any
from . import utils


def generate_random_city_cords(cities_amount: np.uint8, max_city_size=np.uint32(2 ** 16 - 1),
                               max_coordinate_val=np.uint32(2 ** 16 - 1), seed: int = None) \
        -> Tuple[np.ndarray[np.uint32], np.ndarray[np.uint32]]:
    """
    This function generates random cities coordinates on a square map and random city sizes. Returns results as two
    separate ndarrays. Providing the seed allows to reproduce results. This function guarantees, that no 2 cities
    will have the exact same coordinates

    Coordinates are returned as 2D ndarray of shape (cities_amount, 2) and sizes are returned as 1D ndarray of shape
    (cities_amount,)

    :param max_coordinate_val: How big is side of the map. Smallest possible map is 32 x 32, biggest possible map
        is (2^31 - 1) x (2^31 - 1). Not 2^32 to guarantee that the distance does not overflow
    :param max_city_size: How big can be a single city. This value cannot be smaller than 64 and can go up to 2^32-1
    :return: A tuple: (Coordinates, sizes_vector)
    """
    if cities_amount < 2 or cities_amount > 255:
        wrong_cities_amount_message = (f"Cities amount expected to be a number between 2 and 255 (inclusive), "
                                       f"got {cities_amount} instead")
        raise ValueError(wrong_cities_amount_message)
    if max_coordinate_val < 32 or max_coordinate_val > 2 ** 31 - 1:
        wrong_coordinate_val = (f"max_coordinate_val expected to be a number between 32 and 2^31 - 1 (inclusive), "
                                f"got {max_coordinate_val} instead")
        raise ValueError(wrong_coordinate_val)
    rng = np.random.default_rng(seed)
    max_city_size = np.uint32(64) if max_city_size < 64 else max_city_size

    coordinates_list = []
    already_used_coordinates = set()
    while len(coordinates_list) < cities_amount:
        one_pair = rng.integers(max_coordinate_val, size=2, dtype=np.uint32)
        if tuple(one_pair) not in already_used_coordinates:
            coordinates_list.append(one_pair)
            already_used_coordinates.add(tuple(one_pair))
    coordinates = np.array(coordinates_list, dtype=np.uint32)
    cities_sizes = rng.integers(max_city_size, size=(cities_amount,), dtype=np.uint32)
    # noinspection PyTypeChecker
    return coordinates, cities_sizes

def generate_one_candidate(rng: np.random.Generator, cities_amount) -> np.ndarray[Any, np.dtype[bool]]:
    """This method returns one candidate, i.e. one connection matrix with shape (cities_amount, cities_amount).
    This matrix is symmetric, and it's diagonal is always False (since a city cannot be connected to itself)"""

    rng = rng if rng is not None else np.random.default_rng()
    one_candidate = rng.integers(0, 1, size=(cities_amount, cities_amount), dtype=bool, endpoint=True)
    one_candidate = utils.symmetrize_numpy_matrix(one_candidate, "upper")
    np.fill_diagonal(one_candidate, False)
    return one_candidate
