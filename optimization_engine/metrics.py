"""This contains functions needed to evaluate score of any population"""

import numpy as np
from numpy import ndarray
from . import utils
from typing import Callable


# noinspection PyIncorrectDocstring
def function_F(size1: np.uint32, size2: np.uint32, distance: np.uint32, max_result_val: np.uint64 = np.uint64(5e10)) \
        -> np.uint64:
    """
    this function takes two city sizes and a distance between the cities and returns the amount of money earned
    for connecting those two together. By definition F(sizeA, sizeB, dist) = F(sizeB, sizeA, dist).
    With uint64 there is no risk of an overflow when using this function, even with max (2**32 - 1) possible values
    as inputs. If inputs are larger (they never should be) they will be truncated to numpy.iinfo(np.uint32).max

    This function implements tanh(distance/(10^6)) * max_result_val * (ln(size1) + ln(size2)). It is always increasing
    with respect to all of its inputs. However, tanh is scaled in such a way that for distances smaller than 100 000
    it is almost linear. For distance of one million it achieves around 76% of given max_result_val, with distance of
    2 million it is 96%. Increasing distances greater than 100 000 start to yield noticeably lower money increase
    (sigmoid-like shape)

    The above formula is purely arbitrary and any callable taking two sizes and a distance that returns
    one numpy.uint64 can be used instead. If you want to be sure nothing overflows (neither numpy nor this program will
    warn you about it!!) make sure that value returned by your custom callable is never bigger than 5.6e14
    since max money value is  1.844674e+19 and as for this moment max 32 385 connections are possible (255 choose 2).
    Of course this will not be correct if number of cities is greater than 255 (which at the moment of writing this doc
    is not allowed and there are no plans to change it in the future)

    :param max_result_val: this parameter represents the supremum of the whole formula - this value is a limit of
        this expression as all 3 inputs go to +infinity. For maximum possible (2^^32-1) inputs computed value is so
        close to this parameter that a difference is too small to be represented in np.64float
    """
    max_param_val = np.iinfo(np.uint32).max
    size1 = size1 if size1 < max_param_val else max_param_val
    size2 = size2 if size2 < max_param_val else max_param_val
    distance = distance if distance < max_param_val else max_param_val

    # this performs a tanh(distance) * log(size1*size2) function with supremum of max_result_val
    money = (np.log(size1) + np.log(size2)) / (2 * np.log(max_param_val)) * np.tanh(
        distance / (10 ** 6)) * max_result_val
    return money.astype(np.uint64)  # there is absolutely no point in being more precise than an
    # int value, especially since in this case "money" usually is in magnitude of millions or larger

def calculate_distances_matrix(coordinates: ndarray[np.uint32], use_manhattan_metric=True) -> ndarray[np.uint32]:
    """
    this method takes a 2D coordinates array of shape (cities_amount, 2) and returns a 2D ndarray of shape
    (cities_amount, cities_amount) where element with index [i,j] is the distance between cities i and j.
    :param use_manhattan_metric: If set to True (default) evaluates Manhattan distance - sum of the absolute
        differences between the coordinates. When set to False standard Euclidean metric will be used (rounding up
        to a whole integer)
    """
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"Expected coordinates to have shape (cities_amount, 2), got {coordinates.shape} instead")
    cities_amount = len(coordinates)
    distances_matrix = np.zeros(shape=(cities_amount, cities_amount), dtype=np.uint32)
    cords = coordinates.astype(np.int64)  # changing to int64 to allow for negative differences and to 'fit'
    # squaring when calculating Euclidean
    # This loop will leave diagonal not-computed (initialized with 0). It is intentional since distance from a
    # city to itself is always 0
    for row in range(cities_amount):
        for col in range(row):
            difference = cords[row] - cords[col]  # so that it can be negative
            if use_manhattan_metric:
                distances_matrix[row, col] = np.sum(np.abs(difference))
            else:
                difference = np.sqrt(np.sum(np.square(difference)))
                distances_matrix[row, col] = np.round(difference)
    distances_matrix = utils.symmetrize_numpy_matrix(distances_matrix, "lower")  # have to copy lower side
    # because the lower side was computed
    return distances_matrix

def calculate_reward_matrix(distances_matrix: ndarray[np.uint32], sizes_vector: ndarray[np.uint32],
                            achievement_function: Callable = None, max_result_val: np.uint64 = None)\
        -> np.ndarray[np.uint64]:
    """
    This method allows to leverage the fact, that in the solved optimization problem 'alleles' are both binary
    and completely independent. This means one can calculate how much 'reward' each individual allele brings and
    avoid calculating this each time. Of course this 'reward_matrix' can be used only when distances between cities,
    their sizes and the function used to calculate reward remain CONSTANT. When they change, the reward matrix has
    to be calculated once again for new parameters.
    :param distances_matrix: A symmetric 2D numpy array representing distances between cities
    :param sizes_vector: A 1D numpy array representing the size of each city
    :param achievement_function: A Callable that accepts the same parameters as
        EngineWrapper.function_F(). When not provided, the default function_F() will be used. Refer to docs of
        EngineWrapper.function_F() for details about customizing the reward function.
    :param max_result_val: This parameter allows to specify the reward function's value for max possible inputs.
        If None, it will NOT be passed to the reward function at all - your custom function does not have to take it
    :return: A reward matrix - 2D np.ndarray of the same shape as `dist`. Element with index [i,j] represents how
        much money does one earn by connecting cities with indexes i and j. Of course this matrix by definition
        is symmetric.
    """
    achievement_function = function_F if achievement_function is None else achievement_function
    if distances_matrix.ndim !=2 or sizes_vector.ndim != 1:
        dim_mismatch_err_message = (f"Expected distances matrix to have 2 dimensions and sizes vector to have one,"
                                    f"but received distances with {distances_matrix.ndim} dimensions and sizes with"
                                    f" {sizes_vector.ndim} dimensions instead")
        raise ValueError(dim_mismatch_err_message)
    if np.any(distances_matrix != distances_matrix.T):
        raise ValueError("Expected distances_matrix to be symmetric, not symmetric matrix received")

    # noinspection PyTypeChecker
    rewards_matrix: ndarray[np.uint64] = np.zeros_like(distances_matrix, dtype=np.uint64)

    if max_result_val is None:
        for row_no in range(rewards_matrix.shape[0]):
            for col_no in range(row_no):
                rewards_matrix[row_no, col_no] = achievement_function(sizes_vector[row_no], sizes_vector[col_no],
                                                                      distances_matrix[row_no, col_no])
    else:
        for row_no in range(rewards_matrix.shape[0]):
            for col_no in range(row_no):
                rewards_matrix[row_no, col_no] = achievement_function(sizes_vector[row_no], sizes_vector[col_no],
                                                                  distances_matrix[row_no, col_no], max_result_val)

    rewards_matrix = utils.symmetrize_numpy_matrix(rewards_matrix, "lower")  # watch out, lower
    # side is filled so lower has to be copied!
    return rewards_matrix

def goal_achievement_function(distances_matrix: np.ndarray[np.uint32], sizes_vector: np.ndarray[np.uint32],
                              connections_matrix: np.ndarray[bool], max_cost: np.uint64,  max_railway_len: np.uint64,
                              max_connections_count: np.uint32, one_rail_cost: np.uint32,
                              one_infrastructure_cost: np.uint32, rewards_matrix: ndarray[np.uint64] = None) \
        -> np.uint64:
    """This function takes exactly one solution (or solution candidate) and evaluates it with non-negative score.
    On the caller of this function lies the responsibility for ensuring that both matrices and the size_vector have
    appropriate shapes (the same number of cities), that all provided values have correct datatypes and that the
    matrices are symmetrical (they need to be because of what they represent)
    :param distances_matrix: Matrix where element m[i][j] is the distance between city i and city j. By definition
        m[i][j] = m[j][i] (this matrix has to be symmetrical)
    :param connections_matrix: Matrix of booleans where element m[i][j] represents whether there is a connection
        between cities i and j. Of course m[i][j] has to be equal to m[j][i], same as with the distances_matrix.
    :param rewards_matrix: 2D ndarray of shape (cities_amount, cities_amount) - the same shape as distances_matrix.
        If provided, element with index [i,j] will represent the reward for connecting cities with indexes i and j.
        If in your optimization problem distances, sizes and function used to calculate reward are constant
        do not change often calculating this matrix beforehand is HIGHLY RECOMMENDED - it will save a ton of
        computation and improve performance.
     :returns: If the solution does not satisfy any of the constraints (for example too many connections) this function
         returns exactly 0. Otherwise, it returns one positive integer representing a value of this solution -
         the bigger, the better
    """
    if distances_matrix.ndim != 2 or distances_matrix.shape[0] != distances_matrix.shape[1]:
        raise ValueError(f"distances_matrix should be a 2-dimensional square numpy ndarray, got shape "
                         f"({distances_matrix.shape}) instead")
    if sizes_vector.ndim != 1:
        raise ValueError(f"sizes_vector should be a 1D numpy ndarray, instead got shape ({sizes_vector.shape})")
    if sizes_vector.shape[0] != distances_matrix.shape[0]:
        err_shapes_mismatch = f"Distances_matrix and sizes_vector do not have matching shapes, received "\
                            f"distances_matrix: ({distances_matrix.shape}) and sizes_vector: ({sizes_vector.shape})"
        raise ValueError(err_shapes_mismatch)
    if connections_matrix.shape != distances_matrix.shape:
        err_shape_mismatch = (f"shape of connections_matrix: {connections_matrix.shape} does not match with the "
                              f"shape of distances_matrix: {distances_matrix.shape}")
        raise ValueError(err_shape_mismatch)
    if rewards_matrix is not None and rewards_matrix.shape != distances_matrix.shape:
        err_shape_mismatch = (f"Expected the reward matrix to have the same shape as connections matrix, but "
                              f"received arrays with shapes {rewards_matrix.shape} and {connections_matrix.shape} "
                              f"respectively")
        raise ValueError(err_shape_mismatch)
    # first checks for max number of connections, as it requires basically no computation
    connections_count = np.uint32(np.count_nonzero(connections_matrix)) // 2
    # needs to be divided by 2 due to the nature of the connections' matrix. Thanks to numpy vectorization looping over
    # the whole matrix and dividing the result by 2 is way faster than iterating over only half of the matrix
    if connections_count > max_connections_count:
        return np.uint64(0)

    # now checks for max railway length
    built_rails_matrix = np.multiply(connections_matrix, distances_matrix)
    built_rails_count = np.sum(built_rails_matrix, dtype=np.uint64) // 2  # as with connections result is divided by 2
    if built_rails_count > max_railway_len:
        return np.uint64(0)

    # checks for max budget
    rails_cost = built_rails_count * one_rail_cost
    # using np.multiply to cast to np.uint64 to prevent any overflows
    infrastructure_costs = np.multiply(connections_count, one_infrastructure_cost, dtype=np.uint64)
    total_cost = rails_cost + infrastructure_costs
    if total_cost > max_cost:
        return np.uint64(0)

    # if the function did not return up to this point all the additional requirements are met and the actual score can
    # be calculated. Unfortunately I did not find a way to vectorize it using numpy built-ins and have to
    # operate manually using indexes

    upper_connections_half = np.triu(connections_matrix)  # this matrix by definition has to be symmetrical so we can
    # take only one half without any loss of information instead of dividing by 2 at the end - this will save computing
    rows, cols = upper_connections_half.nonzero()  # extracts indices of cities that are connected
    connection_indexes = zip(rows, cols)  # each element is a pair of indexes representing one connection

    individual_scores_matrix = np.zeros_like(connections_matrix, dtype=np.uint64)
    if rewards_matrix is None:
        for row, col in connection_indexes:
            individual_scores_matrix[row,col] = function_F(sizes_vector[row], sizes_vector[col],
                                                                         distances_matrix[row,col])
    else:
        for row, col in connection_indexes:
            individual_scores_matrix[row,col] = rewards_matrix[row,col]
    overall_score = np.sum(individual_scores_matrix)
    return overall_score