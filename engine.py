"""
this file contains computational backbone of the whole project - here the given optimization problem is actually
being solved
"""
from math import ceil

import numpy as np
from typing import Tuple, List, Literal, Any, Union, Callable
import itertools
from numpy import dtype, ndarray
import constants as cn

USE_REWARDS_MATRIX = True

class EngineWrapper:
    """
    This class is made just to take all the parameters, like max_city_size, and pass them to engine functions for you.
    It will allow to avoid repeatedly having to type millions of parameters, which would be both tedious and error-prone.
    Those parameters have to be passed just once, when creating an instance of this class. If any of these parameters
    is not provided it will be defaulted to the value from constants.py (which takes values from config.ini).

    Basic functions (for example function_F) that were previously global in the engine.py module now are static methods
    so one can still access them if a need arises. Function to solve the whole optimization problem is NOT and will not
    be static as it relies on the convenient wrappers
    """
    def __init__(self, cities_amount: np.uint8 = None, max_city_size: np.uint32 = None,
                 max_coordinate_val: np.uint32 = None, max_cost: np.uint64 = None, max_railway_len: np.uint64 = None,
                 max_connections_count: np.uint32 = None, one_rail_cost: np.uint32 = None,
                 infrastructure_cost: np.uint64 = None):

        self.cities_count = np.uint8(cn.cities_amount) if cities_amount is None else cities_amount
        self.max_city_size = np.uint32(cn.max_city_size) if max_city_size is None else max_city_size
        self.max_coordinate_val = np.uint32(cn.max_possible_distance) if max_coordinate_val is None else max_coordinate_val
        self.max_cost = np.uint64(cn.max_cost) if max_cost is None else max_cost
        self.max_railways_len = np.uint64(cn.max_railways_pieces) if max_railway_len is None else max_railway_len
        self.max_connections = np.uint32(cn.max_connections) if max_connections_count is None else max_connections_count
        self.one_rail_cost = np.uint32(cn.one_rail_cost) if one_rail_cost is None else one_rail_cost
        self.infrastructure_cost = np.uint32(cn.infrastructure_cost) if infrastructure_cost is None else infrastructure_cost

        self.rewards_matrix = None
        self.coordinates = None

    @staticmethod
    def generate_random_city_cords(cities_amount:np.uint8, max_city_size = np.uint32(2**16 - 1),
                                   max_coordinate_val = np.uint32(2**16 - 1), seed: int = None) \
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
        if max_coordinate_val < 32 or max_coordinate_val > 2**31 - 1:
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

    @staticmethod
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
        distances_matrix = EngineWrapper.symmetrize_numpy_matrix(distances_matrix, "lower")  # have to copy lower side
        # because the lower side was computed
        return distances_matrix

    def generate_cities(self, cities_amount:np.uint8 = None, max_city_size: np.uint32 = None,
                        max_distance: np.uint32 = None, seed: int = None) \
        -> Tuple[np.ndarray[np.uint32], np.ndarray[np.uint32], np.ndarray[np.uint32]]:
        """
        This method calls the static method self.generate_random_city_cords(...) and then
        self.calculate_distances_matrix(....). If any of given parameters is None, then its value is defaulted to the
        value created in __init__() of this object before it's passed to the
        static function, with seed being the only exception.

        :param seed: Given seed is passed directly to the underlying self.generate_random_city_cords() - if None is
            given, then None will be passed further.
        :return: A tuple of 3 ndarrays - distances_matrix, sizes_vector and cities_coordinates
        """
        cities_amount = self.cities_count if cities_amount is None else cities_amount
        max_city_size = self.max_city_size if max_city_size is None else max_city_size
        max_distance = self.max_coordinate_val if max_distance is None else max_distance
        coordinates, sizes_vector = self.generate_random_city_cords(cities_amount, max_city_size, max_distance, seed)
        distances_matrix = self.calculate_distances_matrix(coordinates, use_manhattan_metric=True)
        return distances_matrix, sizes_vector, coordinates

    # noinspection PyIncorrectDocstring
    @staticmethod
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
        money = (np.log(size1) + np.log(size2))/(2 * np.log(max_param_val)) * np.tanh(distance/(10**6)) * max_result_val
        return money.astype(np.uint64)  # there is absolutely no point in being more precise than an
        # int value, especially since in this case "money" usually is in magnitude of millions or larger

    @staticmethod
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
        achievement_function = EngineWrapper.function_F if achievement_function is None else achievement_function
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

        rewards_matrix = EngineWrapper.symmetrize_numpy_matrix(rewards_matrix, "lower")  # watch out, lower
        # side is filled so lower has to be copied!
        return rewards_matrix

    # noinspection PyIncorrectDocstring
    @staticmethod
    def goal_achievement_function(distances_matrix: np.ndarray[np.uint32], sizes_vector: np.ndarray[np.uint32],
                                  connections_matrix: np.ndarray[np.bool], max_cost: np.uint64,  max_railway_len: np.uint64,
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
                individual_scores_matrix[row,col] = EngineWrapper.function_F(sizes_vector[row], sizes_vector[col],
                                                                             distances_matrix[row,col])
        else:
            for row, col in connection_indexes:
                individual_scores_matrix[row,col] = rewards_matrix[row,col]
        overall_score = np.sum(individual_scores_matrix)
        return overall_score

    def goal_function_convenient(self, distances_matrix: np.ndarray[np.uint32], sizes_vector: np.ndarray[np.uint32],
                                connections_matrix: np.ndarray[np.bool], max_cost: np.uint64 = None,
                                max_railway_len: np.uint64 = None, max_connections_count: np.uint32 = None,
                                one_rail_cost: np.uint32 = None, one_infrastructure_cost: np.uint32 = None,
                                rewards_matrix: ndarray[np.uint64] = None) -> Union[np.uint64, np.ndarray[np.uint64]]:
        """
        This method calls static method self.goal_achievement_function(...) and passes to it all given parameters.
        If any of the parameters is None, then it's value is defaulted to the one created during this instance
        __init__() before being passed further.

        Connections matrix can have more than two dimensions. If so,it is interpreted as a batch
        of 2D connection matrices (two final dimensions are interpreted as connection matrices)

        Refer to docs of self.goal_achievement_function for more details.

        :param connections_matrix: 2 (or more) dimensional ndarray. If two-dimensional it is simply passed to the
            self.goal_achievement_function(...). If more than 2D it is interpreted as a batch and each connection matrix
            is passed individually, For example, if an array of shape (7, 15, 15) is provided then the underlying
            goal_achievement_function() will be called 7 times, each time with 15 x 15 connection matrix.
        :param rewards_matrix: 2D ndarray of the same shape as distances, i.e. of shape (cities_amount,
            cities_amount). If provided it will be used to determine rewards for connecting the cities. Refer to docs
            of self.goal_achievement_function for more details.

        :return: effect of call on the underlying self.goal_achievement_function(). If one connection matrix was provided
            a scalar is returned. If a whole batch was provided then returns an array with the last two dimensions
            reduced
        """
        if connections_matrix.ndim < 2:
            err_wrong_shape = (f"Connection matrix has to contain at least two dimensions (more if a batch), but array "
                               f"of shape {connections_matrix.shape} was provided")
            raise ValueError(err_wrong_shape)
        max_cost = self.max_cost if max_cost is None else max_cost
        max_railway_len = self.max_railways_len if max_railway_len is None else max_railway_len
        max_connections_count = self.max_connections if max_connections_count is None else max_connections_count
        one_rail_cost = self.one_rail_cost if one_rail_cost is None else one_rail_cost
        one_infrastructure_cost = self.infrastructure_cost if one_infrastructure_cost is None else one_infrastructure_cost

        if connections_matrix.ndim == 2:
            return self.goal_achievement_function(distances_matrix, sizes_vector, connections_matrix, max_cost,
                                              max_railway_len, max_connections_count, one_rail_cost,
                                              one_infrastructure_cost, rewards_matrix)

        if connections_matrix.ndim > 2:
            resulting_arr = np.zeros(shape=connections_matrix.shape[:-2], dtype=np.uint64)  # reduces last 2 dimensions

            # unfortunately I did not find an easy way to do this with numpy, so we will perform slow python instead :)
            all_indexes = itertools.product(*[range(dim) for dim in resulting_arr.shape])
            for index in all_indexes:
                one_connection_matrix = connections_matrix[index]
                one_goal_result = self.goal_achievement_function(distances_matrix, sizes_vector, one_connection_matrix,
                                                                 max_cost,max_railway_len, max_connections_count,
                                                                 one_rail_cost,one_infrastructure_cost, rewards_matrix)
                resulting_arr[index] = one_goal_result

            # noinspection PyTypeChecker
            return resulting_arr


    @staticmethod
    def symmetrize_numpy_matrix(matrix: np.ndarray, copied_side: Literal["upper","lower"] = "upper") -> np.ndarray|None:
        """
        this method takes one matrix and returns symmetric matrix - when copied_side is 'upper' then upper side is
        copied to the lower side, analogically with copied_side = 'lower'. If provided array has more than 2 dimensions
        the operation is applied to the final two axes. If provided array has less than 2D None is returned
        """
        if matrix.ndim < 2:
            raise ValueError(f"An array to symmetrize needs to have at least 2 dimensions, got {matrix.ndim} instead")
        if matrix.shape[-1] != matrix.shape[-2]:
            raise ValueError(f"Final two dimensions need to have the same size, got ({matrix.shape[-1]}, {matrix.shape[-2]}) instead")
        if copied_side == "upper":
            return np.triu(matrix, k=1).swapaxes(-1, -2) + np.triu(matrix, k=0)
        if copied_side == "lower":
            return np.tril(matrix, k=-1).swapaxes(-1, -2) + np.tril(matrix, k=0)

    @staticmethod
    def generate_one_candidate(rng: np.random.Generator, cities_amount) -> np.ndarray[Any, dtype[np.bool]]:
        """This method returns one candidate, i.e. one connection matrix with shape (cities_amount, cities_amount).
        This matrix is symmetric, and it's diagonal is always False (since a city cannot be connected to itself)"""

        rng = rng if rng is not None else np.random.default_rng()
        one_candidate = rng.integers(0, 1, size=(cities_amount, cities_amount), dtype=np.bool, endpoint=True)
        one_candidate = EngineWrapper.symmetrize_numpy_matrix(one_candidate, "upper")
        np.fill_diagonal(one_candidate, False)
        return one_candidate


    def _check_one_candidate(self, one_candidate: np.ndarray[np.bool],
                             distances_matrix: np.ndarray[np.uint32], sizes_vector: np.ndarray[np.uint32],
                             rng: np.random.Generator, prune_factor = 0.2, rewards_matrix: ndarray[np.uint64] = None)\
            -> np.ndarray[np.bool] | None:
        """
        This method checks whether the candidate satisfies all the constraints, like max_connections or fitting inside
        the budget. If it does not satisfy the constraints connections are randomly removed until this solution
        has positive goal achievement function. If it is not achieved before there are no connections left, returns None

        :param prune_factor: This float represents how many connections are removed with each subsequent iteration.
            Setting it to higher values will result in less computation required to generate a population (fewer
            iterations) but will result in over-pruned solutions having generally fewer connections than they could.
        """
        while self.goal_function_convenient(distances_matrix, sizes_vector, one_candidate,
                                            rewards_matrix=rewards_matrix) < 1:
            connections = np.nonzero(np.triu(one_candidate, k=1))  # k can be both 0 and 1, diagonal is always False
            connections_amount = len(connections[0])
            if connections_amount <= 1:  # if there is one connection there will be none after removing
                return None
            removed_connections_amount = ceil(prune_factor * connections_amount) # ceil so that it is always at least 1
            removed_connections = rng.integers(connections_amount, size=(removed_connections_amount,))

            for removed_connection_idx in removed_connections:
                row, column = connections[0][removed_connection_idx], connections[1][removed_connection_idx]
                one_candidate[row, column] = False

            one_candidate = self.symmetrize_numpy_matrix(one_candidate, "upper") # the side has to be upper
            # because connections were taken from np.triu(one_candidate)
        return one_candidate

    def _generate_first_population(self, distances_matrix: np.ndarray[np.uint32], sizes_vector: np.ndarray[np.uint32],
                                   population_size: int, cities_amount: int = None,
                                   rng: np.random.Generator = None) -> List[np.ndarray[np.bool]]:
        """
        This function is used to generate the first population for the genetic algorithms. It returns o List of 2D
        numpy ndarrays. Each element of this list is one connection matrix, which corresponds to one candidate solution.
        It ensures that every generated candidate on this list is correct (it satisfies all the constraints - has
        goal_achievement function positive)
        """
        rng = np.random.default_rng() if rng is None else rng
        cities_amount = self.cities_count if cities_amount is None else cities_amount
        population = []
        for i in range(population_size):
            one_candidate = self.generate_one_candidate(rng, cities_amount)
            one_candidate = self._check_one_candidate(one_candidate, distances_matrix, sizes_vector, rng,
                                                      rewards_matrix=self.rewards_matrix)
            population.append(one_candidate)

        #now it needs to check, if there are no None candidates - solutions, that did not pass the constraints check
        for i, elem in enumerate(population):
            if elem is not None:
                continue
            new_elem = None
            while new_elem is None:
                new_elem = self.generate_one_candidate(rng, cities_amount)
                new_elem = self._check_one_candidate(new_elem, distances_matrix, sizes_vector, rng,
                                                     rewards_matrix=self.rewards_matrix)
            population[i] = new_elem
        # noinspection PyTypeChecker
        return population

    @staticmethod
    def select_parents_tournament(goal_scores: np.array, one_fight_size: int = 3,
                                  excluded_candidates: int = 3, rng: np.random.Generator = None) -> ndarray | None:
        """
        This method takes a 1D numpy ndarray with scores and performs tournament selection, then returns indexes of
            elements from the scores array that 'passed' the tournament
        :param goal_scores: 1D numpy array containing scores. Will return None if the array is not 1D
        :param one_fight_size: How many solutions are in one tournament group. Only the best one from each group
            will be selected, others are forgotten forever...
        :param excluded_candidates: How many best solutions are excluded from the tournament. These candidates do not
            take part in the tournament at all, meaning no solution will be 'killed' simply because it was worse in
            comparison.
        :param rng: Numpy random.Generator instance. It is used to 'randomly' divide solutions into tournament groups.
            Passing identical instance of the generator (WITH EXACTLY THE SAME STATE - watch for earlier uses of passed
            rng!) will ensure reproducibility.
        :return: Numpy ndarray containing indexes of chosen parents. Indexing the array used to create `goal_scores`
            with this returned array will give the chosen solutions
        """
        if goal_scores.ndim != 1:
            return None
        rng = np.random.default_rng() if rng is None else rng
        # sorted_scores_idx = np.argsort(goal_scores)[::-1]  # oddly specific comment about the need to reverse the
        # argsort()'s result since it is sorting in ascending order...
        sorted_scores_idx = np.argpartition(goal_scores, kth=-excluded_candidates)[::-1]
        winners = []
        for i in range(excluded_candidates):
            winners.append(sorted_scores_idx[i])
        indexes = rng.permutation(np.arange(start=excluded_candidates, stop=len(goal_scores)))
        # indexes = np.array_split(np.array(indexes),
        #                          indices_or_sections=(len(goal_scores) - guaranteed_survivors)// tournament_size + 1)
        tournament_scores = [goal_scores[i] for i in indexes]

        indexes = np.array_split(np.array(indexes),
                                 indices_or_sections=(len(goal_scores) - excluded_candidates) // one_fight_size + 1)

        tournament_scores = np.array_split(np.array(tournament_scores),
                                           indices_or_sections=(len(goal_scores) - excluded_candidates) // one_fight_size + 1)

        tournament_winners_positions = [np.argmax(one_fight) for one_fight in tournament_scores]
        for fight_no, winner in enumerate(tournament_winners_positions):
            winners.append(indexes[fight_no][winner])

        return np.array(winners)

    @staticmethod
    def random_choose_2_arrays(a: np.ndarray, b: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
        """Takes two ndarrays of the same shape and returns another ndarray of the same shape that has
        its elements randomly chosen from `a` and `b`"""
        if a.shape != b.shape:
            raise ValueError(f"Random choose from 2 arrays expected the arrays to have the same shape, got {a.shape} and {b.shape} instead")
        rng = rng if rng is not None else np.random.default_rng()
        child_indexes = rng.integers(0, 1, size=a.shape, endpoint=True)
        child = np.where(child_indexes == 0, a, b)
        return child

    @staticmethod
    def crossover(parents: np.ndarray, offspring_count: int, rng: np.random.Generator = None)\
            -> np.ndarray:
        """
        This function takes a 3D numpy ndarray with shape (parents_count, (...)) and returns another 3D ndarray of shape
        (parents_count + offspring_count, (...)) in which first parents_count entries are just copied from `parents`
        and every next entry is a 'child' - an array with elements randomly chosen from two random 'parents'.

        :param parents: Numpy ndarray containing exactly 3 dimensions where the first (i.e. shape[0]) dimension represents
            the batch size
        :param offspring_count: How many additional children will be created
        :param rng: If provided the output of this function is fully reproducible
        """
        if parents.ndim != 3:  # batch of 2D connection matrices is expected
            raise ValueError(f"Crossover expected a np.ndarray with exactly 3 dimensions, got {parents.ndim} instead")
        parents_amount = parents.shape[0]
        offspring = np.empty((parents_amount + offspring_count, parents.shape[1], parents.shape[2]), dtype=parents.dtype)
        offspring[0:parents_amount, :, :] = parents
        rng = np.random.default_rng() if rng is None else rng
        for i in range(offspring_count):
            indexes = rng.choice(len(parents), size=(2,), replace=False)
            chosen_parents = parents[indexes]
            one_child = EngineWrapper.random_choose_2_arrays(chosen_parents[0], chosen_parents[1], rng)
            offspring[i + parents_amount, :, :] = one_child
        return offspring

    @staticmethod
    def mutate_bool_ndarray(arr: ndarray[Any, dtype[np.bool]], mutation_chance = 1e-2,rng: np.random.Generator = None,
                            spare_indexes: ndarray = None, create_copy: bool = False) -> ndarray[Any, dtype[np.bool]]:
        """
        This method takes a numpy ndarray of any shape and of np.bool datatype. Each element in this array has
        a `mutation_chance` chance to be logically flipped. Optionally `spare_indexes` can be provided to locally
        prevent mutations, which is necessary to establish elitism. If `create_copy` is provided this function will
         not make any modifications on the original and return a new array instead.

        :param spare_indexes: If boolean - Ndarray with exactly the same shape as `arr`. If an element of this array is
            True then the value with the same index in `arr` is guaranteed to NOT be mutated. If integers - 1D ndarray
            containing indexes which will NOT be mutated. Integers allow only to index the 0-th dimension - if more
            precision is required, use an array of booleans
            """

        if spare_indexes is not None and spare_indexes.dtype == np.bool and spare_indexes.shape != arr.shape:
            err_shape_mismatch_message = (f"boolean spare_indexes need to have the same shape as `arr`, but received "
                                          f"shapes {arr.shape} and {spare_indexes.shape}")
            raise ValueError(err_shape_mismatch_message)
        if spare_indexes is not None and np.issubdtype(spare_indexes.dtype, np.integer) and spare_indexes.ndim != 1:
            err_dim_mismatch_message = (f"integer spare_indexes need to be exactly one-dimensional, but an array with"
                                        f"{spare_indexes.ndim} dimensions was provided")
            raise ValueError(err_dim_mismatch_message)

        new_arr = arr.copy() if create_copy else arr

        rng = rng if rng is not None else np.random.default_rng()
        flipped_mask = rng.choice([True, False], size=arr.shape, p=[mutation_chance, 1 - mutation_chance])
        if spare_indexes is not None:
            flipped_mask[spare_indexes] = False
        new_arr[flipped_mask] = np.logical_not(new_arr[flipped_mask])
        return new_arr

    def genetic_algorithm(self, distances_matrix: np.ndarray[np.uint32], sizes_vector: np.ndarray[np.uint32],
                          initial_population: np.ndarray[np.bool] = None, population_size = 100, seed = None,
                          rng: np.random.Generator = None, iterations = 100, mutation_chance = 0.005,
                          guaranteed_elites: int = 1, silent=False) -> np.ndarray[np.bool]:
        """
        This function performs a genetic algorithm for given amount of iterations. Distances matrix and sizes vector
        need to have appropriate shapes, i.e. (cities_amount, cities_amount) and (cities_amount,) respectively.
        Cities amount is not passed directly, but rather inferred from the aforementioned ndarrays.
        If initial population is not provided, one will be randomly generated using provided rng. If rng is not provided
        new one will be created from given seed.
        :param distances_matrix: A square, symmetrical matrix representing distances between cities.
        :param sizes_vector: A 1D ndarray representing the size of each city.
        :param initial_population: A 3D numpy ndarray where the first (i.e. '0-th') dimension is a batch dimension.
            Each element represents one 2D connection matrix (for more details refer to goal_achievement_function().
            Datatype of this array has to be np.bool. Each connection matrix has the same shape as distances_matrix
        :param population_size: Used to determine the size of newly created population. Ignored if initial_population is
            provided.
        :param seed: Used to create new random Generator. Ignored if rng is provided. If neither is provided then
            a Generator with fresh, unpredictable entropy will be created.
        :param rng: Allows for full control over every randomness used in this function, passing the same generator
            (and in the exact same state!) guarantees complete reproducibility
        :param mutation_chance: Used during the mutation phase of the algorithm - represents a chance for each element
            of connection matrix to be logically flipped (create connection/remove connection). Bigger values will
            aid in overcoming local maximum but setting this value too big will result in instability of the algorithm.
        :param guaranteed_elites: if greater than 0 then this amount of solutions with the highest scores is
            considered 'elites' - they do not 'fight' in the tournament and do not undergo mutations to ensure they are
            never lost
        :param silent: if True, no in-training communicates will be printed - only the best final score. If False,
            every iteration it's best score will be printed to console so that the progress can be tracked
        :return: the best found solution
        """
        rng = rng if rng is not None else np.random.default_rng(seed)
        if distances_matrix.shape[0] != distances_matrix.shape[1] != sizes_vector.shape[0]:
            err_shape_mismatch = f"Distances_matrix and sizes_vector do not have matching shapes, cannot establish one"\
                             f" cities_amount. Received shapes {distances_matrix.shape} and {sizes_vector.shape}"
            raise ValueError(err_shape_mismatch)
        if initial_population is not None and initial_population.dtype != np.bool:
            raise ValueError(f"Initial population must be boolean, got datatype {initial_population.dtype} instead")

        cities_amount = distances_matrix.shape[0]
        if USE_REWARDS_MATRIX:
            self.rewards_matrix = self.calculate_reward_matrix(distances_matrix, sizes_vector)
        else:
            self.rewards_matrix = None

        if initial_population is None:
            initial_population = self._generate_first_population(distances_matrix, sizes_vector, population_size,
                                                                 cities_amount, rng=rng)

        offspring = np.array(initial_population, dtype=np.bool)
        next_goal_achievement = self.goal_function_convenient(distances_matrix, sizes_vector, offspring,
                                                              rewards_matrix=self.rewards_matrix)
        for iteration in range(iterations):
            goal_achievement = next_goal_achievement
            population = offspring
            parents_indexes = self.select_parents_tournament(goal_achievement,excluded_candidates=1, rng=rng)
            chosen_parents = population[parents_indexes]
            offspring_size = len(population) - len(chosen_parents)
            offspring = self.crossover(chosen_parents, offspring_size, rng=rng)
            if guaranteed_elites > 0:  # if it is not greater than 0 there is no point in calculating scores again
                offspring = self.symmetrize_numpy_matrix(offspring, "upper")
                temp_scores = self.goal_function_convenient(distances_matrix, sizes_vector, offspring,
                                                            rewards_matrix=self.rewards_matrix)
                # sorted_solutions_idx = np.argsort(temp_scores)[::-1]
                sorted_solutions_idx = np.argpartition(temp_scores, kth=-guaranteed_elites)[::-1]
                best_solution_idx = sorted_solutions_idx[0:guaranteed_elites]
            else:
                best_solution_idx = None
            offspring = self.mutate_bool_ndarray(offspring, mutation_chance, rng=rng, spare_indexes=best_solution_idx)
            offspring = self.symmetrize_numpy_matrix(offspring, "upper")

            next_goal_achievement = self.goal_function_convenient(distances_matrix, sizes_vector, offspring,
                                                                  rewards_matrix=self.rewards_matrix)
            if not silent:
                print(f"Generation {iteration}, best solution fittness: {np.max(goal_achievement)}")

        print(f"Best found solution fittness: {np.max(next_goal_achievement)}")

        return offspring




if __name__ == "__main__":
    # test block to see if everything works properly. This will never launch if the script is only imported, as it is
    # meant to be. This will be cleaned after completing the whole engine
    print(20 * "-" + "Genetic Algorithm test" + 20 * "-")
    my_rng = np.random.default_rng(42)
    test_instance = EngineWrapper(cities_amount=np.uint8(30), max_coordinate_val=np.uint32(500), max_city_size=np.uint32(500),
                                  max_cost = np.uint64(4_000_000), max_railway_len=np.uint64(200_000),
                                  max_connections_count=np.uint32(200), one_rail_cost=np.uint32(100),
                                  infrastructure_cost=np.uint32(10000))
    dist, sizes, cords = test_instance.generate_cities(seed=42)
    last_population = test_instance.genetic_algorithm(distances_matrix=dist, sizes_vector=sizes, rng=my_rng)
    temp = input("\nDo you wish to print all the solutions (y/n)? ")
    if temp.lower() == "y":
        with np.printoptions(threshold=np.inf):
            for solution in last_population.astype(np.ushort):  # changed to integer type for more concise printing
                print(solution)
                print("\n")
