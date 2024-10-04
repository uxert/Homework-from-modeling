"""
this file contains computational backbone of the whole project - here the given optimization problem is actually
being solved
"""

import numpy as np
from typing import Tuple, List
import constants as cn


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
                 max_distance: np.uint32 = None, max_cost: np.uint64 = None, max_railway_len: np.uint64 = None,
                 max_connections_count: np.uint32 = None, one_rail_cost: np.uint32 = None,
                 infrastructure_cost: np.uint64 = None):

        self.cities_count = np.uint8(cn.cities_amount) if cities_amount is None else cities_amount
        self.max_city_size = np.uint32(cn.max_city_size) if max_city_size is None else max_city_size
        self.max_distance = np.uint32(cn.max_possible_distance) if max_distance is None else max_distance
        self.max_cost = np.uint64(cn.max_cost) if max_cost is None else max_cost
        self.max_railways_len = np.uint64(cn.max_railways_pieces) if max_railway_len is None else max_railway_len
        self.max_connections = np.uint32(cn.max_connections) if max_connections_count is None else max_connections_count
        self.one_rail_cost = np.uint32(cn.one_rail_cost) if one_rail_cost is None else one_rail_cost
        self.infrastructure_cost = np.uint32(cn.infrastructure_cost) if infrastructure_cost is None else infrastructure_cost

    @staticmethod
    def generate_random_city_values(cities_amount:np.uint8, max_city_size: np.uint32 = 2**32-1,
                                    max_distance: np.uint32 = 2**32-1, seed: int = None) \
            -> Tuple[np.ndarray[np.uint32], np.ndarray[np.uint32]]:
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
        cities_amount = np.uint8(255) if cities_amount > 255 else cities_amount
        # distances matrix has to be symmetric by definition, but of course random generation does not provide that
        # symmetry is achieved by taking only upper half of the matrix, transposing it and adding the two pieces together
        # in one of the pieces k=1 so that the 'main' diagonal is taken into the account only once
        distances_matrix: np.ndarray = rng.integers(low=1, high=max_distance, size=(cities_amount, cities_amount),
                                                    dtype=np.uint32)
        distances_matrix = np.triu(distances_matrix, k=1) + (np.triu(distances_matrix, k=0)).T
        city_sizes_vector: np.ndarray = rng.integers(low=1, high=max_city_size, size=(cities_amount,), dtype=np.uint32)
        # int64 used in numpy arrays by default is a drastic overkill, uint32 will be way more than enough
        # noinspection PyTypeChecker
        # for some reason PyCharm thinks those arrays do not have dtype=np.uint32
        return distances_matrix, city_sizes_vector

    def generate_cities(self, cities_amount:np.uint8 = None, max_city_size: np.uint32 = None,
                        max_distance: np.uint32 = None, seed: int = None) \
        -> Tuple[np.ndarray[np.uint32], np.ndarray[np.uint32]]:
        """
        This method calls the static method self.generate_random_city_values(...). If any of given parameters is None,
        then it's value is defaulted to the value created in __init__() of this object before it's passed to the
        static function, with seed being the only exception.

        Refer to docs of self.generate_random_city_values() for more information.
        :param seed: Given seed is passed directly to the underlying self.generate_random_city_values() - if None is
            given, then None will be passed further.
        :return: returns the result from call of the underlying self.generate_random_city_values()
        """
        cities_amount = self.cities_count if cities_amount is None else cities_amount
        max_city_size = self.max_city_size if max_city_size is None else max_city_size
        max_distance = self.max_distance if max_distance is None else max_distance
        return self.generate_random_city_values(cities_amount, max_city_size, max_distance, seed)

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

    # noinspection PyIncorrectDocstring
    @staticmethod
    def goal_achievement_function(distances_matrix: np.ndarray[np.uint32], sizes_vector: np.ndarray[np.uint32],
                                  connections_matrix: np.ndarray[np.bool], max_cost: np.uint64,  max_railway_len: np.uint64,
                                  max_connections_count: np.uint32, one_rail_cost: np.uint32,
                                  one_infrastructure_cost: np.uint32) -> np.uint64:
        """This function takes exactly one solution (or solution candidate) and evaluates it with non-negative score.
        On the caller of this function lies the responsibility for ensuring that both matrices and the size_vector have
        appropriate shapes (the same number of cities), that all provided values have correct datatypes and that the
        matrices are symmetrical (they need to be because of what they represent)
        :param distances_matrix: Matrix where element m[i][j] is the distance between city i and city j. By definition
            m[i][j] = m[j][i] (this matrix has to be symmetrical)
        :param connections_matrix: Matrix of booleans where element m[i][j] represents whether there is a connection
            between cities i and j. Of course m[i][j] has to be equal to m[j][i], same as with the distances_matrix.
         :returns: If the solution does not satisfy any of the constraints (for example too many connections) this function
             returns exactly 0. Otherwise, it returns one positive integer representing a value of this solution -
             the bigger, the better
        """
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
        for row, col in connection_indexes:
            individual_scores_matrix[row,col] = EngineWrapper.function_F(sizes_vector[row], sizes_vector[col],
                                                                         distances_matrix[row,col])
        overall_score = np.sum(individual_scores_matrix)
        return overall_score

    def goal_function_convenient(self, distances_matrix: np.ndarray[np.uint32], sizes_vector: np.ndarray[np.uint32],
                                connections_matrix: np.ndarray[np.bool], max_cost: np.uint64 = None,
                                max_railway_len: np.uint64 = None, max_connections_count: np.uint32 = None,
                                one_rail_cost: np.uint32 = None, one_infrastructure_cost: np.uint32 = None) -> np.uint64:
        """
        This method calls static method self.goal_achievement_function(...) and passes to it all given parameters.
        If any of the parameters is None, then it's value is defaulted to the one created during this instance
        __init__() before being passed further.

        Refer to docs of self.goal_achievement_function for more details.

        :return: effect of call on the underlying self.goal_achievement_function()
        """
        max_cost = self.max_cost if max_cost is None else max_cost
        max_railway_len = self.max_railways_len if max_railway_len is None else max_railway_len
        max_connections_count = self.max_connections if max_connections_count is None else max_connections_count
        one_rail_cost = self.one_rail_cost if one_rail_cost is None else one_rail_cost
        one_infrastructure_cost = self.infrastructure_cost if one_infrastructure_cost is None else one_infrastructure_cost

        return self.goal_achievement_function(distances_matrix, sizes_vector, connections_matrix, max_cost,
                                              max_railway_len, max_connections_count, one_rail_cost,
                                              one_infrastructure_cost)

    def _generate_first_population(self, distances_matrix: np.ndarray[np.uint32], sizes_vector: np.ndarray[np.uint32],
                                   population_size: int,cities_amount: int = None, seed: int = None)\
        -> List[np.ndarray[np.bool]]:
        """
        This function is used to generate the first population for the genetic algorithms. It returns o List of 2D
        numpy ndarrays. Each element of this list is one connection matrix, which corresponds to one candidate solution.
        It ensures that every generated candidate on this list is correct (it satisfies all the constraints - has
        goal_achievement function positive)
        """
        rng = np.random.default_rng(seed)

        cities_amount = self.cities_count if cities_amount is None else cities_amount
        population = []
        for i in range(population_size):
            one_candidate = rng.choice([True, False], (cities_amount, cities_amount), p=[0.3, 0.7]).astype(np.bool)
            population.append(one_candidate)
            print(one_candidate.dtype)
            print(type(one_candidate))

        for i in range(population_size):
            pass

        # noinspection PyTypeChecker
        return population


if __name__ == "__main__":
    # test block to see if everything works properly. This will never launch if the script is only imported, as it is
    # meant to be. This will be cleaned after completing the whole engine
    print(20 * "-" + "Static methods check" + 20 * "-")
    dist, sizes = EngineWrapper.generate_random_city_values(np.uint8(5), seed=42, max_city_size=np.uint32(20),
                                                            max_distance=np.uint32(60))
    np.random.seed(42)
    random_connections = np.random.choice([True, False], size=(5, 5))
    print("goal achievement function for randomly generated inputs:", end = " ")
    print(EngineWrapper.goal_achievement_function(dist, sizes, random_connections, np.uint64(2_000_000),
                                    np.uint64(200000), np.uint32(400),
                                    np.uint32(15), np.uint32(10_009)))
    test_max_result_val = np.float64(5e10)
    result_for_max_params = EngineWrapper.function_F(np.uint32(2 ** 32 - 1), np.uint32(2 ** 32 - 1), np.uint32(2 ** 32 - 1),
                                       max_result_val=test_max_result_val)
    print(f"max F function val: {result_for_max_params:e}")
    print(f"max F function difference from given max_val: {result_for_max_params - 5e10:e}")
    for num in [1_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 2_000_000, 2**32 - 1]:
        print(f"F function with sizes=3 for distance {num:_}: {EngineWrapper.function_F(np.uint32(3), np.uint32(3), 
                                                                                        np.uint32(num)):e}")

    print(20 * "-" + "Convenient calls check" + 20 * "-")
    test_instance = EngineWrapper(cities_amount=np.uint8(5), max_distance=np.uint32(60), max_city_size=np.uint32(20),
                                  max_cost = np.uint64(2_000_000), max_railway_len=np.uint64(200_000),
                                  max_connections_count=np.uint32(400), one_rail_cost=np.uint32(15),
                                  infrastructure_cost=np.uint32(10_009))
    dist, sizes = test_instance.generate_cities(seed=42)
    np.random.seed(42)
    random_connections = np.random.choice([True, False], size=(5, 5))
    print("goal achievement function for randomly generated inputs:", end = " ")
    print(test_instance.goal_function_convenient(dist, sizes, random_connections))

    print(test_instance._generate_first_population(dist, sizes, 3, 5))