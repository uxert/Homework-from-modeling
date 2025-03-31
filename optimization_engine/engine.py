"""
This file contains computational backbone of the whole project - here the given optimization problem is actually
being solved.
It's meant to be imported from the package it's in.
"""
from math import ceil

import numpy as np
from typing import Tuple, List, Literal, Any, Union, Callable
import itertools
from numpy import dtype, ndarray
from . import constants as cn
from . import metrics, random_generators, utils, genetic


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
    USE_REWARDS_MATRIX = True

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

    function_F = staticmethod(metrics.function_F)
    calculate_distances_matrix = staticmethod(metrics.calculate_distances_matrix)
    calculate_reward_matrix = staticmethod(metrics.calculate_reward_matrix)
    goal_achievement_function = staticmethod(metrics.goal_achievement_function)
    symmetrize_numpy_matrix = staticmethod(utils.symmetrize_numpy_matrix)
    random_choose_2_arrays = staticmethod(utils.random_choose_2_arrays)
    generate_random_city_cords = staticmethod(random_generators.generate_random_city_cords)
    generate_one_candidate = staticmethod(random_generators.generate_one_candidate)
    select_parents_tournament = staticmethod(genetic.select_parents_tournament)
    mutate_bool_ndarray = staticmethod(genetic.mutate_bool_ndarray)
    crossover = staticmethod(genetic.crossover)

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

    def goal_function_convenient(self, distances_matrix: np.ndarray[np.uint32], sizes_vector: np.ndarray[np.uint32],
                                connections_matrix: np.ndarray[bool], max_cost: np.uint64 = None,
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

    def _check_one_candidate(self, one_candidate: np.ndarray[bool],
                             distances_matrix: np.ndarray[np.uint32], sizes_vector: np.ndarray[np.uint32],
                             rng: np.random.Generator, prune_factor = 0.2, rewards_matrix: ndarray[np.uint64] = None)\
            -> np.ndarray[bool] | None:
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
                                   rng: np.random.Generator = None) -> List[np.ndarray[bool]]:
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

    def genetic_algorithm(self, distances_matrix: np.ndarray[np.uint32], sizes_vector: np.ndarray[np.uint32],
                          initial_population: np.ndarray[bool] = None, population_size = 100, seed = None,
                          rng: np.random.Generator = None, iterations = 100, mutation_chance = 0.005,
                          guaranteed_elites: int = 1, silent=False) -> np.ndarray[bool]:
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
            Datatype of this array has to be bool. Each connection matrix has the same shape as distances_matrix
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
        if initial_population is not None and not isinstance(initial_population.dtype, bool):
            raise ValueError(f"Initial population must be boolean, got datatype {initial_population.dtype} instead")

        cities_amount = distances_matrix.shape[0]
        if self.USE_REWARDS_MATRIX:
            self.rewards_matrix = self.calculate_reward_matrix(distances_matrix, sizes_vector)
        else:
            self.rewards_matrix = None

        if initial_population is None:
            initial_population = self._generate_first_population(distances_matrix, sizes_vector, population_size,
                                                                 cities_amount, rng=rng)

        offspring = np.array(initial_population, dtype=bool)
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
                print(f"Generation {iteration}, best solution fittness: {np.max(goal_achievement):_}")

        print(f"Best found solution fittness: {np.max(next_goal_achievement):_}")

        return offspring

def simple_test():
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
