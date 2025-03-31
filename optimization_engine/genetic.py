"This contains 'genetic-like' functions, like mutation, crossover or parents_selection"

import numpy as np
from . import utils
from typing import Any


def select_parents_tournament(goal_scores: np.array, one_fight_size: int = 3,
                              excluded_candidates: int = 3, rng: np.random.Generator = None) -> np.ndarray | None:
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
        one_child = utils.random_choose_2_arrays(chosen_parents[0], chosen_parents[1], rng)
        offspring[i + parents_amount, :, :] = one_child
    return offspring

def mutate_bool_ndarray(arr: np.ndarray[Any, np.dtype[bool]], mutation_chance = 1e-2,rng: np.random.Generator = None,
                        spare_indexes: np.ndarray = None, create_copy: bool = False) -> np.ndarray[Any, np.dtype[bool]]:
    """
    This method takes a numpy ndarray of any shape and of bool datatype. Each element in this array has
    a `mutation_chance` chance to be logically flipped. Optionally `spare_indexes` can be provided to locally
    prevent mutations, which is necessary to establish elitism. If `create_copy` is provided this function will
     not make any modifications on the original and return a new array instead.

    :param spare_indexes: If boolean - Ndarray with exactly the same shape as `arr`. If an element of this array is
        True then the value with the same index in `arr` is guaranteed to NOT be mutated. If integers - 1D ndarray
        containing indexes which will NOT be mutated. Integers allow only to index the 0-th dimension - if more
        precision is required, use an array of booleans
        """

    if spare_indexes is not None and isinstance(spare_indexes.dtype, bool) and spare_indexes.shape != arr.shape:
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
