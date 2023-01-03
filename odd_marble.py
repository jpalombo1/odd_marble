import logging
import math
import random
import sys
from enum import Enum
from typing import List, Tuple

import numpy as np

logging.basicConfig(
    format="%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
    stream=sys.stdout,
    level=logging.INFO,
    filemode="w+",
)

NUM_MARBLES = 12


class Weight(Enum):
    """Enumerate light or heavy marble."""

    LIGHT = -1
    EQUAL = 0
    HEAVY = 1


def setup_marbles(
    num_marbles: int, actual_marble: int, actual_weight: Weight
) -> List[Weight]:
    """Make all marbles include placing oddball one in designated place."""
    if actual_marble not in range(num_marbles):
        raise ValueError(f"Specify odd marble index between 0 and {num_marbles - 1}")
    if actual_weight not in [Weight.LIGHT, Weight.HEAVY]:
        raise ValueError("Specify odd marble weight as Light or Heavy")
    return [
        actual_weight if marble_num == actual_marble else Weight.EQUAL
        for marble_num in range(num_marbles)
    ]


def get_weighings(num_marbles: int) -> int:
    """Calculate max weighings needed based on number of marbles."""
    return math.ceil(math.log(num_marbles * 2, len(Weight)))


def weighing(s1: List[Weight], s2: List[Weight]) -> int:
    """Perform weighing between 2 sides. Make sure number of marbles same on both sides then get difference in sum of weights."""
    if len(s1) != len(s2):
        raise ValueError("Invalid weighing! Use same amount on each side.")
    s1_sum = sum(marble.value for marble in s1)
    s2_sum = sum(marble.value for marble in s2)
    return s1_sum - s2_sum


def get_ternary(weighings: int) -> List[Tuple[int, ...]]:
    """Get full ternary of possible weights for all possible combos of weighings.

    e.g. For 3 weighings, 3 weight vals [-1,0,1] there are 3^3 or 27 weighings and yields ternary list [(-1,-1,-1),(-1,-1,0),...(1,1,0),(1,1,1)]
    """
    max_ternary = len(Weight) ** weighings
    return [to_ternary(val, weighings) for val in range(max_ternary)]


def to_ternary(number: int, max_digits: int) -> Tuple[int, ...]:
    """Convert integer number to ternary list of trites.
    Basically fill out max_digits based on number of weighings, digit base num weights.
    Then get digits in base num_weights for given int using divmod, then insert digits into max_digit number.

    e.g. number = 11, num_weighings = 4, num_weights = 3
    divmod (14,3) = (4,2), divmod(4,3)=(1,1) divmod(1,3) = (0,1) ternary_list = [2,1,1] -> [1,0,0]
    ternary = [-1 -1 -1 -1] put in digits in ternary list reverse order [-1,0,0,1]
    """
    init_number = number
    ternary = [Weight.LIGHT.value] * max_digits
    iter = 0
    while number > 0:
        number, remainder = divmod(number, len(Weight))
        ternary[len(ternary) - 1 - iter] = remainder - 1
        iter += 1

    logging.debug(f"Number: {init_number}, ternary: {ternary}")
    return tuple(tern for tern in ternary)


def to_number(ternary: Tuple[int, ...]) -> int:
    """Convert ternary list to decimal number. Shift -1,0,1 to 0,1,2.

    e.g weights = 3 weighings = 4 ternary = [-1,-1,0,1], num is (1+1)*3^0+(0+1)*3^1+(-1+1)*3^2+(-1+1)*3^3=5
    """
    return sum(
        (len(Weight) ** power) * (digit + 1)
        for power, digit in enumerate(reversed(ternary))
    )


def verify_unique_ternary(unique_ternary: List[Tuple[int, ...]]) -> bool:
    """If sum for each column is 0 or sum of marble weights for each weighing, each weighing has equal number per side, making ternary unique."""
    unique_tern_np = np.array(unique_ternary)
    checksum = np.zeros(unique_tern_np.shape[1]).astype(int)
    actualsum = np.sum(unique_tern_np, axis=0).astype(int)
    logging.debug(f"Actual sum per weight: {actualsum}")
    return (actualsum == checksum).all()


def map_unique_ternary(
    unique_ternary: List[Tuple[int, ...]], num_marbles: int, weighings: int
) -> dict[Tuple[int, ...], int]:
    """
    Creates dictionary of proper mapping of marble num/state to unique ternary.

    Randomly gets 2N out of K possible weighings where N is complimentary to other N to balance out L/R
    If ternary subset is balanced, check that map is one to one by putting keys to dictionary.
    If no duplicates, dictionary size num stays with ternary keys non-repeating as they are added to dictionary.
    """
    ternary_map = {}
    num_indices = len(Weight) ** weighings
    while True:
        heavy_ternary_idx, light_ternary_idx = [], []

        # TODO algorithm to search this space more efficiently instead of random

        for marble in range(num_marbles):
            while True:
                idx = random.randint(0, num_indices - 1)
                comp_idx = num_indices - idx - 1
                # No repeats
                if idx not in heavy_ternary_idx and idx not in light_ternary_idx:
                    heavy_ternary_idx.append(idx)
                    light_ternary_idx.append(comp_idx)
                    break

        heavy_ternary = [unique_ternary[heavy_idx] for heavy_idx in heavy_ternary_idx]
        light_ternary = [unique_ternary[light_idx] for light_idx in light_ternary_idx]

        logging.debug("Try again")
        logging.debug(
            f"Heavy ternary idx {heavy_ternary_idx}\nLight ternary idx {light_ternary_idx}"
        )
        logging.debug(f"Ternary {unique_ternary}")
        logging.debug(f"Heavy ternary: {heavy_ternary}\n Light ternary {light_ternary}")

        if verify_unique_ternary(heavy_ternary):
            for marble in range(num_marbles):
                key = heavy_ternary[marble]
                ternary_map[key] = marble
                key = light_ternary[marble]
                ternary_map[key] = marble + num_marbles
            if len(ternary_map) == num_marbles * 2:
                break
    return ternary_map


def get_sets_from_ternary(
    ternary_map: dict[Tuple[int, ...], int], weigh_num: int, num_marbles: int
) -> tuple[list[int], list[int]]:
    """
    Extracts left and right side of scale based on ternary placement value.

    Weighing is ternary list place index, 1 or heavy is left, -1 or light is right side of scale.
    """
    left_side, right_side = [], []
    for key in ternary_map.keys():
        if ternary_map[key] < num_marbles:
            if key[weigh_num] == Weight.HEAVY:
                left_side.append(ternary_map[key])
            if key[weigh_num] == Weight.LIGHT:
                right_side.append(ternary_map[key])
    logging.debug(f"ternary map: {ternary_map}")
    logging.debug(f"Left side indices: {left_side}, \nRight side indics: {right_side}")
    return left_side, right_side


def index_to_sides(all_marbles: List[Weight], indices: list[int]) -> List[Weight]:
    """Use list of indices for side to retrieve marbles for weighing.

    e.g all marbles = [EQUAL,HEAVY,EQUAL,EQUAL,EQUAL], indices = [0,1,3] return [EQUAL,HEAVY,EQUAL]
    """
    return [all_marbles[index] for index in indices]


def get_actual_weights(
    all_marbles: List[Weight],
    num_marbles: int,
    weighings: int,
    ternary_map: dict[Tuple[int, ...], int],
) -> List[int]:
    """Perform weighings based on unique ternary and mapping to marbles."""
    actual_weights = []
    for weigh in range(weighings):
        li, ri = get_sets_from_ternary(ternary_map, weigh, num_marbles)
        ls = index_to_sides(all_marbles, li)
        rs = index_to_sides(all_marbles, ri)
        w = weighing(ls, rs)
        actual_weights.append(w)
    return actual_weights


def main():
    """Main function."""
    for misplace_marble in range(0, NUM_MARBLES):
        for actual_weight in [Weight.LIGHT, Weight.HEAVY]:
            all_marbles = setup_marbles(NUM_MARBLES, misplace_marble, actual_weight)
            logging.debug(f"All marbles: {all_marbles}")
            weighings = get_weighings(NUM_MARBLES)
            logging.debug(f"Number of weighings {weighings}")

            ternary = get_ternary(weighings)
            if misplace_marble == 0 and actual_weight == Weight.LIGHT:
                logging.debug(
                    "Need to search for mapping between marble number/weight and all weightings."
                )
                ternary_map = map_unique_ternary(ternary, NUM_MARBLES, weighings)
            actual_weights = get_actual_weights(
                all_marbles, NUM_MARBLES, weighings, ternary_map
            )
            logging.debug(f"Actual weights {actual_weights}")

            dict_num = ternary_map[tuple(actual_weights)]
            marble_num = dict_num if dict_num < NUM_MARBLES else dict_num % NUM_MARBLES
            marble_off = Weight.HEAVY if dict_num < NUM_MARBLES else Weight.LIGHT
            logging.info(
                f"Actual marble off: {misplace_marble} Guess marble {marble_num} Actually {actual_weight} Guessed {marble_off}"
            )


if __name__ == "__main__":
    main()
