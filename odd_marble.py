import logging
import math
import random
import sys
from typing import Any

import numpy as np

logging.basicConfig(
    format="%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
    stream=sys.stdout,
    level=logging.INFO,
    filemode="w+",
)


def setup_marbles(
    num_marbles: int, actual_marble: int, actual_weight: str
) -> np.ndarray:
    """Make all marbles include placing oddball one in designated place."""
    all_marbles = np.array(["equal" for val in range(num_marbles)])
    if actual_marble < 0 or actual_marble > num_marbles - 1:
        raise ValueError(f"Specify odd marble index between 0 and {num_marbles - 1}")
    if actual_weight != "light" and actual_weight != "heavy":
        raise ValueError("Specify odd marble either 'l' for lighter or 'h' for heavier")
    all_marbles[actual_marble] = actual_weight
    logging.debug(f"All marbles: {all_marbles}")
    return all_marbles


def get_weighings(num_marbles: int) -> int:
    """Calculate max weighings needed based on number of marbles."""
    num_weighings = math.ceil(math.log(num_marbles * 2, 3))
    logging.debug(f"Number of weighings {num_weighings}")
    return num_weighings


def weighing(s1: np.ndarray, s2: np.ndarray):
    """0 is equal weigh, 1 is s1 heavier, -1 is s2 heavier."""
    if len(s1) != len(s2):
        raise ValueError("Invalid weighing! Use same amount on each side.")
    marble_vals = {
        key: val for key, val in zip(["light", "equal", "heavy"], [-1, 0, 1])
    }
    s1_sum = sum([marble_vals[marble] for marble in s1])
    s2_sum = sum([marble_vals[marble] for marble in s2])
    logging.debug(
        f"left: {s1} left sum: {s1_sum} left len: {len(s1)} \n right: {s2} right sum: {s2_sum} right len {len(s2)} \n weight {s1_sum-s2_sum}"
    )
    return s1_sum - s2_sum


def index_to_sides(all_marbles: np.ndarray, indices: list[int]) -> np.ndarray:
    """Use list of indices for side to retrieve marbles for weighing."""
    logging.debug(f"Marbles on side: {all_marbles[indices]}")
    return all_marbles[indices]


def to_ternary(number: int, max_base: int) -> np.ndarray:
    """Convert decimal number to ternary list of trites."""
    ternary_list = []
    remainder = number
    for base in range(max_base, 0, -1):
        digit = math.floor(remainder / (3 ** (base - 1)))
        ternary_list.append(digit - 1)
        remainder -= digit * (3 ** (base - 1))
    logging.debug(f"Number: {number}, ternary: {ternary_list}")
    return np.array(ternary_list)


def to_number(ternary: list[int]) -> int:
    """Convert ternary list to decimal number. Shift -1,0,1 to 0,1,2."""
    val = 0
    ternary_list = list(ternary)
    ternary_list.reverse()
    for power, digit in enumerate(ternary_list):
        digit_val = digit + 1
        val += (3**power) * digit_val
    logging.debug(f"Ternary {ternary_list}, value: {val}")
    return val


def verify_unique_ternary(unique_ternary: np.ndarray) -> bool:
    """If sum for each column is 0, each weighing has equal number per side, making ternary unique."""
    checksum = np.zeros(unique_ternary.shape[1]).astype(int)
    actualsum = np.sum(unique_ternary, axis=0).astype(int)
    logging.debug(f"Actual sum per weight: {actualsum}")
    return (actualsum == checksum).all()


def get_ternary(weighings: int) -> np.ndarray:
    """Get full ternary of possible weigh"""
    min_ternary = 0
    max_ternary = 3**weighings
    return np.array(
        [to_ternary(val, weighings) for val in range(min_ternary, max_ternary)]
    )


def map_unique_ternary(
    unique_ternary: np.ndarray, num_marbles: int, weighings: int
) -> dict[tuple[Any, ...], int]:
    """
    Creates dictionary of proper mapping of marble num/state to unique ternary.

    Randomly gets 2N out of K possible weighings where N is complimentary to other N to balance out L/R
    If ternary subset is balanced, check that map is one to one by putting keys to dictionary.
    If no duplicates, dictionary size num stays with ternary keys non-repeating as they are added to dictionary.
    """
    ternary_map = {}
    while True:
        heavy_ternary_idx, light_ternary_idx = [], []

        # TODO algorithm to search this space more efficiently instead of random
        for marble in range(num_marbles):
            while True:
                idx = random.randint(0, 3**weighings - 1)
                comp_idx = (3**weighings) - idx - 1
                # No repeats
                if (
                    idx not in heavy_ternary_idx and idx not in light_ternary_idx
                ):  # and idx != num_marbles:
                    heavy_ternary_idx.append(idx)
                    light_ternary_idx.append(comp_idx)
                    break

        logging.debug("Try again")
        logging.debug(
            f"Heavy ternary idx {heavy_ternary_idx}\nLight ternary idx {light_ternary_idx}"
        )
        logging.debug(f"Ternary {unique_ternary}")
        heavy_ternary = unique_ternary[heavy_ternary_idx]
        light_ternary = unique_ternary[light_ternary_idx]
        logging.debug(f"Heavy ternary: {heavy_ternary}\n Light ternary {light_ternary}")

        if verify_unique_ternary(
            heavy_ternary
        ):  # and verify_unique_ternary(light_ternary):
            for marble in range(num_marbles):
                key = tuple(heavy_ternary[marble])
                ternary_map[key] = marble
                key = tuple(light_ternary[marble])
                ternary_map[key] = marble + num_marbles
            if (
                len(ternary_map) == num_marbles * 2
            ):  # and len(set(ternary_map.values())) == num_marbles*2:
                break
    return ternary_map


def get_sets_from_ternary(
    ternary_map: dict[tuple[int], int], weigh_num: int, num_marbles: int
) -> tuple[list[int], list[int]]:
    """
    Extracts left and right side of scale based on ternary placement value.

    Weighing is ternary list place index, 1 is left, -1 is right side of scale.
    """
    left_side, right_side = [], []
    for key in ternary_map.keys():
        if ternary_map[key] < num_marbles:
            if key[weigh_num] == 1:
                left_side.append(ternary_map[key])
            if key[weigh_num] == -1:
                right_side.append(ternary_map[key])
    logging.debug(f"ternary map: {ternary_map}")
    logging.debug(f"Left side indices: {left_side}, \nRight side indics: {right_side}")
    return left_side, right_side


def get_actual_weights(
    all_marbles: np.ndarray,
    num_marbles: int,
    weighings: int,
    ternary_map: dict[tuple[int], int],
):
    """Perform weighings based on unique ternary and mapping to marbles."""
    actual_weights = []
    for weigh in range(weighings):
        li, ri = get_sets_from_ternary(ternary_map, weigh, num_marbles)
        ls = index_to_sides(all_marbles, li)
        rs = index_to_sides(all_marbles, ri)
        w = weighing(ls, rs)
        actual_weights.append(w)
    logging.debug(f"Actual weights {actual_weights}")
    return np.array(actual_weights)


def main():
    num_marbles = 12
    for misplace_marble in range(0, num_marbles):
        for actual_weight in ["light", "heavy"]:
            all_marbles = setup_marbles(num_marbles, misplace_marble, actual_weight)
            weighings = get_weighings(num_marbles)

            ternary = get_ternary(weighings)
            if misplace_marble == 0 and actual_weight == "light":
                logging.debug(
                    "Need to search for mapping between marble number/weight and all weightings."
                )
                ternary_map = map_unique_ternary(ternary, num_marbles, weighings)
            actual_weights = get_actual_weights(
                all_marbles, num_marbles, weighings, ternary_map
            )

            dict_num = ternary_map[tuple(actual_weights)]
            marble_num = dict_num if dict_num < num_marbles else dict_num % num_marbles
            marble_off = "heavy" if dict_num < num_marbles else "light"
            logging.info(
                f"Actual marble off: {misplace_marble} Guess marble {marble_num} Actually {actual_weight} Guessed {marble_off}"
            )


if __name__ == "__main__":
    main()
