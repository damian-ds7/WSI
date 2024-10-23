import heapq
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd


def increment_bin_mask(mask, n):
    for i in reversed(range(n)):
        if mask[i] == 0:
            mask[i] = 1
            break
        else:
            mask[i] = 0
    return mask


def knapsack_brute_force(weights, values, M):
    n = len(weights)
    best_value = 0
    best_weight = 0
    best_mask = np.zeros(n, dtype=int)

    binary_mask = np.zeros(n, dtype=int)

    for i in range(2**n):
        weight = np.dot(weights, binary_mask)
        value = np.dot(values, binary_mask)

        if weight <= M and value > best_value:
            best_value = value
            best_weight = weight
            best_mask = binary_mask.copy()

        increment_bin_mask(binary_mask, n)

    return best_value, best_weight, np.array2string(best_mask, separator="")[1:-1]


def knapsack_heuristic(weights, values, M):
    n = len(weights)

    ratio_queue = []
    for i in range(n):
        ratio = values[i] / weights[i]
        heapq.heappush(ratio_queue, (-ratio, weights[i], values[i], i))

    best_weight = 0
    best_value = 0
    taken_items = [0] * n

    while ratio_queue and best_weight <= M:
        ratio, weight, value, index = heapq.heappop(ratio_queue)
        if best_weight + weight <= M:
            best_weight += weight
            best_value += value
            taken_items[index] = 1

    return best_value, best_weight, "".join(map(str, taken_items))


def run_simulation(element_count):
    weights = np.random.randint(1, 1000, element_count)
    M = np.sum(weights) / 2
    values = np.random.randint(1, 2000, element_count)

    time_start = time.process_time()
    knapsack_heuristic(weights, values, M)
    time_end = time.process_time()

    time_h = time_end - time_start

    time_start = time.process_time()
    knapsack_brute_force(weights, values, M)
    time_end = time.process_time()

    time_bf = time_end - time_start

    return time_h, time_bf


def format_float_with_comma(value):
    return f"{value:.6f}".replace(".", ",")


def create_tables(element_counts, tries):
    """
    Runs given heuristic and brute force knapsack methods and generates table files

    Args:
        element_counts: a list of different element times
        tries: number of times methods will be run for each element count
    """

    stats_h = {
        "l. elementów": element_counts,
        "min": [],
        "śr": [],
        "std": [],
        "max": [],
    }
    stats_bf = {
        "l. elementów": element_counts,
        "min": [],
        "śr": [],
        "std": [],
        "max": [],
    }

    for element_count in element_counts:
        times_h = []
        times_bf = []

        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(run_simulation, element_count) for _ in range(tries)
            ]
            for future in futures:
                times_h.append(future.result()[0])
                times_bf.append(future.result()[1])

        stats_h["min"].append(format_float_with_comma(np.min(times_h)))
        stats_h["śr"].append(format_float_with_comma(np.mean(times_h)))
        stats_h["std"].append(format_float_with_comma(np.std(times_h)))
        stats_h["max"].append(format_float_with_comma(np.max(times_h)))

        stats_bf["min"].append(format_float_with_comma(np.min(times_bf)))
        stats_bf["śr"].append(format_float_with_comma(np.mean(times_bf)))
        stats_bf["std"].append(format_float_with_comma(np.std(times_bf)))
        stats_bf["max"].append(format_float_with_comma(np.max(times_bf)))

        print(f"finished {element_count}")

    pd.DataFrame(stats_h).to_csv("tables/heuristic.csv")
    pd.DataFrame(stats_bf).to_csv("tables/bruteforce.csv")


if __name__ == "__main__":
    # # weights = np.array([8, 3, 5, 2]) #masa przedmiotów
    # # M = np.sum(weights)/2 #niech maksymalna masa plecaka będzie równa połowie masy przedmiotów
    # # values = np.array([16, 8, 9, 6]) #wartość przedmiotów

    # # print(knapsack_brute_force(weights, values, M))
    # # print(knapsack_heuristic(weights, values, M))

    # weights = np.random.randint(1, 1000, 25)
    # M = np.sum(weights) / 2
    # values = np.random.randint(1, 2000, 25)

    # print("{0:02f}s".format(timeit.timeit(lambda: knapsack_brute_force(weights, values, M), number=1)))

    create_tables([5, 10, 15, 20, 25], 50)