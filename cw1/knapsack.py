import heapq

import numpy as np


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


if __name__ == "__main__":
    counter = 0
    while True:
        weights = np.random.randint(1, 1000, 15)
        M = np.sum(weights) / 2
        values = np.random.randint(1, 2000, 15)
        bf = knapsack_brute_force(weights, values, M)
        h = knapsack_heuristic(weights, values, M)
        if bf != h:
            print(bf)
            print(h)
            print(values / weights)
            break
        counter += 1

    print(counter)

    # import timeit

    # print(
    #     timeit.timeit(
    #         "knapsack_brute_force(weights, values, M)", globals=globals(), number=1
    #     )
    # )
    # print(
    #     timeit.timeit(
    #         "knapsack_heuristic(weights, values, M)", globals=globals(), number=1
    #     )
    # )

    # weights = np.array([8, 3, 5, 2])
    # M = np.sum(weights) / 2
    # values = np.array([16, 8, 9, 6])

    # print(knapsack_heuristic(weights, values, M))
