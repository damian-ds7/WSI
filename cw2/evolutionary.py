from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
from cec2017.functions import f2, f13
from numpy.typing import NDArray

EVALUATION_LIMIT: int = 10000
CURRENT_DIR: Path = Path(__file__).parent
MAX_X: int = 100
DIMENSIONALITY: int = 10


def selection(mu: int, values: NDArray[float]) -> NDArray[int]:
    indices: NDArray[int] = np.zeros(mu, dtype="int64")

    for i in range(mu):
        item_1, item_2 = np.random.randint(0, mu, size=2)

        if values[item_1] >= values[item_2]:
            indices[i] = item_2
        else:
            indices[i] = item_1

    return indices


def mutation(
    mu: int, sigma: float, population: NDArray[float]
) -> NDArray[NDArray[float]]:
    masks = sigma * np.random.uniform(-1, 1, (mu, DIMENSIONALITY))
    return (population + masks).clip(-MAX_X, MAX_X)


def evolutionary(
    function: Callable, sigma: float, population: NDArray[NDArray[float]]
) -> float:
    mu: int = population.shape[0]
    iteration_limit: int = (EVALUATION_LIMIT) // mu - 1
    values: NDArray[float] = np.apply_along_axis(function, 1, population)

    optimum = values.min()

    for i in range(iteration_limit):
        population = population[selection(mu, values)]
        population = mutation(mu, sigma, population)
        values: NDArray[float] = np.apply_along_axis(function, 1, population)

        temp_opt = values.min()

        optimum = temp_opt if temp_opt < optimum else optimum

    return optimum


if __name__ == "__main__":
    mu = 20
    # population = np.array([[4, 5], [1, 2]])
    population = np.random.uniform(-MAX_X, MAX_X, (mu, DIMENSIONALITY))
    opts = np.zeros(1000)
    # for i in range(1000):
    #     opts[i] = evolutionary(f2, 1.8, population.copy())
# [0.6, 1, 1.3, 1.5, 1.8, 2.1, 2.5, 2.8, 3]
    for sigma in range(10, 20):
        sigma /= 10
        with ProcessPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(evolutionary, f2, sigma, population.copy())
                for _ in range(1000)
            ]
            for i, future in enumerate(futures):
                opts[i] = future.result()

        print(f"sigma: {sigma}")
        print(opts.min())
        print(opts.mean())
        print(np.median(opts))
