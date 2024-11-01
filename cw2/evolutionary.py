from typing import Callable

import numpy as np
from constants import DIMENSIONALITY, EVALUATION_LIMIT, MAX_X
from numpy.typing import NDArray


def selection(mu: int, values: NDArray[float]) -> NDArray[int]:
    # indices: NDArray[int] = np.zeros(mu, dtype="int64")

    indices: NDArray[int] = np.random.randint(0, mu, (mu, 2))

    selected: NDArray[int] = np.where(
        values[indices[:, 0]] < values[indices[:, 1]],
        indices[:, 0],
        indices[:, 1]
    )

    return selected

    # for i in range(mu):
    #     item_1, item_2 = np.random.randint(0, mu, size=2)

    #     if values[item_1] >= values[item_2]:
    #         indices[i] = item_2
    #     else:
    #         indices[i] = item_1

    return indices


def mutation(
    mu: int, sigma: float, population: NDArray[float]
) -> NDArray[NDArray[float]]:
    masks = sigma * np.random.normal(0, 1, (mu, DIMENSIONALITY))
    return (population + masks).clip(-MAX_X, MAX_X)


def evolutionary(
    function: Callable,
    sigma: float,
    population: NDArray[NDArray[float]],
    eval_limit=EVALUATION_LIMIT,
) -> float:
    mu: int = population.shape[0]
    iteration_limit: int = (eval_limit) // mu - 1
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
    from cec2017.functions import f2
    evolutionary(f2, 0.5, np.random.uniform(-100, 100, (20,)))