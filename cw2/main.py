import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Union

import numpy as np
import pandas as pd
from cec2017.functions import f2, f13
from constants import DIMENSIONALITY, EVALUATION_LIMIT, MAX_X, SYMBOLS, TABLE_DIR
from evolutionary import evolutionary
from numpy.typing import NDArray

num_workers = max(1, os.cpu_count() // 2)


def format_float_with_comma(value):
    return f"{value:.2f}".replace(".", ",")


def format_params(params: list[Union[str, float]], name: str) -> list[str]:
    formatted_params: list[str] = []

    for param in params:
        str_param = SYMBOLS[name] + "="
        str_param += str(param).replace(".", ",")
        formatted_params.append(str_param)

    return formatted_params


def run_simulation(
    function: callable, sigma: float, population: NDArray[NDArray[float]]
) -> (float, float):
    optimum_10k = evolutionary(function, sigma, population.copy())
    optimum_50k = evolutionary(
        function, sigma, population.copy(), eval_limit=5 * EVALUATION_LIMIT
    )

    return optimum_10k, optimum_50k


def analyze_mu_impact(function: Callable, sigma: float, mu_list: list[int], tries: int):
    print(f"analyzing mu impact for {function.__name__}, {SYMBOLS["sigma"]}={sigma}")
    formatted_mu: list[str] = format_params(mu_list, "mu")

    stats_10k: dict[str, list] = {
        "mu": formatted_mu,
        "min": [],
        "śr": [],
        "std": [],
        "max": [],
    }

    stats_50k: dict[str, list] = {
        "mu": formatted_mu,
        "min": [],
        "śr": [],
        "std": [],
        "max": [],
    }

    for mu in mu_list:
        results_10k: list[float] = []
        results_50k: list[float] = []

        population = np.random.uniform(-MAX_X, MAX_X, (mu, DIMENSIONALITY))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(run_simulation, function, sigma, population.copy())
                for _ in range(tries)
            ]

            for future in futures:
                results_10k.append(future.result()[0])
                results_50k.append(future.result()[1])

        for stat, result in zip([stats_10k, stats_50k], [results_10k, results_50k]):
            stat["min"].append(format_float_with_comma(np.min(result)))
            stat["śr"].append(format_float_with_comma(np.mean(result)))
            stat["std"].append(format_float_with_comma(np.std(result)))
            stat["max"].append(format_float_with_comma(np.max(result)))

        print(f"finished {SYMBOLS["mu"]}={mu}")

    filename: str = f"{function.__name__}_mu_impact_sigma_{sigma}"

    pd.DataFrame(stats_10k).to_csv(TABLE_DIR / (filename + "_10k.csv"), index=False)
    pd.DataFrame(stats_50k).to_csv(TABLE_DIR / (filename + "_50k.csv"), index=False)


def analyze_sigma_impact(
    function: Callable, mu: int, sigma_list: list[float], tries: int
):
    print(f"analyzing sigma impact for {function.__name__}, {SYMBOLS["mu"]}={mu}")
    formatted_sigma: list[str] = format_params(sigma_list, "sigma")

    stats_10k: dict[str, list] = {
        "sigma": formatted_sigma,
        "min": [],
        "śr": [],
        "std": [],
        "max": [],
    }

    stats_50k: dict[str, list] = {
        "sigma": formatted_sigma,
        "min": [],
        "śr": [],
        "std": [],
        "max": [],
    }

    population = np.random.uniform(-MAX_X, MAX_X, (mu, DIMENSIONALITY))
    for sigma in sigma_list:
        results_10k: list[float] = []
        results_50k: list[float] = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(run_simulation, function, sigma, population.copy())
                for _ in range(tries)
            ]

            for future in futures:
                results_10k.append(future.result()[0])
                results_50k.append(future.result()[1])

        for stat, result in zip([stats_10k, stats_50k], [results_10k, results_50k]):
            stat["min"].append(format_float_with_comma(np.min(result)))
            stat["śr"].append(format_float_with_comma(np.mean(result)))
            stat["std"].append(format_float_with_comma(np.std(result)))
            stat["max"].append(format_float_with_comma(np.max(result)))

        print(f"finished {SYMBOLS["sigma"]}={sigma}")

    filename: str = f"{function.__name__}_sigma_impact_mu_{mu}"

    pd.DataFrame(stats_10k).to_csv(TABLE_DIR / (filename + "_10k.csv"), index=False)
    pd.DataFrame(stats_50k).to_csv(TABLE_DIR / (filename + "_50k.csv"), index=False)


def generate_mu_impact_data():
    sigma_list: list[float] = [0.5, 1.5, 3]
    mu_list: list[int] = [2**i for i in range(0, 7)]
    tries: int = 300
    functions = [f2, f13]

    for f in functions:
        for sigma in sigma_list:
            analyze_mu_impact(f, sigma, mu_list, tries)


100


def generate_sigma_impact_data():
    mu_list: list[int] = [5, 10, 20]
    sigma_list: list[float] = [i / 10 for i in range(5, 31, 5)]
    tries: int = 300
    functions = [f2, f13]

    for f in functions:
        for mu in mu_list:
            analyze_sigma_impact(f, mu, sigma_list, tries)


if __name__ == "__main__":
    generate_mu_impact_data()
    generate_sigma_impact_data()
