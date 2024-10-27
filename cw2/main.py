from concurrent.futures import ProcessPoolExecutor
from typing import Callable

import numpy as np
import pandas as pd
from cec2017.functions import f2, f13
from constants import DIMENSIONALITY, MAX_X, TABLE_DIR
from evolutionary import evolutionary
from numpy.typing import NDArray


def format_float_with_comma(value: float, precision: int = 2):
    return f"{value:.{precision}f}".replace(".", ",")


def generate_param_string(mu: int, sigma: float):
    return f"\u03bc={mu} \u03c3={format_float_with_comma(sigma, 1)}"


def format_params(mu_list: list[int], sigma_list: list[float]):
    formatted: list[str] = []
    for mu in mu_list:
        for sigma in sigma_list:
            formatted.append(generate_param_string(mu, sigma))

    return formatted


def run_simulation(
    function: callable, sigma: float, population: NDArray[NDArray[float]]
) -> (float, float):
    optimum_10k = evolutionary(function, sigma, population.copy())
    optimum_50k = evolutionary(function, sigma, population.copy(), eval_limit=50000)

    return optimum_10k, optimum_50k


def generate_data(function: Callable, tries: int):
    print(function.__name__)
    mu_list: list[int] = [2**i for i in range(1, 7)]
    sigma_list: list[float] = [i / 10 for i in range(1, 31)]

    stats_10k: dict[str, list[str]] = {
        "parametry": format_params(mu_list, sigma_list),
        "min": [],
        "śr": [],
        "std": [],
        "max": [],
    }

    stats_50k: dict[str, list[str]] = {
        "parametry": format_params(mu_list, sigma_list),
        "min": [],
        "śr": [],
        "std": [],
        "max": [],
    }

    for mu in mu_list:
        population: NDArray[NDArray[float]] = np.random.uniform(
            -MAX_X, MAX_X, (mu, DIMENSIONALITY)
        )
        for sigma in sigma_list:
            results_10k = []
            results_50k = []
            with ProcessPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(run_simulation, function, sigma, population.copy())
                    for _ in range(tries)
                ]

                for future in futures:
                    results_10k.append(future.result()[0])
                    results_50k.append(future.result()[1])
            # for i in range(tries):
            #     optimums = run_simulation(function, sigma, population.copy())
            #     results_10k.append(optimums[0])
            #     results_50k.append(optimums[1])

            for stats, results in zip(
                [stats_10k, stats_50k], [results_10k, results_50k]
            ):
                stats["min"].append(format_float_with_comma(np.min(results)))
                stats["śr"].append(format_float_with_comma(np.mean(results)))
                stats["std"].append(format_float_with_comma(np.std(results)))
                stats["max"].append(format_float_with_comma(np.max(results)))

            print("finished ", generate_param_string(mu, sigma))

    if not TABLE_DIR.exists():
        TABLE_DIR.mkdir(exist_ok=True)

    pd.DataFrame(stats_10k).to_csv(
        TABLE_DIR / f"{function.__name__}_10k.csv", index=False
    )
    pd.DataFrame(stats_50k).to_csv(
        TABLE_DIR / f"{function.__name__}_50k.csv", index=False
    )


if __name__ == "__main__":
    generate_data(f2, 100)
    generate_data(f13, 100)
