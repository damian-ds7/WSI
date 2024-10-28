from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Union

import numpy as np
import pandas as pd
from cec2017.functions import f2, f13
from constants import DIMENSIONALITY, EVALUATION_LIMIT, MAX_X, SYMBOLS, TABLE_DIR
from evolutionary import evolutionary
from numpy.typing import NDArray


def format_float_with_comma(value: float, precision: int = 2):
    return f"{value:.{precision}f}".replace(".", ",")


def format_params(
    params: list[Union[str, float]], name: str, round_param: bool
) -> list[str]:
    formatted_params: list[str] = []

    for param in params:
        str_param = SYMBOLS[name] + "="
        str_param += (
            str(param) if not round_param else format_float_with_comma(param, 2)
        )
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
    print(f"analyzing mu impact for f2, {SYMBOLS["sigma"]}={sigma}")
    formatted_mu: list[str] = format_params(mu_list, "mu", False)

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
        with ProcessPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(run_simulation, function, sigma, population.copy())
                for _ in range(tries)
            ]

            for future in futures:
                results_10k.append(future.result()[0])
                results_50k.append(future.result()[1])

        for stat, result in zip([stats_10k, stats_50k], [results_10k, results_50k]):
            stat["min"].append(np.min(result))
            stat["śr"].append(np.mean(result))
            stat["std"].append(np.std(result))
            stat["max"].append(np.max(result))

        print(f"finished {SYMBOLS["mu"]}={mu}")

    filename: str = f"{function.__name__}_mu_impact_sigma_{sigma}"

    pd.DataFrame(stats_10k).to_csv(TABLE_DIR / (filename + "_10k.csv"), index=False)
    pd.DataFrame(stats_50k).to_csv(TABLE_DIR / (filename + "_50k.csv"), index=False)


if __name__ == "__main__":
    analyze_mu_impact(f2, 3, [2**i for i in range(0, 7)], 1000)
