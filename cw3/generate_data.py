import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
from checkers_stud import (
    MINIMAX_DEPTH,
    ai_vs_ai,
    basic_ev_func,
    group_prize_ev_func,
    push_forward_ev_func,
    push_to_opp_half_ev_func,
)
from matplotlib import pyplot as plt

# num_workers = max(1, os.cpu_count() // 2)
num_workers = 14


def generate_ev_function_data():
    tries = 50
    ev_functions = [
        basic_ev_func,
        group_prize_ev_func,
        push_to_opp_half_ev_func,
        push_forward_ev_func,
    ]

    col_count = len(ev_functions)

    plot_path = Path(__file__).parent / "plots"
    plot_path.mkdir(exist_ok=True)

    x = np.arange(col_count)
    categories = ["basic", "group", "opp_half", "forward"]

    bottom_count = np.zeros(col_count)
    mid_count = np.zeros(col_count)
    top_count = np.zeros(col_count)

    for i, function in enumerate(ev_functions):
        print(f"Analyzing {function.__name__} results")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(ai_vs_ai, function) for _ in range(tries)]

            for future in futures:
                if future.result() == [True, True]:
                    mid_count[i] += 1
                elif future.result()[0]:
                    bottom_count[i] += 1
                else:
                    top_count[i] += 1

    plt.bar(x, bottom_count, label="Biały", color="b")
    plt.bar(x, mid_count, bottom=bottom_count, label="Remis", color="g")
    plt.bar(
        x,
        top_count,
        bottom=bottom_count + mid_count,
        label="Czarny",
        color="r",
    )

    plt.xlabel("Funkcja oceny")
    plt.ylabel("Wynik gry")
    plt.legend()

    plt.xticks(x, categories)
    plt.savefig(plot_path / "ev_func_impact.png")


def generate_depth_data(ev_func: Callable):
    tries = 50
    extra_depths = [-2, -1, 0, 1]

    plot_path = Path(__file__).parent / "plots"
    plot_path.mkdir(exist_ok=True)

    col_count = len(extra_depths)

    bottom_count = np.zeros(col_count)
    mid_count = np.zeros(col_count)
    top_count = np.zeros(col_count)

    x = np.arange(col_count)
    categories = [MINIMAX_DEPTH + extra for extra in extra_depths]

    for i, extra_depth in enumerate(extra_depths):
        print(f"Analyzing results for depth={MINIMAX_DEPTH + extra_depth}")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(ai_vs_ai, ev_func, extra_depth) for _ in range(tries)
            ]

            for future in futures:
                if future.result() == [True, True]:
                    mid_count[i] += 1
                elif future.result()[0]:
                    bottom_count[i] += 1
                else:
                    top_count[i] += 1

    plt.bar(x, bottom_count, label="Biały", color="b")
    plt.bar(x, mid_count, bottom=bottom_count, label="Remis", color="g")
    plt.bar(
        x,
        top_count,
        bottom=bottom_count + mid_count,
        label="Czarny",
        color="r",
    )

    plt.xlabel("Głębokość drzewa przeszukiwań")
    plt.ylabel("Wynik gry")
    plt.legend()

    plt.xticks(x, categories)
    plt.savefig(plot_path / f"depth_impact_{ev_func.__name__}.png")
    plt.close()


if __name__ == "__main__":
    # generate_ev_function_data()
    generate_depth_data(basic_ev_func)
    generate_depth_data(group_prize_ev_func)
    generate_depth_data(push_to_opp_half_ev_func)
    generate_depth_data(push_forward_ev_func)
