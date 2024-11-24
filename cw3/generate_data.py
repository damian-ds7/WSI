import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from checkers_stud import (
    ai_vs_ai,
    basic_ev_func,
    group_prize_ev_func,
    push_forward_ev_func,
    push_to_opp_half_ev_func,
)
from matplotlib import pyplot as plt

num_workers = max(1, os.cpu_count() // 2)


def generate_data():
    tries = 100
    ev_functions = [
        basic_ev_func,
        group_prize_ev_func,
        push_to_opp_half_ev_func,
        push_forward_ev_func,
    ]

    f_count = len(ev_functions)

    plot_path = Path(__file__).parent / "plots"
    plot_path.mkdir(exist_ok=True)

    x = np.arange(f_count)
    categories = ["basic", "group", "opp_half", "forward"]

    bottom_count = np.zeros(f_count)
    mid_count = np.zeros(f_count)
    top_count = np.zeros(f_count)

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
    plt.show()


if __name__ == "__main__":
    generate_data()
