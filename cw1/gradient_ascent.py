from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from autograd import grad
from cec2017.functions import f1, f2, f3

ITERATION_LIMIT = 40000


def format_float_with_comma(value):
    return f"{value:.2f}".replace(".", ",")


@dataclass
class OptimumSearch:
    """
    Class for searching for the optimum of a function using the steepest ascent method

    Attributes:
        f: function to optimize
        dimensionality: number of dimensions over which the function is evaluated
        max_x: inclusive bound [-max_x, max_x]
        grad_epsilon: precision for gradient norm. Defaults to 1e-6.
    """

    f: callable
    dimensionality: int
    max_x: int
    grad_epsilon: float = 1e-6

    def __post_init__(self):
        self.plot_step = self.max_x / 100
        self.colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

    def _draw_contour(self, dim_1=0, dim_2=1):
        x_arr = np.arange(-self.max_x, self.max_x, self.plot_step)
        y_arr = np.arange(-self.max_x, self.max_x, self.plot_step)
        X, Y = np.meshgrid(x_arr, y_arr)
        Z = np.empty(X.shape)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                xi = np.zeros(self.dimensionality)
                xi[dim_1] = X[i, j]
                xi[dim_2] = Y[i, j]
                Z[i, j] = self.f(xi)

        plt.contour(X, Y, Z, 20)
        plt.xlabel(f"X{dim_1}")
        plt.ylabel(f"X{dim_2}")
        plt.xlim(-self.max_x, self.max_x)
        plt.ylim(-self.max_x, self.max_x)

    def _draw_arrow(self, point_1, point_2, color, dim_1=0, dim_2=1):
        x, y = point_1[dim_1], point_1[dim_2]
        end_x, end_y = point_2[dim_1], point_2[dim_2]

        head_width = self.max_x / 100
        head_width += 0.9 * head_width

        head_length = self.max_x / 100
        head_length += 0.9 * head_length

        plt.arrow(
            x,
            y,
            end_x - x,
            end_y - y,
            head_width=head_width,
            head_length=head_length,
            fc=color,
            ec=color,
        )

    def _steepest_ascent(
        self,
        xi,
        beta,
        maximum=True,
        dim_1=0,
        dim_2=1,
        draw_arrows=True,
        iteration_limit=None,
        beta_increase=False,
        color=None,
    ):

        grad_f = grad(self.f)
        gradient = grad_f(xi)
        iteration_count = 0

        color = self.colors[0] if color is None else color

        while np.linalg.norm(gradient) > self.grad_epsilon and (
            iteration_limit is None or iteration_limit > iteration_count
        ):
            if maximum:
                gradient = -gradient

            previous_xi = xi.copy()
            xi -= gradient * beta
            xi = np.clip(xi, -self.max_x, self.max_x)

            self._draw_arrow(previous_xi, xi, color, dim_1, dim_2)

            previous_gradient = gradient.copy()
            gradient = grad_f(xi)

            if beta_increase:
                beta = self.bb_method_step_increase(
                    xi, previous_xi, gradient, previous_gradient
                )

            # if iteration_count % 1000 == 0:
            #     print(xi)

            iteration_count += 1

        return xi, self.f(xi), iteration_count

    @staticmethod
    def bb_method_step_increase(
        xi, previous_xi, gradient, previous_gradient, short=True
    ):
        gradient_delta = gradient - previous_gradient
        point_delta = xi - previous_xi
        if short:
            return np.dot(point_delta, gradient_delta) / np.dot(
                gradient_delta, gradient_delta
            )
        else:
            return np.dot(point_delta, point_delta) / np.dot(
                point_delta, gradient_delta
            )

    def run(
        self,
        beta=0.1,
        maximum=False,
        dim_1=0,
        dim_2=1,
        tries=1,
        xi=None,
        plot_name=None,
        iteration_limit=None,
        beta_increase=False,
    ):
        """
        Steepeset ascent method

        Args:
            beta: step size. Defaults to 0.1.
            maximum: set True if searching for function maximum. Defaults to False.
            dim_1: _descrithe index of the first dimension against which the plot is drawn. Defaults to 0.
            dim_2: the index of the second dimension against which the plot is drawn. Defaults to 1.
            tries: number of tries to find the optimum. Defaults to 1.
            xi: array of starting points with length equal to the number of tries, if None then random
                points are generated. Defaults to None.
            plot_name: filename for the plot, if None then the plot is shown,
                       file is saved in directory from which the script is run. Defaults to None.
            iteration_limit: maximum number of iterations, no limit if set to None. Defaults to None.
            beta_increase: set True if you want to use Barzilai-Borwein step size increase method. Defaults to False.
        """

        plt.figure(figsize=(13, 5))
        plt.subplots_adjust(right=0.5, left=0.08, top=0.95)

        self._draw_contour(dim_1, dim_2)

        if xi is None:
            xi = np.random.uniform(
                -self.max_x, self.max_x, (tries, self.dimensionality)
            )

        def start_ascent(i, xi):
            color = self.colors[i]
            end_xi, optimum, iteration_count = self._steepest_ascent(
                xi.copy(),
                beta,
                maximum,
                dim_1,
                dim_2,
                beta_increase=beta_increase,
                iteration_limit=iteration_limit,
                color=color,
            )

            text = (
                f"Starting Beta: {beta:.2e} | "
                f"Start: ({format_float_with_comma(xi[dim_1])}; {format_float_with_comma(xi[dim_2])}) | "
                f"End: ({format_float_with_comma(end_xi[dim_1])}; {format_float_with_comma(end_xi[dim_2])}) | "
                f"Optimum: {format_float_with_comma(optimum)}"
            )

            plt.figtext(0.51, 0.92 - (i * 0.05), text, ha="left", color=color)
            return f"Finishied point {i+1} with {iteration_count} iterations ({beta:.2e}{" beta increase" if beta_increase else ""})"

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(start_ascent, i, xi[i]) for i in range(tries)]
            for future in futures:
                print(future.result())

        if plot_name is not None:
            plt.savefig(plot_name)
            plt.close()
        else:
            plt.show()


def booth_function(x):
    x1, x2 = x
    return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2


def booth_optimum(beta, tries=1, plot_name=None):
    """
    Calculates and plots steps of steepest ascent method for chosen booth function

    Args:
        beta: step size
        tries: number of to create and find optimum from. Defaults to 1.
        plot_name: file name to save plot, if none given plot will only be displayed. Defaults to None.
    """

    print("booth")

    opt_search = OptimumSearch(booth_function, 2, 10)
    opt_search.run(
        beta, plot_name=plot_name, tries=tries, iteration_limit=ITERATION_LIMIT
    )


def f1_optimum():
    print("f1")

    tries = 5
    dim_1 = 0
    dim_2 = 1
    beta_values = [1e-4, 1e-8, 1e-15]

    xi = np.random.uniform(-100, 100, (tries, 10))
    cec_optimum = OptimumSearch(f1, 10, 100, grad_epsilon=1 + 8e-11)

    config = {
        "tries": tries,
        "dim_1": dim_1,
        "dim_2": dim_2,
        "iteration_limit": ITERATION_LIMIT,
    }

    for beta in beta_values:
        cec_optimum.run(
            beta,
            xi=xi.copy(),
            **config,
            plot_name=f"plots/f1/f1_x0x1_{beta}.png",
        )
        cec_optimum.run(
            beta,
            **config,
            xi=xi.copy(),
            plot_name=f"plots/f1/f1_x0x1_{beta}_bb.png",
            beta_increase=True,
        )


def f2_optimum():
    print("f2")

    tries = 5
    dim_1 = 3
    dim_2 = 9
    # beta_values = [1e-15, 1e-20, 1e-27]
    beta_values = [1e-27]

    xi = np.random.uniform(-100, 100, (tries, 10))
    cec_optimum = OptimumSearch(f2, 10, 100, grad_epsilon=1 + 8e-11)

    config = {
        "tries": tries,
        "dim_1": dim_1,
        "dim_2": dim_2,
        "iteration_limit": ITERATION_LIMIT,
    }

    for beta in beta_values:
        # cec_optimum.run(
        #     beta,
        #     xi=xi.copy(),
        #     **config,
        #     plot_name=f"plots/f2/f2_x3x9_{beta}.png",
        # )
        cec_optimum.run(
            beta,
            **config,
            xi=xi.copy(),
            plot_name=f"plots/f2/f2_x3x9_{beta}_bb.png",
            beta_increase=True,
        )


def f3_optimum():
    print("f3")

    tries = 5
    dim_1 = 6
    dim_2 = 8
    beta_values = [1e-5, 1e-10, 1e-15, 1e-20]

    xi = np.random.uniform(-100, 100, (tries, 10))
    cec_optimum = OptimumSearch(f3, 10, 100, grad_epsilon=1 + 8e-11)

    config = {
        "tries": tries,
        "dim_1": dim_1,
        "dim_2": dim_2,
        "iteration_limit": ITERATION_LIMIT,
    }

    for beta in beta_values:
        cec_optimum.run(
            beta,
            xi=xi.copy(),
            **config,
            plot_name=f"plots/f3/f3_x6x8_{beta}.png",
        )
        cec_optimum.run(
            beta,
            **config,
            xi=xi.copy(),
            plot_name=f"plots/f3/f3_x6x8_{beta}_bb.png",
            beta_increase=True,
        )


if __name__ == "__main__":
    # booth_optimum(0.01, tries=5, plot_name="plots/booth/booth_0,01.png")
    # booth_optimum(0.05, tries=5, plot_name="plots/booth/booth_0,05.png")
    # booth_optimum(0.1, tries=5, plot_name="plots/booth/booth_0,1.png")
    # booth_optimum(0.12, tries=5, plot_name="plots/booth/booth_oscillation.png")
    # booth_optimum(0.11, tries=5, plot_name="plots/booth/booth_0,11.png")
    # f3_optimum()
    f2_optimum()
    # f1_optimum()
