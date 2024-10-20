from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from autograd import grad
from cec2017.functions import f1, f2, f3


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

            if iteration_count % 1000 == 0:
                print(xi)
            iteration_count += 1

        return xi, self.f(xi), iteration_count

    @staticmethod
    def bb_method_step_increase(xi, previous_xi, gradient, previous_gradient):
        gradient_delta = gradient - previous_gradient
        return (np.dot(xi - previous_xi, gradient_delta)) / np.dot(
            gradient_delta, gradient_delta
        )

    def run(
        self,
        beta,
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
            beta: step size
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

        def run_iteration(i, xi):
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
                f"Starting Beta: {beta} | "
                f"Start: ({xi[dim_1]:.2f}, {xi[dim_2]:.2f}) | "
                f"End: ({end_xi[dim_1]:.2f}, {end_xi[dim_2]:.2f}) | "
                f"Optimum: {optimum:.2f}"
            )

            plt.figtext(0.51, 0.92 - (i * 0.05), text, ha="left", color=color)
            return f"Finishied point {i+1} with {iteration_count} iterations ({beta}{" beta increase" if beta_increase else ""})"

        with ThreadPoolExecutor(max_workers=tries) as executor:
            futures = [executor.submit(run_iteration, i, xi[i]) for i in range(tries)]
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


def booth_optimum(tries=1, plot_name=None):
    """
    Calculates and plots steps of steepest ascent method for chosen booth function

    Args:
        tries: number of to create and find optimum from. Defaults to 1.
        plot_name: file name to save plot, if none given plot will only be displayed. Defaults to None.
    """
    opt_search = OptimumSearch(booth_function, 2, 10)
    opt_search.run(0.05, plot_name=plot_name, tries=tries)


def f1_optimum():
    tries = 5
    xi = np.random.uniform(-100, 100, (tries, 10))
    cec_optimum = OptimumSearch(f1, 10, 100, grad_epsilon=1 + 8e-11)

    def part_1():
        cec_optimum.run(
            1e-8,
            tries=tries,
            xi=xi.copy(),
            dim_1=3,
            dim_2=4,
            plot_name="plots/f1/f1_x3x4_1e-8.png",
            iteration_limit=40000,
        )
        cec_optimum.run(
            1e-8,
            tries=tries,
            xi=xi.copy(),
            dim_1=0,
            dim_2=1,
            plot_name="plots/f1/f1_x0x1_1e-8.png",
            iteration_limit=40000,
        )

    def part_2():
        cec_optimum.run(
            1e-8,
            tries=tries,
            xi=xi.copy(),
            dim_1=3,
            dim_2=4,
            plot_name="plots/f1/f1_x3x4_1e-8_bb.png",
            beta_increase=True,
            iteration_limit=40000,
        )
        cec_optimum.run(
            1e-8,
            tries=tries,
            xi=xi.copy(),
            dim_1=0,
            dim_2=1,
            plot_name="plots/f1/f1_x0x1_1e-8_bb.png",
            beta_increase=True,
            iteration_limit=40000,
        )

    def part_3():
        cec_optimum.run(
            1e-4,
            tries=tries,
            xi=xi.copy(),
            dim_1=3,
            dim_2=4,
            plot_name="plots/f1/f1_x3x4_1e-4.png",
            iteration_limit=40000,
        )
        cec_optimum.run(
            1e-4,
            tries=tries,
            xi=xi.copy(),
            dim_1=3,
            dim_2=4,
            plot_name="plots/f1/f1_x3x4_1e-4_bb.png",
            beta_increase=True,
            iteration_limit=40000,
        )

    def part_4():
        cec_optimum.run(
            1e-15,
            tries=tries,
            xi=xi.copy(),
            dim_1=3,
            dim_2=4,
            plot_name="plots/f1/f1_x3x4_1e-15.png",
            iteration_limit=40000,
        )
        cec_optimum.run(
            1e-15,
            tries=tries,
            xi=xi.copy(),
            dim_1=3,
            dim_2=4,
            plot_name="plots/f1/f1_x0x1_1e-15_bb.png",
            beta_increase=True,
            iteration_limit=40000,
        )

    part_1()
    part_2()
    part_3()
    part_4()


def f2_optimum():
    tries = 5
    xi = np.random.uniform(-100, 100, (tries, 10))
    cec_optimum = OptimumSearch(f2, 10, 100, grad_epsilon=1 + 8e-11)

    def part_1():
        cec_optimum.run(
            1e-20,
            tries=tries,
            xi=xi.copy(),
            dim_1=3,
            dim_2=9,
            plot_name="plots/f2/f2_x3x9_1e-20.png",
            iteration_limit=40000,
        )
        cec_optimum.run(
            1e-20,
            tries=tries,
            xi=xi.copy(),
            dim_1=3,
            dim_2=9,
            plot_name="plots/f2/f2_x3x9_1e-20_bb.png",
            iteration_limit=40000,
            beta_increase=True,
        )

    def part_2():
        cec_optimum.run(
            1e-27,
            tries=tries,
            xi=xi.copy(),
            dim_1=3,
            dim_2=9,
            plot_name="plots/f2/f2_x3x9_1e-27.png",
            iteration_limit=40000,
        )
        cec_optimum.run(
            1e-27,
            tries=tries,
            xi=xi.copy(),
            dim_1=3,
            dim_2=9,
            plot_name="plots/f2/f2_x3x9_1e-27_bb.png",
            iteration_limit=40000,
            beta_increase=True,
        )

    def part_3():
        cec_optimum.run(
            1e-15,
            tries=tries,
            xi=xi.copy(),
            dim_1=3,
            dim_2=9,
            plot_name="plots/f2/f2_x3x9_1e-15.png",
            iteration_limit=40000,
        )
        cec_optimum.run(
            1e-15,
            tries=tries,
            xi=xi.copy(),
            dim_1=3,
            dim_2=9,
            plot_name="plots/f2/f2_x3x9_1e-15_bb.png",
            iteration_limit=40000,
            beta_increase=True,
        )

    part_1()
    part_2()
    part_3()


def f3_optimum():
    tries = 5
    xi = np.random.uniform(-100, 100, (tries, 10))
    cec_optimum = OptimumSearch(f3, 10, 100, grad_epsilon=1 + 8e-11)

    def part_1():
        cec_optimum.run(
            1e-20,
            tries=tries,
            xi=xi.copy(),
            dim_1=6,
            dim_2=8,
            plot_name="plots/f3/f3_x6x8_1e-20.png",
            iteration_limit=40000,
        )
        cec_optimum.run(
            1e-20,
            tries=tries,
            xi=xi.copy(),
            dim_1=6,
            dim_2=8,
            plot_name="plots/f3/f3_x6x8_1e-20_bb.png",
            beta_increase=True,
            iteration_limit=40000,
        )

    def part_2():
        cec_optimum.run(
            1e-15,
            tries=tries,
            xi=xi.copy(),
            dim_1=6,
            dim_2=8,
            plot_name="plots/f3/f3_x6x8_1e-15.png",
            iteration_limit=40000,
        )
        cec_optimum.run(
            1e-15,
            tries=tries,
            xi=xi.copy(),
            dim_1=6,
            dim_2=8,
            plot_name="plots/f3/f3_x6x8_1e-15_bb.png",
            beta_increase=True,
            iteration_limit=40000,
        )

    def part_3():
        cec_optimum.run(
            1e-10,
            tries=tries,
            xi=xi.copy(),
            dim_1=6,
            dim_2=8,
            plot_name="plots/f3/f3_x6x8_1e-10.png",
            iteration_limit=40000,
        )
        cec_optimum.run(
            1e-10,
            tries=tries,
            xi=xi.copy(),
            dim_1=6,
            dim_2=8,
            plot_name="plots/f3/f3_x6x8_1e-10_bb.png",
            beta_increase=True,
            iteration_limit=40000,
        )

    part_1()
    part_2()
    part_3()


if __name__ == "__main__":
    # booth_optimum()
    # f1_optimum()
    # f2_optimum()
    f3_optimum()
