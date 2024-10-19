from dataclasses import dataclass
import numpy as np
from cec2017.functions import f1, f2, f3
from autograd import grad
import matplotlib.pyplot as plt
from cycler import cycler


@dataclass
class OptimumSearch:
    """
    Class for searching for the optimum of a function using the steepest ascent method
    :param f: function to analyze
    :param dimensionality: number of dimensions over which the function is evaluated
    :param max_x: inclusive bound [-max_x, max_x]
    :param epsilon: precision for gradient norm
    """

    f: callable
    dimensionality: int
    max_x: int
    grad_epsilon: float = 1e-6

    def __post_init__(self):
        self.plot_step = self.max_x / 100
        self.color_cycler = cycler(
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"]
        )
        self.color_iter = iter(self.color_cycler)
        self.color = None

    def _get_color(self):
        try:
            return next(self.color_iter)["color"]
        except StopIteration:
            self.color_iter = iter(self.color_cycler)
            return next(self.color_iter)["color"]

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

    def _draw_arrow(self, point_1, point_2, dim_1=0, dim_2=1):
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
            fc=self.color,
            ec=self.color,
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
    ):
        grad_f = grad(self.f)
        gradient = grad_f(xi)
        iteration_count = 0

        while np.linalg.norm(gradient) > self.epsilon and iteration_limit > 0:
            if maximum:
                gradient = -gradient

            previous_xi = xi.copy()
            xi -= gradient * beta
            xi = np.clip(xi, -self.max_x, self.max_x)

            self._draw_arrow(previous_xi, xi, dim_1, dim_2)

            previous_gradient = gradient.copy()
            gradient = grad_f(xi)
            print(xi)
            iteration_limit -= 1

        return xi, self.f(xi)

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
        Steepest ascent method, starting point(s) are randomly generated
        :param beta: step size
        :key maximum: set True if searching for function maximum
        :key dim_1: the index of the first dimension against which the plot is drawn
        :key dim_2: the index of the second dimension against which the plot is drawn
        :key tries: number of tries to find the optimum
        :key xi: array of starting points with length equal to the number of tries, if None then random
                 points are generated
        :key plot_name: filename for the plot, if None then the plot is shown,
                        file is saved in directory from which the script is run
        :key iteration_limit: maximum number of iterations, no limit if set to None
        :key beta_increase: set True if you want to use Barzilai-Borwein step size increase method
        """

        plt.figure(figsize=(13, 5))
        plt.subplots_adjust(right=0.5, left=0.08, top=0.95)

        self._draw_contour(dim_1, dim_2)

        if xi is None:
            xi = np.random.uniform(
                -self.max_x, self.max_x, (tries, self.dimensionality)
            )

        for i, xi in enumerate(xi):
            self.color = self._get_color()
            end_xi, optimum = self._steepest_ascent(
                xi.copy(),
                beta,
                maximum,
                dim_1,
                dim_2,
                beta_increase=beta_increase,
                iteration_limit=iteration_limit,
            )

            text = (
                f"Starting Beta: {beta} | "
                f"Start: ({xi[dim_1]:.2f}, {xi[dim_2]:.2f}) | "
                f"End: ({end_xi[dim_1]:.2f}, {end_xi[dim_2]:.2f}) | "
                f"Optimum: {optimum:.2f}"
            )

            plt.figtext(0.51, 0.92 - (i * 0.05), text, ha="left", color=self.color)

        if plot_name is not None:
            plt.savefig(plot_name)
            plt.close()
        else:
            plt.show()

        self.color_iter = iter(self.color_cycler)


def booth_function(x):
    x1, x2 = x
    return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2


def booth_optimum(tries=1, plot_name=None):
    """
    Calculates and plots steps of steepest ascent method for chosen cec2017 function
    :key plot_name: file name to save plot, if none given plot will only be displayed
    """

    opt_search = OptimumSearch(booth_function, 2, 10)
    opt_search.run(0.05, plot_name=plot_name, tries=tries)


if __name__ == "__main__":
    # booth_optimum()
    cec_optimum = OptimumSearch(f1, 10, 100)
    cec_optimum.run(1e-8, tries=5, dim_1=3, dim_2=4)
