from dataclasses import dataclass
import numpy as np
from cec2017.functions import f1, f2, f3
from autograd import grad
import matplotlib.pyplot as plt


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
    epsilon: float = 1e-6

    def __post_init__(self):
        self.plot_step = self.max_x / 100

    def _draw_contour(self, dim1=0, dim2=1):
        x_arr = np.arange(-self.max_x, self.max_x, self.plot_step)
        y_arr = np.arange(-self.max_x, self.max_x, self.plot_step)
        X, Y = np.meshgrid(x_arr, y_arr)
        Z = np.empty(X.shape)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                xi = np.zeros(self.dimensionality)
                xi[dim1] = X[i, j]
                xi[dim2] = Y[i, j]
                Z[i, j] = self.f(xi)

        plt.contour(X, Y, Z, 20)
        plt.colorbar(label="Function Value")
        plt.xlim(-self.max_x, self.max_x)
        plt.ylim(-self.max_x, self.max_x)

    def _draw_arrow(self, point_1, point_2, dim_1=0, dim_2=1):
        x, y = point_1[dim_1], point_1[dim_2]
        end_x, end_y = point_2[dim_1], point_2[dim_2]

        head_width = self.max_x / 100
        head_width += 0.75 * head_width

        head_length = self.max_x / 100
        head_length += 0.75 * head_length

        plt.arrow(
            x,
            y,
            end_x - x,
            end_y - y,
            head_width=head_width,
            head_length=head_length,
            fc="k",
            ec="k",
        )

    def _steepest_ascent(
        self, xi, beta, maximum=True, dim_1=0, dim_2=1, draw_arrows=True
    ):
        """
        Steepest ascent method
        :param xi: starting point
        :param beta: step size
        :key maximum: set True if searching for function maximum
        :key dim_1: the index of the first dimension against which the plot is drawn
        :key dim_2: the index of the second dimension against which the plot is drawn
        :key draw_arrows: set True if you want to draw arrows over function plot showing the path
        :return: optimum
        """

        grad_f = grad(self.f)
        gradient = grad_f(xi)
        iteration_limit = 1000

        while np.linalg.norm(gradient) > self.epsilon and iteration_limit > 0:
            if maximum:
                gradient = -gradient

            previous_xi = xi.copy()
            xi -= gradient * beta
            xi = np.clip(xi, -self.max_x, self.max_x)

            self._draw_arrow(previous_xi, xi, dim_1, dim_2)

            gradient = grad_f(xi)
            print(xi)
            iteration_limit -= 1

        return xi, round(self.f(xi), 6)

    def run(
        self,
        beta,
        maximum=False,
        dim_1=0,
        dim_2=1,
        tries=1,
        plot_name=None,
    ):
        """
        Steepest ascent method, starting point(s) are randomly generated
        :param beta: step size
        :key maximum: set True if searching for function maximum
        :key dim_1: the index of the first dimension against which the plot is drawn
        :key dim_2: the index of the second dimension against which the plot is drawn
        :key tries: number of tries to find the optimum
        :key plot_name: filename for the plot, if None then the plot is shown
        :return: optimum
        """

        optima = []

        self._draw_contour(dim_1, dim_2)

        for xi in np.random.uniform(
            -self.max_x, self.max_x, (tries, self.dimensionality)
        ):
            optima.append(self._steepest_ascent(xi, beta, maximum, dim_1, dim_2))

        if plot_name is not None:
            plt.savefig(plot_name)
            plt.close()
        else:
            plt.show()

        return optima


def booth_function(x):
    x1, x2 = x
    return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2


def booth_optimum(tries=1, plot_name=None):
    """
    Calculates and plots steps of steepest ascent method for chosen cec2017 function
    :key plot_name: file name to save plot, if none given plot will only be displayed
    """

    opt_search = OptimumSearch(booth_function, 2, 10)
    opt_search.run(0.01, plot_name=plot_name, tries=tries)


if __name__ == "__main__":
    # booth_optimum()
    cec_optimum = OptimumSearch(f1, 10, 100)
    cec_optimum.run(1e-8, tries=5)
