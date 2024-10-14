import numpy as np
from cec2017.functions import f1, f2, f3
from autograd import grad
import matplotlib.pyplot as plt


def draw_arrow(xi, gradient, max_x, dim1=0, dim2=1):
    x, y = xi[dim1], xi[dim2]

    # calculate end of the arrow and reduce to bounds
    end_x = x + gradient[dim1]
    end_y = y + gradient[dim2]

    end_x = np.clip(end_x, -max_x, max_x)
    end_y = np.clip(end_y, -max_x, max_x)

    plt.arrow(
        xi[dim1],
        xi[dim2],
        end_x - x,
        end_y - y,
        head_width=0.3,
        head_length=0.3,
        fc="red",
        ec="red",
    )


def steepest_ascent(
    xi,
    f,
    beta,
    minimum=False,
    epsilon=1e-6,
    draw_arrows=False,
    max_x=10,
    dim1=0,
    dim2=1,
):
    """
    Steepest ascent method
    :param xi: starting point
    :param f: function to analyze
    :param beta: step size
    :key minimum: set True if searching for function minimum
    :key epsilon: precision
    :key draw_arrows: set True if you want to draw arrows over function plot
    :key max_x: inclusive bound [-max_x, max_x]
    :key dim1: the index of the first dimension against which the arrows are drawn
    :key dim2: the index of the second dimension against which the arrows are drawn
    :return: optimum
    """

    grad_fct = grad(f)
    gradient = grad_fct(xi)
    direction = -1 if minimum else 1

    while np.linalg.norm(gradient) > epsilon:
        gradient = np.clip(gradient, -max_x, max_x)
        if draw_arrows:
            draw_arrow(xi, gradient, max_x)

        xi += direction * gradient * beta
        xi = np.clip(-max_x, max_x, xi)
        gradient = grad_fct(xi)

    return xi, round(f(xi), 6)


def booth_function(x):
    x1, x2 = x
    return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2


def booth_optimum(plot_name=None):
    """
    Calculates and plots steps of steepest ascent method for chosen cec2017 function
    :key plot_name: file name to save plot, if none given plot will only be displayed
    """
    MAX_X = 10
    PLOT_STEP = 0.1

    x_arr = np.arange(-MAX_X, MAX_X, PLOT_STEP)
    y_arr = np.arange(-MAX_X, MAX_X, PLOT_STEP)
    X, Y = np.meshgrid(x_arr, y_arr)
    Z = np.empty(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = booth_function(np.array([X[i, j], Y[i, j]]))

    plt.contour(X, Y, Z, 20)
    plt.xlim(-MAX_X, MAX_X)
    plt.ylim(-MAX_X, MAX_X)
    steepest_ascent(
        np.array([10, 10], dtype=float),
        # np.random.uniform(-MAX_X, MAX_X, 2),
        booth_function,
        0.1,
        minimum=True,
        draw_arrows=True,
    )

    if plot_name is None:
        plt.show()
    else:
        plt.savefig(plot_name)
    plt.clf()


def cec_optimum(f, dimensionality=10, dim1=0, dim2=1, plot_name=None):
    """
    Calculates and plots steps of steepest ascent method for chosen cec2017 function
    :param f: function to analyze
    :key dimensionality: number of dimensions over which the function is evaluated
    :key dim1: the index of the first dimension to plot the function against
    :key dim2: the index of the second dimension to plot the function against
    :key plot_name: file name to save plot, if no name is given plot will only be displayed
    """
    MAX_X = 100
    PLOT_STEP = 1

    x_arr = np.arange(-MAX_X, MAX_X, PLOT_STEP)
    y_arr = np.arange(-MAX_X, MAX_X, PLOT_STEP)
    X, Y = np.meshgrid(x_arr, y_arr)
    Z = np.empty(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            xi = np.zeros(10)
            xi[dim1] = X[i, j]
            xi[dim2] = Y[i, j]
            Z[i, j] = f(xi)

    plt.contour(X, Y, Z, 20)
    plt.xlim(-MAX_X, MAX_X)
    plt.ylim(-MAX_X, MAX_X)

    steepest_ascent(
        np.zeros(10, dtype=float),
        # np.random.uniform(-MAX_X, MAX_X, 10),
        f,
        1,
        max_x=MAX_X,
        draw_arrows=True,
        minimum=True,
        dim1=dim1,
        dim2=dim2,
    )

    if plot_name is None:
        plt.show()
    else:
        plt.savefig(plot_name)
    plt.clf()


def main():
    # UPPER_BOUND = 100
    # DIMENSIONALITY = 2
    # x = np.random.uniform(-UPPER_BOUND, UPPER_BOUND, size=DIMENSIONALITY)

    # # wyznacz ocenę x
    # q = f1(x)
    # print("q(x) = %.6f" % q)

    # print(steepest_ascent(np.random.uniform(-10, 10, 2), booth, 0.1, minimum=True))
    # booth_optimum()
    print(cec_optimum(f1))


if __name__ == "__main__":
    main()
