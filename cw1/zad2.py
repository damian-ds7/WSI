import numpy as np
from cec2017.functions import f1
from autograd import grad
import matplotlib.pyplot as plt


def draw_arrow(xi, gradient, max_x):
    scale = 0.1
    x, y = xi

    # calculate end of the arrow and reduce to bounds
    end_x = x + gradient[0] * scale
    end_y = y + gradient[1] * scale

    end_x = np.clip(end_x, -10, 10)
    end_y = np.clip(end_y, -10, 10)

    plt.arrow(
        *xi, end_x - x, end_y - y, head_width=0.3, head_length=0.3, fc="red", ec="red"
    )


def steepest_ascent(
    xi, f, beta, minimum=False, epsilon=1e-6, draw_arrows=False, max_x=10
):
    """
    Steepest ascent method
    :param x: starting point (can be an array of points)
    :param f: function to analyze
    :param beta: step size
    :param epsilon: precision
    :return: optimum
    """

    grad_fct = grad(f)
    gradient = grad_fct(xi)
    direction = -1 if minimum else 1

    while np.linalg.norm(gradient) > epsilon:
        xi = np.clip(-max_x, max_x, xi)
        if draw_arrows:
            draw_arrow(xi, gradient, max_x)

        xi += direction * gradient * beta
        gradient = grad_fct(xi)

    return xi, round(f(xi), 6)


def booth_function(x):
    x1, x2 = x
    return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2


def booth_optimum():
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
        0.1101,
        minimum=True,
        draw_arrows=True,
    )
    plt.show()


def main():
    # UPPER_BOUND = 100
    # DIMENSIONALITY = 2
    # x = np.random.uniform(-UPPER_BOUND, UPPER_BOUND, size=DIMENSIONALITY)

    # # wyznacz ocenę x
    # q = f1(x)
    # print("q(x) = %.6f" % q)

    # print(steepest_ascent(np.random.uniform(-10, 10, 2), booth, 0.1, minimum=True))
    booth_optimum()


if __name__ == "__main__":
    main()
