import numpy as np

from cec2017.functions import f2, f13

EVALUATION_LIMIT = 10000


if __name__ == "__main__":
    x = np.random.uniform(-10, 10, size=10)
    print(f13(x))
