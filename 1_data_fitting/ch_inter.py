#!/usr/bin/python3

from math import cos, pi

import matplotlib.pyplot as plt
import numpy as np


# Function to interpolate
def f(x):
    return 1.0 / (1.0 + (25.0 * x * x))  # Runge's function
    # return abs(x)


# Function to evaluate the Lagrange interpolation
def lagr(x_interp: float, x_vec: np.ndarray, y_vec: np.ndarray) -> float:
    y_interp = 0.0
    for k in range(x_vec.size):
        # xc = xp[k]
        L_k = 1.0
        for j in range(x_vec.size):
            if j != k:
                L_k *= (x_interp - x_vec[j]) / (x_vec[k] - x_vec[j])

        y_interp += y_vec[k] * L_k

    return y_interp


if __name__ == "__main__":
    # Control points i.e. the given data points of (x,y) pairs to fit
    # n = 6
    # n = 16
    n = 32
    # n = 64

    # Linearly spaced
    # xp = np.linspace(-1, 1, n)

    # Chebyshev points
    xp = np.array([cos((2.0 * j + 1.0) * pi / (2.0 * n)) for j in range(n)])

    yp = np.array([f(x) for x in xp])

    # Sample points (to plot)
    x_plot = np.linspace(-1, 1, 500)
    y_inter = np.array([lagr(x_interp, xp, yp) for x_interp in x_plot])
    y_truth = np.array([f(x) for x in x_plot])

    # Plot figure using Matplotlib
    plt.figure()
    plt.plot(x_plot, y_inter, label="Interpolant")
    plt.plot(x_plot, y_truth, label="Function")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
