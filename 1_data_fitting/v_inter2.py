#!/usr/bin/python3

from math import exp

import matplotlib.pyplot as plt
import numpy as np

from numpy.polynomial.polynomial import polyfit, polyval


def f(x: float) -> float:
    return exp(-x)


# Evaluate the interpolated polynomial at a particular scalar point x_interp
def eval_interp(x_interp: float, b_coeffs) -> float:
    # b_coeffs cannot have less than 2 entries since the lowest order polynomial (line) has 2 coeffs
    if len(b_coeffs) < 2:
        raise ValueError("b_vec_coeffs cannot have less than 2 entries")

    # Horner's method
    # initialise calculated interp value to b_0 (corresponding to x^n)
    y_interp = b_coeffs[0]
    idx = 0
    while idx < len(b_coeffs) - 1:
        y_interp = (y_interp * x_interp) + b_coeffs[idx + 1]
        idx += 1

    return y_interp


if __name__ == "__main__":
    n_points = 12  # no of data-pairs (x_i, y_i) in the given dataset that is to be fitted with an interpolant
    x_min, x_max = 0.0, 3.0
    x_vec_given = np.linspace(x_min, x_max, n_points)  # array of x-values of data
    y_vec_given = np.array([f(x) for x in x_vec_given])  # array of y-values of data

    # solve Vandermonde problem (V b = y)
    V = np.vander(x_vec_given)
    b_coeffs = np.linalg.solve(V, y_vec_given)  # b_0 x^n + b_1 x^(n-1) + ... + b_n
    b_coeffs_numpy = polyfit(x_vec_given, y_vec_given, n_points - 1)
    # print(b_coeffs)
    # print(np.flip(b_coeffs_numpy))

    # Add optional random perturbation to coeffs
    # b_coeffs += 1e-6 * np.random.rand(n_points)

    # Plot the interpolant evaluated at a fine grid of x-values
    n_plotpoints = 301
    x_vec_plot = np.linspace(x_min, x_max, n_plotpoints)
    y_truth_plot = np.array([f(x) for x in x_vec_plot])
    # y_vec_interp = np.array([eval_interp(x, b_coeffs) for x in x_vec_plot])
    # y_vec_interp = polyval(x_vec_plot, np.flip(b_coeffs))
    y_vec_interp = polyval(x_vec_plot, b_coeffs_numpy)
    y_err = y_vec_interp - y_truth_plot

    # Plot the fitted/interpolated curve and the original data using Matplotlib
    plt.figure()
    plt.plot(x_vec_plot, y_vec_interp, label="Interpolant")
    plt.plot(x_vec_plot, y_truth_plot, label="exp(-x)")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # Plot the error between interpolated curve and original data
    plt.figure()
    plt.plot(x_vec_plot, y_err, label="Error")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("error")
    plt.show()
