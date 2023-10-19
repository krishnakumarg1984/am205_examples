#!/usr/bin/python3

from math import exp

import matplotlib.pyplot as plt
import numpy as np


# Function to calculate a polynomial at the scalar point x_interp
def calc_interpolant(x_interp, b_coeffs):
    # b_coeffs cannot have less than 2 entries since the least polynomial is a straight line
    if len(b_coeffs) < 2:
        return 0

    y = b_coeffs[0]
    idx = 0
    while idx < len(b_coeffs) - 1:
        y = (y * x_interp) + b_coeffs[idx + 1]
        idx += 1

    return y

    # z = b_coeffs[0]
    # for i in range(1, n_points):
    #     z *= x_interp
    #     z += b_coeffs[i]
    # return z


# Initialize x points and function values
n_points = 12  # no of data-pairs (x_i, y_i) in the given dataset that is to be fitted with an interpolant
x_min, x_max = 0.0, 3.0
x_given = np.linspace(x_min, x_max, n_points)
y_given = np.array([exp(-q) for q in x_given])

# Solve Vandermonde problem
V = np.vander(x_given)
b = np.linalg.solve(V, y_given)

# Add optional random perturbation
b += 1e-6 * np.random.rand(n_points)

# Plot interpolant
n_plotpoints = 301
x_plot = np.linspace(x_min, x_max, n_plotpoints)
y_interp = np.array([calc_interpolant(x, b) for x in x_plot])
y_truth = np.array([exp(-q) for q in x_plot])
# y_err = np.array([calc_interpolant(q, b) - exp(-q) for q in x_plot])
y_err = y_interp - y_truth

# Plot the fitted/interpolated curve and the original data using Matplotlib
plt.figure()
plt.plot(x_plot, y_interp, label="Interpolant")
plt.plot(x_plot, y_truth, label="exp(-x)")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Plot the error between interpolated curve and original data
plt.figure()
plt.plot(x_plot, y_err, label="Error")
plt.legend()
plt.xlabel("x")
plt.ylabel("error")
plt.show()
