#!/usr/bin/python3

from math import exp

import matplotlib.pyplot as plt
import numpy as np

n = 5  # Maximum exponential term (can tune)


# Evaluates sum of coeff-scaled exponentials at a particular x
# Model (which is a linear function in b)
def sum_bk_exp_kx(x: float, b: np.ndarray) -> float:
    y_eval = 0.0
    for k in range(-n, n + 1):
        y_eval += b[k + n] * exp(k * x)
    return y_eval


if __name__ == "__main__":
    # Create matrix where each column is a suitably scaled exponential, as opposed to a monomial in the usual Vandermonde construction
    x_data_min, x_data_max, n_fitpoints = -1.0, 1.0, 20
    x_data = np.linspace(x_data_min, x_data_max, n_fitpoints)

    # A = np.array([[exp((i - n) * xx) for i in range(s)] for xx in x_data])
    A = np.array([[exp(k * x) for k in range(-n, n + 1)] for x in x_data])
    y_data = np.cos(4 * x_data) * np.exp(-x_data)

    # Solve using the least-squares function
    b = np.linalg.lstsq(A, y_data, rcond=None)[0]  # obtain the coeffs
    print(
        "Norm(r)/Norm(b) :", np.linalg.norm(y_data - np.dot(A, b)) / np.linalg.norm(b)
    )

    # Plot results
    x_plot_min, x_plot_max, n_plotpoints = -1.0, 1.0, 201
    x_plot = np.linspace(x_plot_min, x_plot_max, n_plotpoints)
    y_plot = [sum_bk_exp_kx(x, b) for x in x_plot]  # evaluate the fitted function

    plt.figure()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x_data, y_data, "o", x_plot, y_plot, "-")
    plt.legend(["data", "least sq."], loc="best")
    plt.show()
