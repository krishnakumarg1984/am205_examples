#!/usr/bin/python3


import matplotlib.pyplot as plt
import numpy as np


# n: polynomial degree
def kth_lagrange_polynomial(
    k: int, x: np.ndarray, x_knotpoints: np.ndarray, n: int
) -> np.ndarray:
    L_k = np.ones(x.shape)
    for j in range(n + 1):  # python range function is exclusive
        if j != k:
            L_k *= (x - x_knotpoints[j]) / (x_knotpoints[k] - x_knotpoints[j])
    return L_k


if __name__ == "__main__":
    n_knotpoints = 5
    x_min, x_max = 0.0, n_knotpoints - 1.0
    x_knotpoints = np.linspace(x_min, x_max, n_knotpoints)  # array of x-values

    n_plotpoints = 100 * n_knotpoints + 1
    x_vec_plot = np.linspace(x_min, x_max, n_plotpoints)
    lk_vec = np.array(np.zeros((n_knotpoints, n_plotpoints)))
    for k in range(n_knotpoints):
        # k = 3
        lk_vec[k, :] = kth_lagrange_polynomial(
            k, x_vec_plot, x_knotpoints, n_knotpoints - 1
        )

    # Plot the fitted/interpolated curve and the original data using Matplotlib
    plt.figure()
    for k in range(n_knotpoints):
        # k = 3
        plt.plot(x_vec_plot, lk_vec[k, :])
        plt.plot(
            x_knotpoints,
            kth_lagrange_polynomial(k, x_knotpoints, x_knotpoints, n_knotpoints - 1),
            "x",
        )
    plt.xlabel("x")
    plt.ylabel("$L_\mathrm{k}(x)$")
    plt.grid()
    plt.show()
