#!/usr/bin/python3


import matplotlib.pyplot as plt
import numpy as np


# Evaluatet the kth lagrange polynomial of degree n
def kth_lagrange_polynomial(
    k: int, x: np.ndarray, x_knotpoints: np.ndarray, n: int
) -> np.ndarray:
    L_k = np.ones(x.shape)
    for j in range(n + 1):  # python range function is exclusive
        if j != k:
            L_k *= (x - x_knotpoints[j]) / (x_knotpoints[k] - x_knotpoints[j])

    return L_k


if __name__ == "__main__":
    n_knotpoints = 6
    # x_min, x_max = 0.0, n_knotpoints - 1.0
    x_min_plot, x_max_plot = -1.0, 1.0
    x_knotpoints = np.linspace(x_min_plot, x_max_plot, n_knotpoints)

    n_plotpoints = 100 * n_knotpoints + 1
    x_vec_plot = np.linspace(x_min_plot, x_max_plot, n_plotpoints)
    lk_vec = np.array(np.zeros((n_knotpoints, n_plotpoints)))
    for k in range(n_knotpoints):
        lk_vec[k, :] = kth_lagrange_polynomial(
            k, x_vec_plot, x_knotpoints, n_knotpoints - 1
        )

    plt.figure()
    for k in range(n_knotpoints):
        k = 2
        plt.plot(x_vec_plot, lk_vec[k, :])
        plt.plot(x_knotpoints, np.zeros(x_knotpoints.shape), "x")

        # plt.plot(
        #     x_knotpoints,
        #     kth_lagrange_polynomial(k, x_knotpoints, x_knotpoints, n_knotpoints - 1),
        #     "x",
        # )
    plt.xlabel("x")
    plt.ylabel("$L_\mathrm{k}(x)$")
    # plt.grid()
    plt.xlim(x_min_plot, x_max_plot)
    plt.show()
