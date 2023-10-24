#!/usr/bin/python3


from math import cos, pi

import matplotlib.pyplot as plt
import numpy as np


# Function to calculate the sum of absolute values of Lagrange polynomials
def lsum(x, xp):
    ls = 0
    for k in range(xp.size):
        li = 1
        for j in range(xp.size):
            if j != k:
                li *= (x - xp[j]) / (xp[k] - xp[j])
        ls += abs(li)
    return ls


if __name__ == "__main__":
    # Control points (either linearly spaced, or Chebyshev)
    n = 31
    xp = np.linspace(-1.0, 1.0, n)
    # xp = np.array([cos((2.0 * j + 1.0) * pi / (2.0 * n)) for j in range(n)])

    # Sample points (to plot)
    x_plot = np.linspace(-1, 1, 500)
    lsum_plotpoints = np.array([lsum(x, xp) for x in x_plot])

    print(f"The Lebesgue constant is {np.max(lsum_plotpoints):.2f}")

    # Plot figure using Matplotlib
    plt.figure()
    plt.plot(x_plot, lsum_plotpoints)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
