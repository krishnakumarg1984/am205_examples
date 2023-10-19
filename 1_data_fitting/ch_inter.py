#!/usr/bin/python3

# from math import *
import matplotlib.pyplot as plt
import numpy as np


# Function to interpolate
def f(x):
    return abs(x)


# Function to evaluate the Lagrange interpolation
def lagr(x_interp, xp, yp):
    y_interp = 0
    for k in range(xp.size):
        # xc = xp[k]
        L_k = 1
        for j in range(xp.size):
            if j != k:
                L_k *= (x_interp - xp[j]) / (xp[k] - xp[j])
        y_interp += yp[k] * L_k
    return y_interp


if __name__ == "__main__":
    # Control points (given data) to fit
    # n = 16
    n = 6
    xp = np.linspace(-1, 1, n)  # (Linearly spaced)
    # xp=np.array([cos((2*j+1)*pi/(2*n)) for j in range(n)]) # (Chebyshev points)
    yp = np.array([f(q) for q in xp])

    # Sample points (for computing the interpolant coefficients)
    x_plot = np.linspace(-1, 1, 500)
    y_inter = np.array([lagr(x_interp, xp, yp) for x_interp in x_plot])
    y_truth = np.array([f(q) for q in x_plot])

    # Plot figure using Matplotlib
    plt.figure()
    plt.plot(x_plot, y_inter, label="Interpolant")
    plt.plot(x_plot, y_truth, label="Function")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
