#!/usr/bin/python3

from math import *
import matplotlib.pyplot as plt
import numpy as np


def polyval(x_interp: float, b_coeffs: np.ndarray) -> float:
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


# Vandermonde interpolation function
def vand_f(x, b):
    fx = b[0]
    for i in range(n - 1):
        fx *= x
        fx += b[i + 1]
    return fx


# Synthesise data points to be fitted and a truncated Vandermonde matrix
n = 12  # a 11th degree polynomial to be fitted
x_data_min, x_data_max, n_datapoints = (0.2, 1.0, 5)  # no of points << degree
x_data = np.linspace(x_data_min, x_data_max, n_datapoints)  # sample x-values to fit
A = np.vander(x_data, n)  # rectangular vandermonde matrix
y_data = np.cos(4 * x_data)  # y-values to fit

# Solve using least squares routine. For an underdetermined system this finds
# the interpolant that minimizes the norm of the parameter vector b1.
b1 = np.linalg.lstsq(A, y_data, rcond=None)[0]
norm_r_lsq = np.linalg.norm(y_data - np.dot(A, b1))
norm_b_lsq = np.linalg.norm(b1)
print(f"lstsq solve : Norm(res): {norm_r_lsq:3.2e}")
print(f"lstsq solve : Norm(sol): {norm_b_lsq:3.2e}")
print(f"lstsq solve : Norm(res)/Norm(sol): {norm_r_lsq / norm_b_lsq:3.2e}")

print("\n------------------------------------")

# Solve using normal equations + regularizer
mu = 0.05
# mu = 0.001
# mu = 0.5  # penalise the size of the solution at the cost of missing giving data points

# mu = 0.1 * np.ones(n)
# mu[3:] = 0.1  # heavily penalise the cubic and higher order coefficients
# mu = np.diag(mu)
# print(mu)

AT = np.transpose(A)
ATA = np.dot(AT, A)
print(f"Condition number without regularizer (A^T A): {np.linalg.cond(ATA):3.2e}")
print(
    f"Condition number of regularised system (A^T A + S^T S): {np.linalg.cond(ATA + (mu * mu * np.identity(n))):3.2e}"
)
b2 = np.linalg.solve(ATA + mu * mu * np.identity(n), np.dot(AT, y_data))
norm_r_normaleq = np.linalg.norm(y_data - np.dot(A, b2))
norm_b_normaleq = np.linalg.norm(b2)
print(f"Normal eqs. : Norm(res): {norm_r_normaleq:3.2e}")
print(f"Normal eqs. : Norm(sol): {norm_b_normaleq:3.2e}")
print(f"Normal eqs. : Norm(res)/Norm(sol): {norm_r_normaleq / norm_b_normaleq:3.2e}")

# Plot the two solutions
n_plotpoints = 201
# x_plot_min, x_plot_max = 0.0, 1.0
# x_plot_min, x_plot_max = -1.5, 2.5
x_plot_min, x_plot_max = 0.0, 1.0
x_plot = np.linspace(x_plot_min, x_plot_max, n_plotpoints)
y_lsq_fit_plot = [polyval(x, b1) for x in x_plot]
y_normal_eqn_plot = [polyval(x, b2) for x in x_plot]

plt.figure()
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x_data, y_data, "o", x_plot, y_lsq_fit_plot, "-", x_plot, y_normal_eqn_plot)
plt.legend(["data", "least sq.", "normal eqn."], loc="best")
plt.show()
