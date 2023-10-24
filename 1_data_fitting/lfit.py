#!/usr/bin/python


from math import *
import matplotlib.pyplot as plt
import numpy as np

# Vandermonde interpolation function
# def vand_f(x, b):
#     fx = b[0]
#     for i in range(n - 1):
#         fx *= x
#         fx += b[i + 1]
#     return fx


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

    # z = b_coeffs[0]
    # for i in range(1, n_points):
    #     z *= x_interp
    #     z += b_coeffs[i]
    # return z


# Synthesise data points to be fitted and a truncated Vandermonde matrix
n = 12  # a 11th degree polynomial to be fitted
x_data_min, x_data_max, n_datapoints = 0.0, 1.0, 50
x_data = np.linspace(x_data_min, x_data_max, n_datapoints)  # sample x-values to fit
A = np.vander(x_data, n)  # truncated columns of a full vandermonde matrix
y_data = np.cos(4 * x_data)  # y-values to fit

# Solve using the least-squares function
b1, res1, _, _ = np.linalg.lstsq(A, y_data, rcond=None)  # best-fit polynomial coeffs
print(f"The norm of the least-squares fit is: {np.linalg.norm(res1):3.2e}")

# Solve the normal equations directly
AT = np.transpose(A)
ATA = np.dot(AT, A)
print(f"Condition number of (A^T)A: {np.linalg.cond(ATA)}")
b2 = np.linalg.solve(ATA, np.dot(AT, y_data))  # obtain best-fit coeffs for polynomial

# Evaluate difference between the two parameter sets
print(f"Difference between the coefficient values of lsqfit & normal eqn: {b1 - b2}")

# Plot results
n_plotpoints = 201
# x_plot_min, x_plot_max = 0.0, 1.0
# x_plot_min, x_plot_max = -1.5, 2.5
x_plot_min, x_plot_max = -1.2, 1.5
x_plot = np.linspace(x_plot_min, x_plot_max, n_plotpoints)
y_lsq_fit_plot = [polyval(x, b1) for x in x_plot]
y_normal_eqn_plot = [polyval(x, b2) for x in x_plot]

plt.figure()
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x_data, y_data, "o", x_plot, y_lsq_fit_plot, "-", x_plot, y_normal_eqn_plot)
plt.legend(["data", "least sq.", "normal eqn. direct"], loc="best")
plt.show()


# plt.figure()
# plt.xlabel("x")
# plt.ylabel("y")
# plt.plot(x_data, y_data, "o", x_plot, y_lsq_fit_plot)
# plt.legend(["data", "least sq."], loc="best")
# plt.show()
