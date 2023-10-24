#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Construct splines for two sets of similar data points
x = np.linspace(0, 5, 6)
y = np.array([0, 2, 0, -4, 0, 6])
y2 = np.array([0, 2.1, 0, -4, 0, 6])
s = interp1d(x, y, kind="cubic")
s2 = interp1d(x, y2, kind="cubic")

# Plot the splines and data
x_plot = np.linspace(0, 5, 200)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, y, "o", x_plot, s(x_plot), "-", x, y2, "o", x_plot, s2(x_plot), "--")
plt.legend(["data", "cubic", "data2", "cubic2"], loc="best")
plt.show()

# Optional plot of the difference
plt.plot(x_plot, s(x_plot) - s2(x_plot), "-")

plt.show()
