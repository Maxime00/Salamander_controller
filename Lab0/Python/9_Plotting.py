#!/usr/bin/env python3

""" This script shows how to plot in Python with Matplotlib """

import cmc_pylog as pylog  # import pylog for log messages
import numpy as np  # Import numpy as np
from matplotlib import pyplot as plt  # Import pyplot from matplotlib

### MATPLOTLIB ###

pylog.info(3*'\t' + 20*'#' + ' MATPLOTLIB ' + 20*'#' + 3*'\n')

# Main plotting library for Python
# Works well with Numpy and Scipy
# Uses methods similar to MATLAB plotting tools

# Create an exponential function y = exp(x)
# Where x is between [-10, 10]

# Create a numpy array of 100 points between -10 and 10
x = np.linspace(-10, 10, 100)

y = np.exp(x)  # Compute the exponential of x

# Plotting the function y
# Create a new figure window
plt.figure()
# Plot x and y
plt.plot(x, y, label='$y = exp(x)$')
# Turn on grid
plt.grid(True)
# X axis label
plt.xlabel('X-axis')
# Y axis label
plt.ylabel('Y-axis')
# Title of the plot
plt.title('Plotting an exponential function')
# Legend of the plot
plt.legend()
# Show the figure
plt.show()

