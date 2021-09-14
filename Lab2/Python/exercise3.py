""" Lab 2 """

import numpy as np
import matplotlib.pyplot as plt

import cmc_pylog as pylog
from cmcpack import integrate, DEFAULT, parse_args

from ex3_pendulum import PendulumParameters, pendulum_system


DEFAULT["label"] = [r"$\theta$ [rad]", r"$d\theta/dt$ [rad/s]"]


def exercise3(clargs):
    """ Exercise 3 """
    parameters = PendulumParameters()  # Checkout pendulum.py for more info
    pylog.info(parameters)
    # Simulation parameters
    time = np.arange(0, 30, 0.01)  # Simulation time
    x0 = [0.1, 0.0]  # Initial state

    # To use/modify pendulum parameters (See PendulumParameters documentation):
    # parameters.g = 9.81  # Gravity constant
    # parameters.m = 1.  # Mass
    # parameters.L = 1.  # Length
    # parameters.I = 1. # Inertia (Automatically computed!)
    # parameters.d = 0.3  # damping
    # parameters.sin = np.sin  # Sine function
    # parameters.dry = False  # Use dry friction (True or False)

    # Example of system integration (Similar to lab1)
    # (NOTE: pendulum_equation must be imlpemented first)
    pylog.debug("Running integration example")
    res = integrate(pendulum_system, x0, time, args=(parameters,))
    res.plot_state("State")
    res.plot_phase("Phase")

    # Evolutions
    # Write code here (You can add functions for the different cases)
    pylog.warning(
        "Evolution of pendulum in normal conditions must be implemented"
    )
    pylog.warning(
        "Evolution of pendulum without damping must be implemented"
    )
    pylog.warning(
        "Evolution of pendulum with perturbations must be implemented"
    )
    pylog.warning(
        "Evolution of pendulum with dry friction must be implemented"
    )

    # Show plots of all results
    if not clargs.save_figures:
        plt.show()


if __name__ == "__main__":
    CLARGS = parse_args()
    exercise3(CLARGS)

