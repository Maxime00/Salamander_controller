""" Lab 1 - Exercise 2 """

import numpy as np
import matplotlib.pyplot as plt

from systems import system_analysis
from cmcpack import integrate
import cmc_pylog as pylog


def ode(x, _=None, A=np.eye(2)):
    """ System x_dot = A*x """
    return np.dot(A, x)


def integration(x0, time, A, name, **kwargs):
    """ System integration """
    labels = kwargs.pop("label", ["State {}".format(i) for i in range(2)])
    sys_int = integrate(ode, x0, time, args=(A,))
    sys_int.plot_state("{}_state".format(name), labels)
    sys_int.plot_phase("{}_phase".format(name))


def exercise2(clargs):
    """ Exercise 2 """
    pylog.info("Running exercise 2")

    # System definition
    A = np.array([[1, 4], [-4, -2]])
    system_analysis(A)  # Optional
    time_total = 10
    time_step = 0.01
    x0, time = [0, 1], np.arange(0, time_total, time_step)

    # Normal run
    pylog.info("Running system integration")
    integration(x0, time, A, "system_integration")

    # Stable point (Optional)
    pylog.info("Running stable point integration")
    x0 = [0, 0]
    integration(x0, time, A, "stable")

    # Periodic
    pylog.info("Running periodic system integration")
    A = np.array([[2, 4], [-4, -2]])
    x0 = [1, 0]
    integration(x0, time, A, "periodic")

    # Plot
    if not clargs.save_figures:
        plt.show()


if __name__ == "__main__":
    from cmcpack import parse_args
    CLARGS = parse_args()
    exercise2(CLARGS)