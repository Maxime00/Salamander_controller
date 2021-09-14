""" Lab 1 - Exercise 2 """

import numpy as np
import matplotlib.pyplot as plt

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
    pylog.warning("Proper matrix A must be implemented")
    A = np.array([[1, 0], [0, 1]])
    time_total = 10
    time_step = 0.01
    x0, time = [0, 1], np.arange(0, time_total, time_step)

    # Normal run
    pylog.warning("System integration must be implemented")
    # integration(x0, time, A, "example")

    # Stable point (Optional)
    pylog.warning("Stable point integration must be implemented")

    # Periodic
    pylog.warning("Periodic system must be implemented")

    # Plot
    if not clargs.save_figures:
        plt.show()


if __name__ == "__main__":
    from cmcpack import parse_args
    CLARGS = parse_args()
    exercise2(CLARGS)

