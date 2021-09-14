""" Lab 4 """

import matplotlib.pyplot as plt
import numpy as np

import cmc_pylog as pylog
from cmcpack import DEFAULT, integrate, integrate_multiple, parse_args
from pendulum_system import PendulumSystem
from system_animation import SystemAnimation
from system_parameters import PendulumParameters

DEFAULT["label"] = [r"$\theta$ [rad]", r"$d\theta/dt$ [rad/s]"]


def pendulum_integration(state, time, *args):
    """ Function for system integration """
    pylog.warning(
        "Pendulum equation with spring and damper must be implemented")  # _S
    pendulum = args[0]
    return pendulum.pendulum_system(
        state[0], state[1], time, torque=0.0
    )[:, 0]


def pendulum_perturbation(state, time, *args):
    """ Function for system integration with perturbations.
    Setup your time based system pertubations within this function.
    The pertubation can either be applied to the states or as an
    external torque.
    """
    pendulum = args[0]
    if time > 5 and time < 5.1:
        pylog.info("Applying state perturbation to pendulum_system")
        state[1] = 2.
    return pendulum.pendulum_system(
        state[0], state[1], time, torque=0.0)[:, 0]


def exercise1():
    """ Exercise 1  """
    pylog.info("Executing Lab 4 : Exercise 1")
    
    #Default parameters
    pendulum = PendulumSystem()
    pendulum.parameters = PendulumParameters()

    pylog.info(
        "Find more information about Pendulum Parameters in SystemParameters.py")
    pylog.info("Executing Lab 4 : Exercise 1")

    # Initialize a new pendulum system with default parameters
    #pendulum_1 = PendulumSystem()
    #pendulum_1.parameters = PendulumParameters()

    pylog.info(
        "Find more information about Pendulum Parameters in SystemParameters.py")

    # To change a parameter you could so by,
    # >>> pendulum.parameters.L = 1.5
    # The above line changes the length of the pendulum

    pendulum_1 = PendulumSystem()
    pendulum_1.parameters = PendulumParameters()
    pendulum_1.parameters.k1 = 40
    pendulum_1.parameters.k2 = 40
    pendulum_1.parameters.s_theta_ref1 = 0
    pendulum_1.parameters.s_theta_ref2 = 0



    # You can instantiate multiple pendulum models with different parameters
    # For example,
    """
    >>> pendulum_1 = PendulumSystem()
    >>> pendulum_1.parameters = PendulumParameters()
    >>> pendulum_1.parameters.L = 0.5
    >>> parameters_2 = PendulumParameters()
    >>> parameters_2.L = 0.5
    >>> pendulum_2 = PendulumSystem(paramters=parameters_2)
    """
    # Above examples shows how to create two istances of the pendulum
    # and changing the different parameters using two different approaches

    pylog.warning("Loading default pendulum pendulum.parameters")
    pylog.info(pendulum.parameters.showParameters())

    # Simulation Parameters
    t_start = 0.0
    t_stop = 10.0
    dt = 0.01
    pylog.warning("Using large time step dt={}".format(dt))
    time = np.arange(t_start, t_stop, dt)
    x0 = [0.5, 0.0]

    #res = integrate(pendulum_integration, x0, time, args=(pendulum,))
    pylog.info("Instructions for applying pertubations")
    # Use pendulum_perturbation method to apply pertubations
    # Define the pertubations inside the function pendulum_perturbation
    res = integrate(pendulum_integration, x0, time, args=(pendulum,))
    res_1 = integrate(pendulum_integration, x0, time, args=(pendulum_1,))
    res.plot_state("State")
    res.plot_phase("Phase")

    res_1.plot_state("State _ 1")
    res_1.plot_phase("Phase_1")

    # To animate the model, use the SystemAnimation class
    # Pass the res(states) and systems you wish to animate
    simulation1 = SystemAnimation(res, pendulum, handler="Pendulum animation")
    simulation2 = SystemAnimation(res_1, pendulum_1, handler="Pendulum animation 2")
    # If you want to disable the parameters in the animation plot
    # simulation2 = SystemAnimation(res, pendulum, handler="Pendulum animation 2", show_parameters=False)
    # To start the animation, use either simulation.animate() or plt.show()
    # simulation.animate()

    if DEFAULT["save_figures"] is False:
        plt.show()
    return


if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    exercise1()

