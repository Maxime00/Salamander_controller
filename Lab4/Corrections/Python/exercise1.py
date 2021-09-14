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
    pylog.warning("Pendulum equation with spring and damper must be implemented")  #  _S
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



def pendulum_limit_cycle(x0, time=0.0, *args):
    # Initialize the parameters without damping
    pendulum = args[0]
    pendulum.parameters.b1 = 0.0

    pendulum.parameters.b2 = 0.0
    pylog.info(pendulum.parameters.showParameters())
    pylog.info(
        "1a. Running pendulum_system with springs to study limit cycle behavior")
    title = "{} Limit Cycle(x0 = {})"
    res = integrate(pendulum_perturbation, x0, time, args=args)
    res.plot_state(title.format("State", x0))
    res.plot_phase(title.format("Phase", x0))


def pendulum_spring_constant(x0, time=0.0, *args):
    # Initialize the parameters to bias spring 1
    pendulum = args[0]
    pendulum.parameters.b1 = 0.0
    pendulum.parameters.b2 = 0.0
    pendulum.parameters.k1 = 0.1
    pendulum.parameters.k2 = 0.1
    pylog.info(
        "1b. Running pendulum_system for analysing role of spring constant")
    pylog.info(pendulum.parameters.showParameters())
    title = "{} Spring Constant 1(x0 = {})"
    res = integrate(pendulum_integration, x0, time, args=args)
    res.plot_state(title.format("State", x0))
    res.plot_phase(title.format("Phase", x0))

    # Initialize the parameters to bias spring 2
    pendulum.parameters.b1 = 0.0
    pendulum.parameters.b2 = 0.0
    pendulum.parameters.k1 = 100.
    pendulum.parameters.k2 = 100.
    pylog.info(
        "1b. Running pendulum_system for analysing role of spring constant")
    pylog.info(pendulum.parameters.showParameters())
    title = "{} Spring Constant 2(x0 = {})"
    res = integrate(pendulum_integration, x0, time, args=args)
    res.plot_state(title.format("State", x0))
    res.plot_phase(title.format("Phase", x0))

    # Initialize the pendulum.parameters to bias spring 1
    pendulum.parameters.b1 = 0.0
    pendulum.parameters.b2 = 0.0
    pendulum.parameters.k1 = 1.0
    pendulum.parameters.k2 = 100.0
    pylog.info(
        "1b. Running pendulum_system for analysing role of variable spring constant")
    pylog.info(pendulum.parameters.showParameters())
    title = "{} Variable Spring Constant 1(x0 = {})"
    res = integrate(pendulum_integration, x0, time, args=args)
    res.plot_state(title.format("State", x0))
    res.plot_phase(title.format("Phase", x0))


def pendulum_spring_reference(x0, time=0.0, *args):
    # Initialize the parameters to bias spring 1 reference angle
    pendulum = args[0]
    pendulum.parameters.b1 = 0.0
    pendulum.parameters.b2 = 0.0
    pendulum.parameters.k1 = 10.0
    pendulum.parameters.k2 = 10.0
    pendulum.parameters.s_theta_ref1 = np.deg2rad(-10.0)
    pendulum.parameters.s_theta_ref2 = np.deg2rad(10.0)

    pylog.info(
        "1b. Running pendulum_system for analysing role of spring reference")
    pylog.info(pendulum.parameters.showParameters())
    title = "{} Spring Reference 1(x0 = {})"
    res = integrate(pendulum_integration, x0, time, args=args)
    res.plot_state(title.format("State", x0))
    res.plot_phase(title.format("Phase", x0))

    # Initialize the pendulum.parameters to bias spring 2 reference angle
    pendulum.parameters.b1 = 0.0
    pendulum.parameters.b2 = 0.0
    pendulum.parameters.k1 = 10.0
    pendulum.parameters.k2 = 10.0
    pendulum.parameters.s_theta_ref1 = np.deg2rad(-75.0)
    pendulum.parameters.s_theta_ref2 = np.deg2rad(75.)
    pylog.info(
        "1b. Running pendulum_system for analysing role of spring reference")
    pylog.info(pendulum.parameters.showParameters())
    title = "{} Spring Reference 2(x0 = {})"
    res = integrate(pendulum_integration, x0, time, args=args)
    res.plot_state(title.format("State", x0))
    res.plot_phase(title.format("Phase", x0))

    # Initialize the pendulum.parameters to bias spring 2 reference angle
    pendulum.parameters.b1 = 0.0
    pendulum.parameters.b2 = 0.0
    pendulum.parameters.k1 = 10.0
    pendulum.parameters.k2 = 10.0
    pendulum.parameters.s_theta_ref1 = np.deg2rad(0.0)
    pendulum.parameters.s_theta_ref2 = np.deg2rad(75.0)

    pylog.info(
        "1c. Running pendulum_system for analysing role of variable spring reference")
    pylog.info(pendulum.parameters.showParameters())
    title = "{} Variable Spring Reference 1(x0 = {})"
    res = integrate(pendulum_integration, x0, time, args=args)
    res.plot_state(title.format("State", x0))
    res.plot_phase(title.format("Phase", x0))


def pendulum_spring_damper(x0, time=0.0, *args):
    """ Function to analyse the pendulum spring damper system"""
    pendulum = args[0]
    pendulum.parameters.b1 = 0.5
    pendulum.parameters.b2 = 0.5
    pendulum.parameters.k1 = 50.0
    pendulum.parameters.k2 = 50.0
    pendulum.parameters.s_theta_ref1 = np.deg2rad(-45.0)
    pendulum.parameters.s_theta_ref2 = np.deg2rad(45.0)
    pylog.info(
        "20. Running pendulum_system for analysing role of spring and damper muscle")
    pylog.info(pendulum.parameters.showParameters())
    title = "{} Spring Damper (x0 = {})"
    res = integrate(pendulum_perturbation, x0, time, args=args)
    res.plot_state(title.format("State", x0))
    res.plot_phase(title.format("Phase", x0))


def pendulum_set_position(x0, time=0.0, *args):
    """ Function to analyse the pendulum spring damper system"""
    pendulum = args[0]
    pendulum.parameters.b1 = 1.
    pendulum.parameters.b2 = 1.
    pendulum.parameters.k1 = 50.0
    pendulum.parameters.k2 = 50.0
    pendulum.parameters.s_theta_ref1 = np.deg2rad(0.0)
    pendulum.parameters.s_theta_ref2 = np.deg2rad(65.6)

    pylog.info(
        "1b. Running pendulum_system to set fixed position")
    pylog.info(pendulum.parameters.showParameters())
    title = "{} Pendulum Fixed Position (x0 = {})"
    res = integrate(pendulum_integration, x0, time, args=args)
    pylog.debug('Position : {}'.format(np.rad2deg(res.state[-1])))
    res.plot_state(title.format("State", x0))
    res.plot_phase(title.format("Phase", x0))



def exercise1():
    """ Exercise 1  """
    pylog.info("Executing Lab 4 : Exercise 1")
    pendulum = PendulumSystem()
    pendulum.parameters = PendulumParameters()

    pylog.info(
        "Find more information about Pendulum Parameters in SystemParameters.py")
    pylog.info("Executing Lab 4 : Exercise 1")

    # Initialize a new pendulum system with default parameters
    pendulum = PendulumSystem()
    pendulum.parameters = PendulumParameters()

    pylog.info(
        "Find more information about Pendulum Parameters in SystemParameters.py")

    # To change a parameter you could so by,
    # >>> pendulum.parameters.L = 1.5
    # The above line changes the length of the pendulum

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

    pendulum.parameters.k1 = 50
    pendulum.parameters.k2 = 50
    pendulum.parameters.s_theta_ref1 = 1.0
    pendulum.parameters.s_theta_ref2 = 1.0
    pendulum.parameters.b1 = 0.5
    pendulum.parameters.b2 = 0.5

    pylog.warning("Loading default pendulum pendulum.parameters")
    pylog.info(pendulum.parameters.showParameters())

    # Simulation Parameters
    t_start = 0.0
    t_stop = 10.0
    dt = 0.01
    pylog.warning("Using large time step dt={}".format(dt))
    time = np.arange(t_start, t_stop, dt)
    x0 = [0.5, 0.0]

    res = integrate(pendulum_integration, x0, time, args=(pendulum,))
    pylog.info("Instructions for applying pertubations")
    # Use pendulum_perturbation method to apply pertubations
    # Define the pertubations inside the function pendulum_perturbation
    # res = integrate(pendulum_perturbation, x0, time, args=(pendulum,))
    res.plot_state("State")
    res.plot_phase("Phase")

    x0 = [0.5, 0.1]
    pendulum_limit_cycle(x0, time, pendulum)
    pendulum_spring_constant(x0, time, pendulum)
    pendulum_spring_reference(x0, time, pendulum)
    pendulum_spring_damper(x0, time, pendulum)
    pendulum_set_position(x0, time, pendulum)


    if DEFAULT["save_figures"] is False:
        plt.show()
    return


if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    exercise1()