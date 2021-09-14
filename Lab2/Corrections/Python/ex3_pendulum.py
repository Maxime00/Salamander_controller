""" Pendulum """

import numpy as np
import cmc_pylog as pylog


class PendulumParameters(object):
    """ Pendulum system parameters

    with:
        - g: Gravity constant [m/s**2]
        - m: Mass [kg]
        - L: Length [m]
        - I: Inerita [kg-m**2]
        - d: Damping coefficient
        - sin: Sine function
        - dry: Use dry friction (bool: True or False)

    Examples:

        >>> pendulum_parameters = PendulumParameters(g=9.81, L=0.1)
        >>> pendulum_parameters = PendulumParameters(d=0.3, dry=True)

    Note that not giving arguments to instanciate the object will result in the
    following default values:

        - g = 9.81
        - m = 1.
        - L = 1.
        - I = 1.
        - d = 0.3
        - sin = np.sin
        - dry = False

    These parameter variables can then be called from within the class using
    for example:

        To assign a new value to the object variable from within the class:

        >>> self.g = 9.81 # Reassign gravity constant

        To assign to another variable from within the class:

        >>> example_g = self.g

    To call the parameters from outside the class, such as after instatiation
    similarly to the example above:

        To assign a new value to the object variable from outside the class:

        >>> pendulum_parameters = PendulumParameters(L=1.0)
        >>> pendulum_parameters.L = 0.3 # Reassign length

        To assign to another variable from outside the class:

        >>> pendulum_parameters = PendulumParameters()
        >>> example_g = pendulum_parameters.g # example_g = 9.81

    You can display the parameters using:

    >>> pendulum_parameters = PendulumParameters()
    >>> print(pendulum_parameters)
    Pendulum system parameters:
        g: 9.81 [m/s**2]
        m: 1. [kg]
        L: 1.0 [m]
        I: 1.0 [kg-m**2]
        d: 0.3
        sin: <ufunc 'sin'>
        dry: False

    Or using pylog:

    >>> pendulum_parameters = PendulumParameters()
    >>> pylog.info(pendulum_parameters)
    """

    def __init__(self, **kwargs):
        super(PendulumParameters, self).__init__()
        self._mass = 1.0  # Internal parameter : Do NOT use it!
        self._length = 1.0  # Internal parameter : Do NOT use it!
        self.g = kwargs.pop("g", 9.81)  # Gravity constant
        self.I = None  # Inertia be set automatically when mass and length are set
        self.m = kwargs.pop("m", 1.)  # Mass
        self.L = kwargs.pop("L", 1.)  # Length
        self.d = kwargs.pop("d", 0.3)  # damping
        self.sin = kwargs.pop("sin", np.sin)  # Sine function
        self.dry = kwargs.pop("dry", False)  # Use dry friction (True or False)
        pylog.info(self)

    @property
    def L(self):
        """ Get the Length of the pendulum."""
        return self._length

    @L.setter
    def L(self, value):
        """
        Set the Length of the pendulum.
        Setting/Changing Length will automatically recompute the inertia.
        """
        self._length = value
        # ReCompute inertia
        # Inertia = m*l**2
        self.I = self.m*self._length**2

    @property
    def m(self):
        """ Get the mass of the pendulum."""
        return self._mass

    @m.setter
    def m(self, value):
        """
        Set the mass of the pendulum.
        Setting/Changing mass will automatically recompute the inertia.
        """
        self._mass = value
        # ReCompute inertia
        # Inertia = m*l**2
        self.I = self._mass*self.L**2

    def __str__(self):
        return self.msg()

    def msg(self, endl="\n"+4*" "):
        """ Message """
        return (
            "Pendulum system parameters:"
            + 5*(endl + "{}: {} {}")
        ).format(
            "g", self.g, "[m/s**2]",
            "m", self.m, "[kg]"
            "L", self.L, "[m]",
            "I", self.I, "[kg-m**2]"
            "d", self.d, "",
            "sin", self.sin, "",
            "dry", self.dry, ""
        )

    @classmethod
    def from_list(cls, parameter_list):
        """ Generate object from list of arguments

        Order is:
        0) gravity (float)
        1) Length (float)
        2) Damping (float)
        3) Sine function (function)
        4) Dry friction (bool)
        """
        size = len(parameter_list)
        keys = ["g", "L", "d", "sin", "dry"]
        return cls(**{
            key: parameter_list[i]
            for i, key in enumerate(keys)
            if size > i
        })


def pendulum_equation(theta, dtheta, time=0, parameters=None, torque=0.):
    """ Pendulum equation

    with:
        - theta: Angle [rad]
        - dtheta: Angular velocity [rad/s]
        - g: Gravity constant [m/s**2]
        - m: Mass [kg]
        - L: Length [m]
        - d: Damping coefficient []
        - torque: External torque [N-m]
    """
    if parameters is None:
        parameters = PendulumParameters()
        pylog.warning(
            "Parameters not given, using defaults\n{}".format(parameters)
        )
    g, m, L, I, d, sin, dry = (
        parameters.g,
        parameters.m,
        parameters.L,
        parameters.I,
        parameters.d,
        parameters.sin,
        parameters.dry
    )
    if not dry:
        ddtheta = -g*(1./L)*sin(theta) - d*dtheta - torque/I
    else:
        ddtheta = -g*(1./L)*sin(theta) - d*sign(dtheta) - torque/I
    return ddtheta


def sign(dtheta):
    """ return sign """
    # return np.sign(dtheta)  # May run into integration instability
    tol = 1e8
    return np.sign(dtheta) if tol*np.abs(dtheta) > 1 else tol*dtheta


def pendulum_system(state, time=None, parameters=None, torque=0.0):
    """ Function for system integration

    State corresponds of: state = [theta, dtheta]

    """
    theta, dtheta = state
    return np.array([
        dtheta,
        pendulum_equation(theta, dtheta, time, parameters, torque)  # d2theta
    ])