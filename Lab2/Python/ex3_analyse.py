""" Analyse system symbolically (corrections) """

import numpy as np
import cmc_pylog as pylog
from ex3_pendulum import PendulumParameters, pendulum_system, pendulum_equation

USE_SYMPY = False
try:
    import sympy as sp
    USE_SYMPY = True
except ImportError as err:
    pylog.error(err)
    USE_SYMPY = False

