""" Lab 2 - Exercise 4 """

import numpy as np
import matplotlib.pyplot as plt

import cmc_pylog as pylog
from cmcpack import integrate, integrate_multiple, parse_args

from ex4_hopf import (
    HopfParameters,
    CoupledHopfParameters,
    hopf_equation,
    coupled_hopf_equation
)


def hopf_ocillator():
    """ 4a - Hopf oscillator simulation """
    pylog.warning("Hopf oscillator must be implemented")
    # params = HopfParameters()


def coupled_hopf_ocillator():
    """ 4b - Coupled Hopf oscillator simulation """
    pylog.warning("Coupled Hopf oscillator must be implemented")
    # param = CoupledHopfParameters()


def exercise4(clargs):
    """ Exercise 4 """
    hopf_ocillator()
    coupled_hopf_ocillator()
    # Show plots of all results
    if not clargs.save_figures:
        plt.show()


if __name__ == "__main__":
    CLARGS = parse_args()
    exercise4(CLARGS)

