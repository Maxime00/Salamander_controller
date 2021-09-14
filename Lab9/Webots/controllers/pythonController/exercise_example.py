"""Exercise example"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters


def exercise_example(world, timestep, reset):
    """Exercise example"""
    # Parameters
    n_joints = 10
    phase_offset = 1.4 #put optimal offset
    amp = 0.3

    parameter_set = [
        SimulationParameters(
            simulation_duration=10,
            drive= 1.0,
            amplitudes=[amp]*n_joints,
            phase_lag=[0,0,0,0,0,0,0,0,0,0,phase_offset, phase_offset, phase_offset, phase_offset],
            turn=None,
            amplitude_gradient=None, #[4.0,2.0],
            backward = None,
        )
        #for drive in np.linspace(1, 2, 2)
        # for amplitudes in ...
        # for ...
    ]

    # Grid search
    for simulation_i, parameters in enumerate(parameter_set):
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000*parameters.simulation_duration/timestep),
            logs="./logs/example/simulation_{}.npz".format(simulation_i)
        )

