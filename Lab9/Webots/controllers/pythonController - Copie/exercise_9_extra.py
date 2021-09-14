"""Exercise 9 extra"""

# from run_simulation import run_simulation

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
from plot_results import plot_positions
from plot_results import plot_trajectory
import matplotlib.pyplot as plt


def exercise_9_extra(world, timestep, reset):
    """Exercise 9 extra"""

    n_joints = 10

    parameter_set = [
        SimulationParameters(
            simulation_duration=20,
            drive=1.5,
            amplitudes=None,
            phase_lag=None,
            turn=None,
            amplitude_gradient=None,
            backward=None,
            frequency = None,
            # ...
        )
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
            logs="./logs/9extra/simulation_{}.npz".format(simulation_i)
        )

    #path = '/Users/peterbonnesoeur/Documents/epfl cours/Computational Motor Control/2019/Lab9/Webots/controllers/pythonController/'
    path = 'D:/EPFL/MA2/Computational Motor Control/Local/Lab9/Webots/controllers/pythonController/'

    with np.load(path+'logs/9extra/simulation_0.npz', allow_pickle=True) as data:
        timestep = float(data["timestep"])
        pos_data = np.mean(data["links"][:, :, :], 1)
        print(pos_data.shape)
        joints_data = data["joints"][:, :10, 0]
        print(joints_data.shape)

    times = np.arange(0, timestep*np.shape(joints_data)[0], timestep)
    print(times.shape)

    # Plot data
    plt.figure("Positions")
    plot_trajectory(pos_data)

    plt.figure("Trajectories")
    plt.plot(times, joints_data)

    plt.show()
