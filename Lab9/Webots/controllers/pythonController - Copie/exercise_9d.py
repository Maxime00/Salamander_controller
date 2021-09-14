"""Exercise 9d"""

import numpy as np 
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
from plot_results import plot_positions
from plot_results import plot_trajectory
import matplotlib.pyplot as plt


def exercise_9d1(world, timestep, reset): 
    """Exercise 9d1""" 
    #Turning
    n_joints = 10
   
    parameters=SimulationParameters(
            simulation_duration=10,
            drive=4.0,
            amplitudes=None,
            phase_lag=None,
            turn=0.15,
            amplitude_gradient=None, #[4.0,2.0],
            backward = None,
            frequency = None,
            # ...
    )
      

    reset.reset()
    run_simulation(
        world,
        parameters,
        timestep,
        int(1000*parameters.simulation_duration/timestep),
        logs="./logs/9d/simulation_d1.npz"
    )


    # Load data
    with np.load('logs/9d/simulation_d1.npz') as data:
        timestep = float(data["timestep"])
        pos_data = np.mean(data["links"][:, :, :],1)

        joints_data = data["joints"][:,:10,0]

    times = np.arange(0, timestep*np.shape(joints_data)[0], timestep)
    

    # Plot data
    plt.figure("Positions")
    plot_trajectory(pos_data)
    
    plt.figure("Trajectories")
    plt.plot(times, joints_data)
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Joints Angle [rad]")
    plt.grid


    plt.show()     


def exercise_9d2(world, timestep, reset):
    """Exercise 9d2"""
    #Backward
    n_joints = 10
   
    parameters=SimulationParameters(
            simulation_duration=10,
            drive=3.1, #4.0 toujorus un peu violent
            amplitudes=None,
            phase_lag=None,
            turn=None,
            amplitude_gradient=None, #[4.0,2.0],
            backward = 1.0,
            # ...
        )
      

    reset.reset()
    run_simulation(
        world,
        parameters,
        timestep,
        int(1000*parameters.simulation_duration/timestep),
        logs="./logs/9d/simulation_d2.npz"
    )


    # Load data
    with np.load('logs/9d/simulation_d2.npz') as data:
        timestep = float(data["timestep"])
        pos_data = np.mean(data["links"][:, :, :],1)

        joints_data = data["joints"][:,:10,0]
        

    times = np.arange(0, timestep*np.shape(joints_data)[0], timestep)
    

    # Plot data
    plt.figure("Positions")
    plot_trajectory(pos_data)
    
    plt.figure("Trajectories")
    plt.plot(times, joints_data)
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Joints Angle [rad]")
    plt.grid


    plt.show()     
 



    pass


def main ():

    """Exercise 9d1""" 
    #Turning
    n_joints = 10
    
    parameters=SimulationParameters(
            simulation_duration=10,
            drive=4.0,
            amplitudes=None,
            phase_lag=None,
            turn=0.15,
            amplitude_gradient=None, #[4.0,2.0],
            backward = None,
            # ...
    )
    
    """
    reset.reset()
    run_simulation(
        world,
        parameters,
        timestep,
        int(1000*parameters.simulation_duration/timestep),
        logs="./logs/9d/simulation_d1.npz"
    )"""

    path = 'D:/EPFL/MA2/Computational Motor Control/Local/Lab9/Webots/controllers/pythonController/'
    # Load data
    with np.load(path+'logs/9d/simulation_d1.npz') as data:
        timestep = float(data["timestep"])
        pos_data = np.mean(data["links"][:, :, :],1)

        print(pos_data.shape)
        joints_data = data["joints"][:,:10,0]
        print(joints_data.shape)

    times = np.arange(0, timestep*np.shape(joints_data)[0], timestep)
    print(times.shape)

    # Plot data
    plt.figure("Positions")
    plot_trajectory(pos_data)
    
    plt.figure("Trajectories")
    plt.plot(times, joints_data)
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Joints Angle [rad]")
    plt.grid


    plt.show()     

if __name__ == '__main__':
    main()