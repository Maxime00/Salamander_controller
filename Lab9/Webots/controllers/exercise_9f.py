"""Exercise 9f"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
from plot_results import plot_positions
from plot_results import plot_trajectory
from plot_results import plot_2d
import matplotlib.pyplot as plt



def exercise_9f(world, timestep, reset):
    """Exercise 9f"""
    n_joints = 10

    #phase_offset = 1.4 #put optimal offset
    amp = 0.3
    
    parameter_set = [
        SimulationParameters(
            simulation_duration=10,
            drive= 1.5,
            amplitudes=[amp]*n_joints,
            phase_lag=[0,0,0,0,0,0,0,0,0,0,phase_offset, phase_offset, phase_offset, phase_offset],
            turn=None,
            amplitude_gradient=None, #[4.0,2.0],
            backward = None,
            frequency = None,
            
        )
        #for drive in np.linspace(1,2,5)
        for phase_offset in np.linspace(-2*np.pi,2*np.pi,30)
        #for amp in np.linspace(0,0.55, 30)
        # for ...
    ]
    """
    # Grid search
    for simulation_i, parameters in enumerate(parameter_set):
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000*parameters.simulation_duration/timestep),
            logs="./logs/9ftest/simulation_{}.npz".format(simulation_i)
        )
    
    
    #grid search for phase offset
    for simulation_i, parameters in enumerate(parameter_set):
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000*parameters.simulation_duration/timestep),
            logs="./logs/9ftest/simulation_{}.npz".format(simulation_i)
        )
    """
    results_vel = np.zeros([len(parameter_set),3])
    
    
    path ='D:/EPFL/MA2/Computational Motor Control/Local/Lab9/Webots/controllers/pythonController/'

    #9f3 - phase offset
    #9f - amplitude

    for i in range(len(parameter_set)):
        with np.load(path+'logs/9ftest/simulation_'+str(i)+'.npz',allow_pickle=True) as data:
            
            timestep = float(data["timestep"])

            results_vel[i] = np.mean(np.mean(data["joints"][:,:,1]))
           
        
    
    phase_off = np.linspace(-2*np.pi,2*np.pi,30)
    #ampli = np.linspace(0,0.55, 30)
    
    plt.figure("Velocity")
    plt.plot(phase_off, -results_vel)
    #plt.plot(ampli, -results_vel)
    plt.xlabel("Phase offset [rad]")
    #plt.xlabel("Nominal radius [rad]")
    plt.ylabel("Velocity [m/s]")

    plt.show()

    

   
