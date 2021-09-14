"""Exercise 9f"""

import numpy as np
import os
import time
from math import sqrt
import plot_results 
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
import matplotlib.pyplot as plt



def exercise_9f(world, timestep, reset):
    """Exercise 9f"""
    n_joints = 10

    #phase_offset = 1.4 #put optimal offset
    amp = 0.3
    
    parameter_set = [
        SimulationParameters(
            simulation_duration=15,
            drive= 3.0,
            amplitudes=[amp]*n_joints,
            phase_lag=[0,0,0,0,0,0,0,0,0,0,phase_offset, phase_offset, phase_offset, phase_offset],
            turn=None,
            amplitude_gradient=None, #[4.0,2.0],
            backward = None,
            phase_offset = phase_offset,
        )
        #for drive in np.linspace(1,2,5)
        for phase_offset in np.linspace(0,2*np.pi,24)
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
    
    """
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
    
    plot_9f(parameter_set)



def main():
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
            phase_offset = phase_offset
        )
        for phase_offset in np.linspace(-2*np.pi,2*np.pi,30)

    ]
    
    plot_9f(parameter_set)

    

def plot_9f(parameter_set):
    results_vel = np.zeros([len(parameter_set),3])
    phase_offset = np.zeros(len(parameter_set))
    amplitudes = np.zeros(len(parameter_set))
   
    t = time.time()

    path = os.path.dirname(__file__)
    print(path)
    for i in range(len(parameter_set)):
        with np.load(path+'/logs/9ftest/simulation_'+str(i)+'.npz',allow_pickle=True) as data:
            
            #? initialisation for the computation
            position = data["links"][:, 0, :]
            n_steps = len(position)
            
            phase_offset[i] = data["phase_offset"]
            amplitudes[i] = data["amplitudes"][0]

            timestep = float(data["timestep"])
            begin_step = (int)(5/timestep)

            #! Velocity

            vel = (position[n_steps-1,:] - position[begin_step,:])**2
            results_vel[i]= sqrt(np.sum(vel))/((n_steps-begin_step)*timestep)
 
    print ('Time elapsed for the velocity plot' + str(time.time()-t))

    plt.figure("Velocity")
    plt.plot(phase_offset, results_vel)
    #plt.plot(ampli, -results_vel)
    plt.xlabel("Phase offset [rad]")
    #plt.xlabel("Nominal radius [rad]")
    plt.ylabel("Velocity [m/s]")

    plt.show()

