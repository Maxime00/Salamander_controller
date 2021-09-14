"""Exercise 9b"""

import time
import numpy as np
import os
from math import sqrt
import matplotlib.pyplot as plt
from plot_results import plot_2d
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters


def exercise_9b(world, timestep, reset):
    """Exercise 9b"""

    print ('a')
    n_joints = 10
    amp = 0.35
    lag = 0.0

    parameter_set = [
        SimulationParameters(
            simulation_duration = 15,
            drive=4.0,
            amplitude_gradient=None,#[4.0,2.0] 
            amplitudes = [amp]*n_joints,
            phase_lag=[lag]*n_joints,
            turn=None,
            backward = None,
            frequency = None,
            
        )
        #for amp in np.linspace(0.1,0.6,11)
        #for lag in np.linspace(0, np.pi/5, 11)
    ]

    
    # Grid search
    for simulation_i, parameters in enumerate(parameter_set):
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000*parameters.simulation_duration/timestep),
            logs="./logs/9b/simulation_{}.npz".format(simulation_i)
        )
    


    plot_9b(parameter_set)


def main():
    n_joints = 10
    parameter_set = [
        SimulationParameters(
            simulation_duration = 10,
            drive=4.0,
            amplitude_gradient=None,#[4.0,2.0] 
            amplitudes = [amp]*n_joints,
            phase_lag=[lag]*n_joints,
            turn=None,
            backward = None,
            
        )
        for amp in np.linspace(0.2,0.6,11)
        for lag in np.linspace(0, np.pi/5, 11)
    ]
    
    plot_9b(parameter_set)


def plot_9b(parameter_set):
    results_vel = np.zeros([len(parameter_set),3])
    results_en = np.zeros([len(parameter_set),3])
    cnt = 0
    amp = -1
   
    t = time.time()

    #path = os.path.dirname(__file__)
    path = 'D:/EPFL/MA2/Computational Motor Control/Local/Lab9/Webots/controllers/pythonController/'
    print(path)
    for i in range(len(parameter_set)):
        with np.load(path+'/logs/9b/simulation_'+str(i)+'.npz',allow_pickle=True) as data:
            
            #? initialisation for the computation
            position = data["links"][:, 0, :]
            n_steps = len(position)
            
            timestep = float(data["timestep"])

            results_vel[i][0] = data["phase_lag"][0]
            results_vel[i][1] = data["amplitudes"][0] 

            results_en[i][:2] = results_vel[i][:2] 

            #! Velocity

            begin_step = (int)(4/timestep)

            vel = (position[n_steps-1,:] - position[begin_step,:])**2
            results_vel[i][2] = sqrt(np.sum(vel))/((n_steps-begin_step)*timestep)

            #! Energy

            joint_vel = data["joints"][begin_step:,:,1]
            joint_tor = data["joints"][begin_step:,:,3]

            energy = joint_vel * joint_tor
            
            results_en[i][2] = np.log10(np.mean(np.sum(energy,1)))
            
    print ('Time elapsed for the velocity plot' + str(time.time()-t))



    plt.figure("Velocity")
    plot_2d(results_vel,['Phase lag [rad]', 'Amplitude [rad]', 'Velocity [m/s]'])
    plt.figure("Energy")
    plot_2d(results_en,['Phase lag [rad]', 'Amplitude [rad]', '$log_{10}(Energy)$[J]'])

    t = time.time()
    
    plt.show()  


if __name__ == '__main__':
    main()

