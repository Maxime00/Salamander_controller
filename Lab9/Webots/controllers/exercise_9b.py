"""Exercise 9b"""

import time
import numpy as np
import matplotlib.pyplot as plt
from plot_results import plot_2d
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters


def exercise_9b(world, timestep, reset):
    """Exercise 9b"""

    
    n_joints = 10
    #amp = 0.14
    #lag = 0.11

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
        for amp in np.linspace(0.1,0.5,10)
        for lag in np.linspace(0, 0.5, 10)
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
    

    results_vel = np.zeros([len(parameter_set),3])
    results_en = np.zeros([len(parameter_set),3])
   
    #phase_lags = []
    t = time.time()

    for i in range(len(parameter_set)):
        with np.load('logs/9b/simulation_'+str(i)+'.npz',allow_pickle=True) as data:
            
            timestep = float(data["timestep"])

            #position = data["links"][:, 0, :]
            #n_steps = len(position)
            
                     
            results_vel[i,0] = data["phase_lag"][0]
            results_vel[i,1] = data["amplitudes"][0]       

            #print(np.mean(np.mean(data["joints"][:,:,1])))

            results_vel[i,2] = -(np.mean(np.mean(data["joints"][:,:,1])))
            
            #joints = data["joints"]
            #~ print(joints.shape)
   
   

    plt.figure("Velocity")
    plot_2d(results_vel,['Phase lag [rad]', 'Amplitude [rad]', 'Velocity [m/s]'])
    
    
    
    for i in range(len(parameter_set)):
        with np.load('logs/9btest/simulation_'+str(i)+'.npz',allow_pickle=True) as data:
            
            timestep = float(data["timestep"])
            results_vel[i,0] = data["phase_lag"][0]
            results_vel[i,1] = data["amplitudes"][0]  
            results_en[i,2]=np.sum(np.sum((data["joints"][:,:,1]*data["joints"][:,:,3]*timestep)))
            

            
    print(results_en)
    
    
    plt.figure("Energy")
    plot_2d(results_en,['Phase lag [rad]', 'Amplitude [rad]', 'Energy'])
    
   
    plt.show()     


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
        for amp in np.linspace(0.1,0.5,10)
        for lag in np.linspace(0, 0.5, 10)
    ]


    
    results_vel = np.zeros([len(parameter_set),3])
    results_en = np.zeros([len(parameter_set),3])
   
    t = time.time()

    path = 'D:/EPFL/MA2/Computational Motor Control/Local/Lab9/Webots/controllers/pythonController/'
    for i in range(len(parameter_set)):
        with np.load(path+'logs/9btest/simulation_'+str(i)+'.npz',allow_pickle=True) as data:
            
            timestep = float(data["timestep"])    
            results_vel[i,0] = data["phase_lag"][0]
            results_vel[i,1] = data["amplitudes"][0]       
            results_vel[i,2] = -1*(np.mean(np.mean(data["joints"][:,:,1])))

            
    print ('Time elapsed for the velocity plot' + str(time.time()-t))



    plt.figure("Velocity")
    plot_2d(results_vel,['Phase lag [rad]', 'Amplitude [rad]', 'Velocity [m/s]'])
    
    t = time.time()
    
    for i in range(len(parameter_set)):
        with np.load(path + 'logs/9btest/simulation_'+str(i)+'.npz',allow_pickle=True) as data:
            
            timestep = float(data["timestep"])
           

            n_iterations = len(data["links"][:, 0, 0])
               
            results_en[i,0] = data["phase_lag"][0]
            results_en[i,1] = data["amplitudes"][0]       

            #energy_in_step = np.zeros([n_iterations])
            #energy_joints = np.zeros([n_joints])
            
            results_en[i,2]=np.sum(np.sum((data["joints"][:,:,1]*data["joints"][:,:,3]*timestep)))


    print ('Time elapsed for the energy plot' + str(time.time()-t))
    
    
    plt.figure("Energy")
    plot_2d(results_en,['Phase lag [rad]', 'Amplitude [rad]', 'Energy'])

    plt.show()     

if __name__ == '__main__':
    main()