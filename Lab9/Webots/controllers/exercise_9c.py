"""Exercise 9c"""

import time
import numpy as np
import matplotlib.pyplot as plt
from plot_results import plot_2d
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters

def exercise_9c(world, timestep, reset):
    """Exercise 9c"""

    n_joints = 10
    Rhead = 0.37
    Rtail = 0.37
        
    parameter_set = [
        SimulationParameters(
            simulation_duration=10,
            drive=4.0,
            amplitudes=None,
            phase_lag=None,
            turn=None,
            amplitude_gradient=[Rhead, Rtail], #[4.0,2.0],
            backward = None,
            # ...
        )
        
        #for Rhead in np.linspace(0.2,0.5,10)
        #for Rtail in np.linspace(0.5,0.2,10)
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
            logs="./logs/9c1/simulation_{}.npz".format(simulation_i)
        )

    
    results_vel = np.zeros([len(parameter_set),3])
    results_en = np.zeros([len(parameter_set),3])
   
    t = time.time()

    #path = '/Users/peterbonnesoeur/Documents/epfl cours/Computational Motor Control/2019/Lab9/Webots/controllers/pythonController/'
    for i in range(len(parameter_set)):
        with np.load('logs/9c1/simulation_'+str(i)+'.npz',allow_pickle=True) as data:
            timestep = float(data["timestep"])    
            results_vel[i,0] =  data['amplitude_gradient'][0]#data["phase_lag"][0]
            results_vel[i,1] =  data['amplitude_gradient'][1]#data["amplitudes"][0]       
            results_vel[i,2] = -(np.mean(np.mean(data["joints"][:,:,1])))

            
    print ('Time elapsed for the velocity plot' + str(time.time()-t))



    plt.figure("Velocity")
    plot_2d(results_vel,['Amplitude Head [rad]', 'Amplitude Tail [rad]', 'Velocity [m/s]'])
    
    t = time.time()
    
    for i in range(len(parameter_set)):
        with np.load('logs/9c1/simulation_'+str(i)+'.npz',allow_pickle=True) as data:
            
            timestep = float(data["timestep"])
            results_en[i,0] =  data['amplitude_gradient'][0]#data["phase_lag"][0]
            results_en[i,1] =  data['amplitude_gradient'][1]
            results_en[i,2]=np.sum(np.sum((data["joints"][:,:,1]*data["joints"][:,:,3]*timestep)))
            

    print ('Time elapsed for the energy plot' + str(time.time()-t))
    
    
    plt.figure("Energy")
    plot_2d(results_en,['Amplitude Head [rad]', 'Amplitude Tail [rad]', 'Energy'])

    plt.show()      
    
   

def main():


    n_joints = 10
    
    parameter_set = [
        SimulationParameters(
            simulation_duration=10,
            drive=4.0,
            amplitudes=None,
            phase_lag=None,
            turn=None,
            amplitude_gradient=[Rhead, Rtail], #[4.0,2.0],
            backward = None,
            # ...
        )
        
        for Rhead in np.linspace(0.2,0.5,10)
        for Rtail in np.linspace(0.5,0.2,10)
        # for amplitudes in ...
        # for ...
    ]


    results_vel = np.zeros([len(parameter_set),3])
    results_en = np.zeros([len(parameter_set),3])
   
    t = time.time()

    path = 'D:/EPFL/MA2/Computational Motor Control/Local/Lab9/Webots/controllers/pythonController/'
    for i in range(len(parameter_set)):
        with np.load(path+'logs/9c/simulation_'+str(i)+'.npz',allow_pickle=True) as data:
            timestep = float(data["timestep"])    
            results_vel[i,0] =  data['amplitude_gradient'][0]#data["phase_lag"][0]
            results_vel[i,1] =  data['amplitude_gradient'][1]#data["amplitudes"][0]       
            results_vel[i,2] = -1*(np.mean(np.mean(data["joints"][:,:,1])))

            
    print ('Time elapsed for the velocity plot' + str(time.time()-t))



    plt.figure("Velocity")
    plot_2d(results_vel,['Amplitude Head [rad]', 'Amplitude Tail [rad]', 'Velocity [rad/s]'])
    
    t = time.time()
    
    for i in range(len(parameter_set)):
        with np.load(path + 'logs/9c/simulation_'+str(i)+'.npz',allow_pickle=True) as data:
            
            timestep = float(data["timestep"])
            results_en[i,0] =  data['amplitude_gradient'][0]#data["phase_lag"][0]
            results_en[i,1] =  data['amplitude_gradient'][1]
            results_en[i,2] =  np.sum(np.sum((data["joints"][:,:,1]*data["joints"][:,:,3]*timestep)))
            

    print ('Time elapsed for the energy plot' + str(time.time()-t))
    
    
    plt.figure("Energy")
    plot_2d(results_en,['Amplitude Head [rad]', 'Amplitude Tail [rad]', 'Energy'])

    plt.show()     

if __name__ == '__main__':
    main()