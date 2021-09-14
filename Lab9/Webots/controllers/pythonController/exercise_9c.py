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
    
    Rhead = 0.44
    Rtail = 0.23


    parameter_set = [
        SimulationParameters(
            simulation_duration=15,
            drive=4.0,
            amplitudes=None,
            phase_lag=None,
            turn=None,
            amplitude_gradient=[Rhead, Rtail], #[4.0,2.0],
            backward = None,
            frequency = 1,
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
            logs="./logs/9c/simulation_{}.npz".format(simulation_i)
        )

    

    plot_9c(parameter_set)
 

   

def main():


    n_joints = 10

    #Rhead = 0.44
    #Rtail = 0.27  
    
    parameter_set = [
        SimulationParameters(
            simulation_duration=15,
            drive=4.0,
            amplitudes=None,
            phase_lag=None,
            turn=None,
            amplitude_gradient=[Rhead, Rtail], #[4.0,2.0],
            backward = None,
            frequency = 1,
            # ...
        )
        
        for Rhead in np.linspace(0.2,0.5,10)
        for Rtail in np.linspace(0.5,0.2,10)
        # for amplitudes in ...
        # for ...
    ]

    plot_9c(parameter_set)

    
def plot_9c(parameter_set):
    results_vel = np.zeros([len(parameter_set),3])
    results_en = np.zeros([len(parameter_set),3])
    ratio_vel_en = np.zeros([len(parameter_set),3])
    
    
    sal_pos_t = []
    sal_pos_t_bad = []

   
    t = time.time()

    #path = os.path.dirname(__file__)
    path = 'D:/EPFL/MA2/Computational Motor Control/Local/Lab9/Webots/controllers/pythonController/'
    print(path)
    for i in range(len(parameter_set)):
        with np.load(path+'/logs/9c/simulation_'+str(i)+'.npz',allow_pickle=True) as data:
            
            #? initialisation for the computation
            position = data["links"][:, 0, :]
            n_steps = len(position)
            
            timestep = float(data["timestep"])

            results_vel[i][0] = data["amplitude_gradient"][0]
            results_vel[i][1] = data["amplitude_gradient"][1] 

            results_en[i][:2]   = results_vel[i][:2] 
            ratio_vel_en[i][:2] = results_vel[i][:2]

            

            #! Velocity

            begin_step = (int)(4/timestep)

            vel = (position[n_steps-1,:] - position[begin_step,:])**2
            results_vel[i][2] = np.sqrt(np.sum(vel))/((n_steps-begin_step)*timestep)

            #! Energy

            joint_vel = data["joints"][begin_step:,:,1]
            joint_tor = data["joints"][begin_step:,:,3]

            energy = joint_vel * joint_tor
            
            results_en[i][2] = np.log10(np.mean(np.sum(energy,1)))
            
            #! Ratio 

            ratio_vel_en[i][2]  = results_vel[i][2]/results_en[i][2]
         
           
    print ('Time elapsed for the velocity plot' + str(time.time()-t))



    plt.figure("Velocity")
    plot_2d(results_vel,['Amplitude Head [rad]', 'Amplitude Tail [rad]', 'Velocity [m/s]'])
    plt.figure("Energy")
    plot_2d(results_en,['Amplitude Head [rad]', 'Amplitude Tail [rad]', '$log_{10}(Energy)$[J]'])
    plt.figure("Ratio")
    plot_2d(ratio_vel_en,['Amplitude Head [rad]', 'Amplitude Tail [rad]', 'Ratio V/E $[s\cdot kg^{-1}\cdot m^{-1}]$'])
    
    t = time.time()
    
    plt.show()  

if __name__ == '__main__':
    main()
