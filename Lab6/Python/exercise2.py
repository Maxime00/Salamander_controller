""" Lab 6 Exercise 2

This file implements the pendulum system with two muscles attached

"""

from math import sqrt, cos, sin

import cmc_pylog as pylog
import numpy as np
from matplotlib import pyplot as plt

from cmcpack import DEFAULT
from cmcpack.plot import save_figure
from muscle import Muscle
from muscle_system import MuscleSytem
from neural_system import NeuralSystem
from pendulum_system import PendulumSystem
from system import System
from system_animation import SystemAnimation
from system_parameters import (MuscleParameters, NetworkParameters,
                               PendulumParameters)
from system_simulation import SystemSimulation


# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels

DEFAULT["save_figures"] = True


def exercise2():
    """ Main function to run for Exercise 2.

    Parameters
    ----------
        None

    Returns
    -------
        None
    """

    # Define and Setup your pendulum model here
    # Check PendulumSystem.py for more details on Pendulum class
    pendulum_params = PendulumParameters()  # Instantiate pendulum parameters
    pendulum_params.L = 0.5  # To change the default length of the pendulum
    pendulum_params.m = 1.  # To change the default mass of the pendulum
    pendulum = PendulumSystem(pendulum_params)  # Instantiate Pendulum object

    #### CHECK OUT PendulumSystem.py to ADD PERTURBATIONS TO THE MODEL #####

    pylog.info('Pendulum model initialized \n {}'.format(
        pendulum.parameters.showParameters()))

    # Define and Setup your pendulum model here
    # Check MuscleSytem.py for more details on MuscleSytem class
    M1_param = MuscleParameters()  # Instantiate Muscle 1 parameters
    M1_param.f_max = 1500  # To change Muscle 1 max force
    M2_param = MuscleParameters()  # Instantiate Muscle 2 parameters
    M2_param.f_max = 1500  # To change Muscle 2 max force
    M1 = Muscle(M1_param)  # Instantiate Muscle 1 object
    M2 = Muscle(M2_param)  # Instantiate Muscle 2 object
    # Use the MuscleSystem Class to define your muscles in the system
    muscles = MuscleSytem(M1, M2)  # Instantiate Muscle System with two muscles
    pylog.info('Muscle system initialized \n {} \n {}'.format(
        M1.parameters.showParameters(),
        M2.parameters.showParameters()))



    # Define Muscle Attachment points
    m1_origin = np.array([-0.17, 0.0])  # Origin of Muscle 1
    m1_insertion = np.array([0.0, -0.17])  # Insertion of Muscle 1

    m2_origin = np.array([0.17, 0.0])  # Origin of Muscle 2
    m2_insertion = np.array([0.0, -0.17])  # Insertion of Muscle 2

    # Attach the muscles
    muscles.attach(np.array([m1_origin, m1_insertion]),
                   np.array([m2_origin, m2_insertion]))

    # Create a system with Pendulum and Muscles using the System Class
    # Check System.py for more details on System class
    sys = System()  # Instantiate a new system
    sys.add_pendulum_system(pendulum)  # Add the pendulum model to the system
    sys.add_muscle_system(muscles)  # Add the muscle model to the system

    ##### Time #####
    t_max = 2.5# Maximum simulation time
    time = np.arange(0., t_max, 0.001)  # Time vector

    
    frequencies = np.arange(2,50,48)

    def sinus (freq, time):
        return 1/2*(1+  np.sin(freq*time))

    def cosinus (freq, time):
        return 1/2*(1+  np.cos(freq*time))

    def squares1 (freq, time):
        result = []
        for t in time:
            if (sin(freq*t)>=0):
                result.append(1.0)
            else :
                result.append(0.0)
        return result

    def squares2 (freq, time):
        result = []
        for t in time:
            if (cos(freq*t)>=0):
                result.append(1.0)
            else :
                result.append(0.0)
        return result




    ##### Model Initial Conditions #####
    x0_P = np.array([0., 0.])  

    # REMINDER X( theta theta_dot, Act1, length1, Act2, length2)

    # Muscle Model initial condition
    x0_M = np.array([0., M1.L_OPT, 0., M2.L_OPT])

    x0 = np.concatenate((x0_P, x0_M))  # System initial conditions

    ##### System Simulation #####
    # For more details on System Simulation check SystemSimulation.py
    # SystemSimulation is used to initialize the system and integrate
    # over time

    sim = SystemSimulation(sys)  # Instantiate Simulation object

    # Add muscle activations to the simulation
    # Here you can define your muscle activation vectors
    # that are time dependent

    act1 = np.array(sinus(10,time)).reshape(len(time), 1)  #np.ones((len(time), 1)) * 1
    
    act2 = np.array(sinus(-10,time)).reshape(len(time), 1)#np.ones((len(time), 1)) * 0.05#np.array(sinus(5,time)).reshape(len(time), 1) *0.05#sinus(5,time)#np.ones((len(time), 1)) * 0.05

    activations = np.hstack((act1, act2))

    # Method to add the muscle activations to the simulation

    sim.add_muscle_activations(activations)

    # Simulate the system for given time

    sim.initalize_system(x0, time)  # Initialize the system state

    #: If you would like to perturb the pedulum model then you could do
    # so by
    sim.sys.pendulum_sys.parameters.PERTURBATION = True
    # The above line sets the state of the pendulum model to zeros between
    # time interval 1.2 < t < 1.25. You can change this and the type of
    # perturbation in
    # pendulum_system.py::pendulum_system function

    # Integrate the system for the above initialized state and time
    sim.simulate()

    # Obtain the states of the system after integration
    # res is np.array [time, states]
    # states vector is in the same order as x0
    res = sim.results()


    # In order to obtain internal states of the muscle
    # you can access the results attribute in the muscle class
    muscle1_results = sim.sys.muscle_sys.Muscle1.results
    muscle2_results = sim.sys.muscle_sys.Muscle2.results


    thetas=res[:,1]





    ## COMPUTATION OF THE MOMENTS FOR THE MUSCLES 

    torques1 = []
    torques2 = []

    L1 = []
    L2 = []

    H1 = []
    H2 = []

    for theta in thetas:
        L1.append(sqrt(np.sum(pow(m1_insertion,2)) +np.sum(pow(m1_origin,2))+ 2*sin(theta)*sqrt(np.sum(pow(m1_insertion,2)))*sqrt(np.sum(pow(m1_origin,2)))))
        L2.append(sqrt(np.sum(pow(m2_insertion,2)) + np.sum(pow(m2_origin,2))+ 2*sin(-theta)*sqrt(np.sum(pow(m2_insertion,2)))*sqrt(np.sum(pow(m2_origin,2)))))


    for l1,l2,theta in zip(L1, L2, thetas):
        H1.append( sqrt(np.sum(pow(m1_insertion,2)))*sqrt(np.sum(pow(m1_origin,2)))*cos(theta)/l1)
        H2.append( sqrt(np.sum(pow(m2_insertion,2)))*sqrt(np.sum(pow(m2_origin,2)))*cos(-theta)/l2)


    # for h1,h2, muscle1, muscle2 in zip(H1, H2, muscle1_results.tendon_force, muscle2_results.tendon_force):
    #     torques1.append( muscle1*h1)
    #     torques2.append( muscle2*h2)



    fig, ax = plt.subplots()
    ax.plot(thetas, L1, label='Length of muscle 1 [m]' )
    ax.plot(thetas, L2, label='Length of muscle 2 [m]' )

    ax.plot(thetas, H1, label='Moment of muscle 1[N.m]' )
    ax.plot(thetas, H2, label='Moment of muscle 2[N.m]' )
    plt.title('Length and moments of the muscles as a function of the angle theta')
    plt.ylabel('Moment[N.m] & Length[m]')
    plt.xlabel('Theta [rad]')
    plt.grid()
    leg = ax.legend()

    fig, ax = plt.subplots()
    ax.plot(time, res[:,3], label='Activation of muscle 1' )
    ax.plot(time, act1, label='Excitation of muscle 1' )

    ax.plot(time, res[:,5], label='Activation of muscle 2' )
    ax.plot(time, act2, label='Excitation of muscle 2' )
    plt.title('Activation of muscle 2 compared to their excitation  function ')
    plt.ylabel('Activation')
    plt.xlabel('Time [s]')
    plt.grid()
    leg = ax.legend(loc ='upper right')

    fig, ax = plt.subplots()
    ax.plot(time, res[:,1], label='Angle' )
    ax.plot(time, res[:,2], label='Angular speed' )
    plt.title('angle and angular speed during the simulation')
    plt.ylabel('angle [rad] & angular speed [rad.s^-1]')
    plt.xlabel('Time [s]')
    plt.grid()
    leg = ax.legend(loc ='upper right')

    plt.figure('Pendulum')
    plt.title('Pendulum Phase')
    plt.plot(res[:, 1], res[:, 2])
    plt.xlabel('Position [rad]')
    plt.ylabel('Velocity [rad.s]')
    plt.grid()



 ## ## ## ## ## ## ## ##  2 C  ## ## ## ## ## ## ## ##  



    # frequencies = np.arange(1,100,20)
    # print(frequencies)
    # thetas =[]

    # x0_P = np.array([0., 0.])  # Pendulum initial condition np.pi/4
    # x0_M = np.array([0., M1.L_OPT, 0., M2.L_OPT])
    # x0 = np.concatenate((x0_P, x0_M))  # System initial conditions

    # fig, ax = plt.subplots()

    # for freq in frequencies:

    #     print (freq)

    #     act1 = np.array(sinus(freq,time)).reshape(len(time),1) 
        
    #     act2 = np.array(sinus(-freq,time)).reshape(len(time), 1) 

    #     activations = np.hstack((act1, act2))

    #     sim.add_muscle_activations(activations)

    #     sim.initalize_system(x0, time)  # Initialize the system state

    #     sim.sys.pendulum_sys.parameters.PERTURBATION = True
       
    #     sim.simulate()

    #     res = sim.results()
       
    #     ax.plot(res[10:, 1], res[10:, 2], label='fréquence = '+ freq.astype('str'))

        
    # plt.title('Pendulum phase as a function of the activation frequency')
    # plt.xlabel('Position [rad]')
    # plt.ylabel('Velocity [rad.s]')
    # plt.grid()
    # leg = ax.legend(loc ='upper right')
        

    # ax.plot(frequencies ,thetas, label='Amplitude' )
    # plt.title('Amplitude as a function of the frequency ')
    # plt.xlabel('frequency(Hz)')
    # plt.ylabel('Amplitude (rad)')
    # plt.grid()
    # leg = ax.legend()

    



    # To animate the model, use the SystemAnimation class
    # Pass the res(states) and systems you wish to animate
    simulation = SystemAnimation(res, pendulum, muscles)
    # To start the animation
    if DEFAULT["save_figures"] is False:
        simulation.animate()

    if not DEFAULT["save_figures"]:
        plt.show()
    else:
        figures = plt.get_figlabels()
        pylog.debug("Saving figures:\n{}".format(figures))
        for fig in figures:
            plt.figure(fig)
            save_figure(fig)
            plt.close(fig)


if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    exercise2()

