""" Lab 5 - Exercise 1 """

import matplotlib.pyplot as plt
import numpy as np

import cmc_pylog as pylog
from muscle import Muscle
from mass import Mass
from cmcpack import DEFAULT, parse_args
from cmcpack.plot import save_figure
from system_parameters import MuscleParameters, MassParameters
from isometric_muscle_system import IsometricMuscleSystem
from isotonic_muscle_system import IsotonicMuscleSystem

DEFAULT["label"] = [r"$\theta$ [rad]", r"$d\theta/dt$ [rad/s]"]

# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels

DEFAULT["save_figures"] = True


def exercise1a():
    """ Exercise 1a
    The goal of this exercise is to understand the relationship
    between muscle length and tension.
    Here you will re-create the isometric muscle contraction experiment.
    To do so, you will have to keep the muscle at a constant length and
    observe the force while stimulating the muscle at a constant activation."""

    # Defination of muscles

    parameters = MuscleParameters()
    pylog.warning("Loading default muscle parameters")
    pylog.info(parameters.showParameters())
    pylog.info("Use the parameters object to change the muscle parameters")

    # Create muscle object
    muscle = Muscle(parameters)

    pylog.warning("Isometric muscle contraction to be completed")

    # Instatiate isometric muscle system
    sys = IsometricMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)

    # You can still access the muscle inside the system by doing
    # >>> sys.muscle.L_OPT # To get the muscle optimal length

    # Evalute for a single muscle stretch
    muscle_stretch = 0.2

    # Evalute for a single muscle stimulation
    muscle_stimulation = 1.0

    # Set the initial condition
    x0 = [0.0, sys.muscle.L_OPT]
    # x0[0] --> muscle stimulation intial value
    # x0[1] --> muscle contracticle length initial value

    # Set the time for integration
    t_start = 0.0
    t_stop = 0.2
    time_step = 0.001

    time = np.arange(t_start, t_stop, time_step)

    # Run the integration
    result = sys.integrate(x0=x0,
                           time=time,
                           time_step=time_step,
                           stimulation=muscle_stimulation,
                           muscle_length=muscle_stretch)

    
    # Plotting
    plt.figure('Isometric muscle experiment')
    plt.plot(result.time, result.v_ce)
    plt.title('Isometric muscle experiment')
    plt.xlabel('Time [s]')
    plt.ylabel('Muscle contractilve velocity')
    plt.grid()

    #1.b - doubl boucle for for stim + length
    """
    muscle_stim = np.linspace(0, 1, 5)


    stretches = np.linspace(1.55*sys.muscle.L_OPT,2.8*sys.muscle.L_OPT,22)
    Forces = []
    Force_actives = []
    Force_passives = []



    for stim in muscle_stim :

        #ßstretch = np.linspace(1.55*sys.muscle.L_OPT,2.8*sys.muscle.L_OPT,20)
        Force = []
        Force_active = []
        Force_passive = []

        for stretch in stretches:
        
          result = sys.integrate(x0=x0,
                            time=time,
                            time_step=time_step,
                            stimulation=stim,
                            muscle_length=stretch)
        
          Force.append((result.active_force[-1] + result.passive_force[-1] ))
          Force_active.append((result.active_force[-1]))
          Force_passive.append((result.passive_force[-1] ))
        
        Forces.append(Force)
        Force_actives.append(Force_active)
        Force_passives.append(Force_passive)

    
    plt.figure('Isometric muscle experiment')
    fig, ax = plt.subplots()
    for Force ,Force_active, Force_passive , stim in zip(Forces ,Force_actives, Force_passives, muscle_stim) :
        ax.plot(stretches, Force, label='Activation = '+stim.astype('str') )
        #ax.plot(stretches, Force_active, label='Active force')
        #ax.plot(stretches,Force_passive, label='Passive force')
    plt.title('Isometric muscle experiment - Force(stretch)')
    plt.xlabel('Length [m]')
    plt.ylabel('Muscle Force [N]')
    plt.grid()
    leg = ax.legend()

    #1.a and 1.c
   
    stretches = np.linspace(0.15, 0.22, 50)
    #stretches = np.linspace(0.5*sys.muscle.L_OPT,3*sys.muscle.L_OPT,50)
    Force = []
    Force_active = [] 
    Force_passive = []
    for stretch in stretches:
        
        result = sys.integrate(x0=x0,
                            time=time,
                            time_step=time_step,
                            stimulation=muscle_stimulation,
                            muscle_length=stretch)
        
        Force.append((result.active_force[-1] + result.passive_force[-1] ))
        Force_active.append((result.active_force[-1]))
        Force_passive.append((result.passive_force[-1] ))

    plt.figure('Isometric muscle experiment')
    fig, ax = plt.subplots()
    ax.plot(stretches, Force, label='Total force')
    ax.plot(stretches, Force_active, label='Active force')
    ax.plot(stretches,Force_passive, label='Passive force')
    plt.title('Isometric muscle experiment')
    plt.xlabel('Length [m]')
    plt.ylabel('Muscle Force [N]')
    plt.grid()
    leg = ax.legend()"""















def exercise1d():
    """ Exercise 1d

    Under isotonic conditions external load is kept constant.
    A constant stimulation is applied and then suddenly the muscle
    is allowed contract. The instantaneous velocity at which the muscle
    contracts is of our interest."""

    # Defination of muscles
    muscle_parameters = MuscleParameters()
    print(muscle_parameters.showParameters())

    mass_parameters = MassParameters()
    print(mass_parameters.showParameters())

    # Create muscle object
    muscle = Muscle(muscle_parameters)

    # Create mass object
    mass = Mass(mass_parameters)

    pylog.warning("Isotonic muscle contraction to be implemented")

    # Instatiate isotonic muscle system
    sys = IsotonicMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)

    # Add the mass to the system
    sys.add_mass(mass)

    # You can still access the muscle inside the system by doing
    # >>> sys.muscle.L_OPT # To get the muscle optimal length

    # Evalute for a single load
    load = 100.

    # Evalute for a single muscle stimulation
    muscle_stimulation = 1.

    # Set the initial condition
    x0 = [0.0, sys.muscle.L_OPT,
          sys.muscle.L_OPT + sys.muscle.L_SLACK, 0.0]
    # x0[0] - -> activation
    # x0[1] - -> contractile length(l_ce)
    # x0[2] - -> position of the mass/load
    # x0[3] - -> velocity of the mass/load

    # Set the time for integration
    t_start = 0.0
    t_stop = 0.3
    time_step = 0.001
    time_stabilize = 0.2

    time = np.arange(t_start, t_stop, time_step)
    
    
    # Run the integration
    result = sys.integrate(x0=x0,
                           time=time,
                           time_step=time_step,
                           time_stabilize=time_stabilize,
                           stimulation=muscle_stimulation,
                           load=load)

    # Plotting
    plt.figure('Isotonic muscle experiment')
    plt.plot(result.v_ce, result.active_force)
    plt.title('Isotonic muscle experiment')
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Muscle Force')
    plt.grid()

    
    ##1.f"""
    """muscle_stim = np.linspace(0, 1, 5)
    #v_loads = np.linspace(0.01*load,10*load,40)


    fig, ax = plt.subplots()
    for stim in muscle_stim :

      
        result = sys.integrate(x0=x0,
                            time=time,
                            time_step=time_step,
                            time_stabilize=time_stabilize,
                            stimulation=stim,
                            load= load)

            
        ax.plot(result.v_ce ,result.active_force, label='Activation = '+stim.astype('str') )
    plt.title('Isotonic muscle experiment - Force(velocity)')
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Muscle Force [N]')
    plt.grid()
    leg = ax.legend()

    



    plt.show() """

    ##1.f - Max edition
    muscle_stim = np.linspace(0, 1, 5)
    v_loads = np.linspace(0.01*load,50*load,100)


    fig, ax = plt.subplots()
    for stim in muscle_stim :

        Force = []
        Force_active = [] 
        Force_passive = []
        velocity = []


        for v_load in v_loads:

            result = sys.integrate(x0=x0,
                            time=time,
                            time_step=time_step,
                            time_stabilize=time_stabilize,
                            stimulation=stim,
                            load= v_load)

            Force.append((result.active_force[-1] + result.passive_force[-1] ))
            Force_active.append((result.active_force[-1]))
            Force_passive.append((result.passive_force[-1] ))
            if (result.l_mtc[-1]>(sys.muscle.L_OPT + sys.muscle.L_SLACK)):
                velocity.append(np.max(result.v_ce))
            else:
                velocity.append(np.min(result.v_ce))
            
        ax.plot(velocity ,Force_active, label='Activation = '+stim.astype('str') )
    plt.title('Isotonic muscle experiment - Force(velocity)')
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Muscle Force [N]')
    plt.grid()
    leg = ax.legend()

    



    plt.show() 

    
    """v_load = np.linspace(0.01*load,30*load,50)
    Force = []
    Force_active = [] 
    Force_passive = []
    velocity = []

    for i in range (50):
        
        result = sys.integrate(x0=x0,
                               time=time,
                               time_step=time_step,
                               time_stabilize=time_stabilize,
                               stimulation=muscle_stimulation,
                               load=v_load[i])
        
        Force.append((result.active_force[-1] + result.passive_force[-1] ))
        Force_active.append((result.active_force[-1]))
        Force_passive.append((result.passive_force[-1] ))
        #ve = np.max(result.v_ce)
        #ve2 = np.min(result.v_ce)
        if (result.l_mtc[-1] > (sys.muscle.L_OPT + sys.muscle.L_SLACK)):
            velocity.append(np.max(result.v_ce))
            
        else:
            velocity.append(np.min(result.v_ce))
            

    ## Doesn't work for some reason, used [-1] for 1.d graph but might be false : why is vel = 0 ???

    #il y a un rebond sur la figure, c'est normal car la vélocité maximale est dans un premier temps celle qui va vers le bas quand le muscle se détend avec une faioble charge.
    #puis la vélocité maximale est celle quand la charge remonte quand celle-ci est assez lourde (on perd l'effet amortisseur du muscle -> comportement ressort)

    plt.figure('Isotonic muscle experiment')
    fig, ax = plt.subplots()
    ax.plot( v_load, velocity , label='Velocity(Tension)')
    plt.title('Isotonic muscle experiment')
    plt.xlabel('Load [kg]')
    plt.ylabel('Velocity [m/s]')
    plt.grid()
    leg = ax.legend()

    plt.figure('Isotonic muscle experiment')
    fig, ax = plt.subplots()
    ax.plot(  velocity,Force,  label='Force(Velocity')
    #ax.plot(  Force_active,velocity ,  label='Force_active(Velocity')
    #ax.plot(  Force_passive,velocity , label='Force_passive(Velocity')
    plt.title('Isotonic muscle experiment2')
    plt.xlabel('Velocity')
    plt.ylabel('Force')
    plt.grid()
    leg = ax.legend()

    plt.show()  """






    ##Double boucle, variation load et stim (incorrect??)
    """
    muscle_stim = np.linspace(0, 1, 5)
    
    v_load = np.linspace(0.01*load,10*load,30)
    Forces = []
    Force_actives = [] 
    Force_passives = []
    velocities = []


    for stim in muscle_stim :
        
        Force = []
        Force_active = [] 
        Force_passive = []
        velocity = []

        for load in v_load:
        
            result = sys.integrate(x0=x0,
                               time=time,
                               time_step=time_step,
                               time_stabilize=time_stabilize,
                               stimulation=stim,
                               load=load)
        
            Force.append((result.active_force[-1] + result.passive_force[-1] ))
            Force_active.append((result.active_force[-1]))
            Force_passive.append((result.passive_force[-1] ))

            if (result.l_mtc[-1] < (sys.muscle.L_OPT + sys.muscle.L_SLACK)):
                velocity.append(np.max(result.v_ce[-1]))
            else:
                velocity.append(np.min(result.v_ce[-1]))

        Forces.append(Force)
        Force_actives.append(Force_active)
        Force_passives.append(Force_passive)
        velocities.append(velocity)
    


    
    plt.figure('Isotonic muscle experiment')
    fig, ax = plt.subplots()
    for Force ,Force_active, Force_passive , stim, velocity in zip(Forces ,Force_actives, Force_passives, muscle_stim, velocities) :
        ax.plot(velocity, Force, label='Activation = '+stim.astype('str') )
        #ax.plot(velocity, Force_active, label='Active force')
        #ax.plot(stretches,Force_passive, label='Passive force')
    plt.title('Isotonic muscle experiment - Force(Velocity)')
    plt.xlabel('Velocity')
    plt.ylabel('Force')
    plt.grid()
    leg = ax.legend()"""






def exercise1():
    exercise1a()
    exercise1d()

    if DEFAULT["save_figures"] is False:
        plt.show()
    else:
        figures = plt.get_figlabels()
        print(figures)
        pylog.debug("Saving figures:\n{}".format(figures))
        for fig in figures:
            plt.figure(fig)
            save_figure(fig)
            plt.close(fig)


if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    exercise1()

