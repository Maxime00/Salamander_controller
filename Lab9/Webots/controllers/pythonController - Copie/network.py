"""Oscillator network ODE"""

import numpy as np

from math import *
from solvers import euler, rk4
from robot_parameters import RobotParameters

#FREQ_LIMIT_LEG = 0.7
#FREQ_LIMIT_BODY = 1.3
#DRIVE_LEG = 2.0
#DRIVE_BODY = DRIVE_LEG*FREQ_LIMIT_LEG/FREQ_LIMIT_BODY


def network_ode(_time, state, parameters):
    """Network_ODE

    returns derivative of state (phases and amplitudes)

    """
    #print(parameters)
    phases_dot = [] 
    amplitudes_dot = []
    
    phases = state[:parameters.n_oscillators]
    amplitudes = state[parameters.n_oscillators:2*parameters.n_oscillators]

    backward = None
    turning = None

    # ? This part is only if you want to trigger the going backward behavior
    #backward = 1
    # ? This part is only if you want to trigger the turning behavior behavior
    # ? the values of turning are between [-1,1]
    # turning = 0


    for i in range (parameters.n_oscillators):
        sum_amplitude = 0
        for j in range (parameters.n_oscillators):
            if (backward is None):
                sum_amplitude = sum_amplitude + amplitudes[j]*parameters.coupling_weights[i][j]*sin(phases[j]-phases[i]-parameters.phase_bias[i][j])
            else:
                sum_amplitude = sum_amplitude + amplitudes[j]*parameters.coupling_weights[i][j]*sin(phases[j]-phases[i]+parameters.phase_bias[i][j])

        phases_dot.append(2*pi*parameters.freqs[i] + sum_amplitude)
        amplitudes_dot.append(parameters.rates[i]*(parameters.nominal_amplitudes[i]-amplitudes[i]))

        """if (i in range (parameters.n_oscillators_body, parameters.n_oscillators)):
            if(parameters.freqs[i]>=0 and parameters.freqs[i]<=FREQ_LIMIT_LEG):
                phases_dot.append(2*pi*parameters.freqs[i] + DRIVE_LEG*parameters.freqs[i]*sum_amplitude)
                amplitudes_dot.append(DRIVE_LEG*parameters.freqs[i]*parameters.rates[i]*(parameters.nominal_amplitudes[i]-amplitudes[i]))
            else:
                phases_dot.append(0)
                amplitudes_dot.append(0)
        else:
            if(parameters.freqs[i]>=0 and parameters.freqs[i]<=FREQ_LIMIT_BODY):
                phases_dot.append(2*pi*parameters.freqs[i] + DRIVE_BODY*parameters.freqs[i]*sum_amplitude)
                amplitudes_dot.append(DRIVE_BODY*parameters.freqs[i]*parameters.rates[i]*(parameters.nominal_amplitudes[i]-amplitudes[i]))

            else:
                phases_dot.append(0)
                amplitudes_dot.append(0)
        """

        # TODO make a turning function that actually works
        """if (turning is not None):
            if (turning>0):     # ! movement to the right
                amplitudes[parameters.n_body_joints:2*parameters.n_body_joints] = turning*amplitudes[parameters.n_body_joints:2*parameters.n_body_joints]
            else:               # ! movement to the left
                amplitudes[:parameters.n_body_joints] = -turning*amplitudes[:parameters.n_body_joints] 
        """

    return np.concatenate([phases_dot, amplitudes_dot])


def motor_output(phases, amplitudes, nb_body_joints, nb_leg_joints, frequencies):
    """Motor output"""
    spinal_joint_angle = []
    my_amplitude = 0

    # TODO Implement the frequency dependent amplitudes with the drive
    for nb_joint in range(nb_body_joints):

        my_amplitude = amplitudes[nb_joint]*(1+cos(phases[nb_joint]))
        my_amplitude = my_amplitude - amplitudes[nb_joint+nb_body_joints]*(1+cos(phases[nb_joint+nb_body_joints]))
        """if (frequencies[nb_joint]>=0 and frequencies[nb_joint]<=FREQ_LIMIT_BODY):
            my_amplitude = amplitudes[nb_joint]*(1+cos(phases[nb_joint]))
        else:
            my_amplitude = 0
        
        if(frequencies[nb_joint+nb_body_joints]>=0 and frequencies[nb_joint+nb_body_joints]<=FREQ_LIMIT_BODY):
            my_amplitude = my_amplitude + amplitudes[nb_joint+nb_body_joints]*(1+cos(phases[nb_joint+nb_body_joints]))
        else :
            my_amplitude = my_amplitude + 0"""

        spinal_joint_angle.append(my_amplitude)
    


    # TODO Implement the frequencies dependent drive for the leg output of the motor
    for nb_leg in range(2*nb_body_joints,2*nb_body_joints+nb_leg_joints):
        if (amplitudes[nb_leg] == 0):
            spinal_joint_angle.append(0)    
        else:
            spinal_joint_angle.append(-phases[nb_leg])
        """if (frequencies[nb_leg]>=0 and frequencies[nb_leg]<=FREQ_LIMIT_LEG):
            my_amplitude = amplitudes[nb_leg]*(1+cos(phases[nb_leg]))
        else:
            my_amplitude = 0
        """
       


    # print(spinal_joint_angle)
    return (spinal_joint_angle)


class ODESolver(object):
    """ODE solver with step integration"""

    def __init__(self, ode, timestep, solver=rk4):
        super(ODESolver, self).__init__()
        self.ode = ode
        self.solver = solver
        self.timestep = timestep
        self._time = 0

    def integrate(self, state, *parameters):

        """Step"""
        diff_state = self.solver(
            self.ode,
            self.timestep,
            self._time,
            state,
            *parameters
        )
        
        self._time += self.timestep
        return diff_state

    def time(self):
        """Time"""
        return self._time


class RobotState(np.ndarray):
    """Robot state"""

    def __init__(self, *_0, **_1):
        super(RobotState, self).__init__()
        self[:] = 0.0

    @classmethod
    def salamandra_robotica_2(cls):
        """State of Salamandra robotica 2"""
        return cls(2*24, dtype=np.float64, buffer=np.zeros(2*24))

    @property
    def phases(self):
        """Oscillator phases"""
        return self[:24]

    @phases.setter
    def phases(self, value):
        self[:24] = value

    @property
    def amplitudes(self):
        """Oscillator phases"""
        return self[24:]

    @amplitudes.setter
    def amplitudes(self, value):
        self[24:] = value


class SalamanderNetwork(ODESolver):
    """Salamander oscillator network"""

    def __init__(self, timestep, parameters):
        super(SalamanderNetwork, self).__init__(
            ode=network_ode,
            timestep=timestep,
            solver=euler  # Feel free to switch between Euler (euler) or
                        # Runge-Kutta (rk4) integration methods
        )
        # States
        self.state = RobotState.salamandra_robotica_2()
        # Parameters
        self.parameters = RobotParameters(parameters)
        # Set initial state
        self.state.phases = 1e-4*np.random.ranf(self.parameters.n_oscillators)

    def step(self):
        """Step"""
        self.state += self.integrate(self.state, self.parameters)

    def get_motor_position_output(self):
        """Get motor position"""
        #print(self.parameters)
        return motor_output(self.state.phases, self.state.amplitudes, self.parameters.n_body_joints, self.parameters.n_legs_joints, self.parameters.freqs)

