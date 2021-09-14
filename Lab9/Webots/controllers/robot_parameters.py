"""Robot parameters"""

import numpy as np
import cmc_pylog as pylog
from math import *


class RobotParameters(dict):
    """Robot parameters""" 

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


    CONV_RATE = 20.0
    WEIGHT_LINE = 10
    WEIGHT_COL = 10
    WEIGHT_COUPLING = 30
    
    NOMINAL_AMPLITUDE_BODY = 2
    NOMINAL_AMPLITUDE_LEGS = 2

    DRIVE_UP_BODY = 5.0
    DRIVE_UP_LEG = 3.0

    DRIVE_DOWN_BODY = 1.0
    DRIVE_DOWN_LEG = 1.0

    def __init__(self, parameters):
        super(RobotParameters, self).__init__() 

        print(parameters)
        # Initialise parameters
        self.n_body_joints = parameters.n_body_joints
        self.n_legs_joints = parameters.n_legs_joints
        self.n_joints = self.n_body_joints + self.n_legs_joints
        self.n_oscillators_body = 2*self.n_body_joints
        self.n_oscillators_legs = self.n_legs_joints
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs
        self.freqs = np.zeros(self.n_oscillators)

        self.coupling_weights = np.zeros([
            self.n_oscillators,
            self.n_oscillators
        ])
        self.phase_bias = np.zeros([self.n_oscillators, self.n_oscillators])
        self.rates = np.zeros(self.n_oscillators)
        self.nominal_amplitudes = np.zeros(self.n_oscillators)
        self.update(parameters)

    def update(self, parameters):
        """Update network from parameters"""
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # theta_i
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i

    def set_frequencies(self, parameters):
        """Set frequencies"""
        if(parameters.drive<=self.DRIVE_UP_BODY and parameters.drive>= self.DRIVE_DOWN_BODY):
            freq_body = 0.3 + 0.2*parameters.drive
        else:
            freq_body = 0.0

        if(parameters.drive<=self.DRIVE_UP_LEG and parameters.drive>= self.DRIVE_DOWN_LEG):
            freq_leg = 0.2*parameters.drive
        else:
            freq_leg = 0.0

        freq_body = freq_body *np.ones(self.n_oscillators_body)
        freq_leg = freq_leg*np.ones(self.n_oscillators_legs)

        if parameters.frequency is not None:
            freq_body = parameters.frequency*np.ones(self.n_oscillators_body)
            freq_leg  = np.zeros(self.n_oscillators_legs)
            
        self.freqs = np.concatenate((freq_body,freq_leg))

        print ('frequencies : ')
        print(self.freqs)

    def set_coupling_weights(self, parameters):
        """Set coupling weights"""

        weight_line = self.WEIGHT_LINE
        weight_col = self.WEIGHT_COL
        weight_body_coupling = self.WEIGHT_COUPLING
      
        for i in range(self.n_body_joints):
            for j in range(i-1,i+2):
                if (j>=0 and j< self.n_body_joints):
                    self.coupling_weights[i,j] = abs(j-i)*weight_line
                    self.coupling_weights[i,i + self.n_body_joints] = weight_col
                    self.coupling_weights[i + self.n_body_joints,j+self.n_body_joints] = self.coupling_weights[i,j] 
                    self.coupling_weights[i + self.n_body_joints,i] = weight_col

        for i in range(self.n_oscillators_body , self.n_oscillators):
            for j in range (self.n_oscillators_body , self.n_oscillators):
                if((j is not i)): 
                    self.coupling_weights[i,j] = weight_col           
                    if((j+i-2*self.n_oscillators_body)%(self.n_oscillators_legs-1) is 0):
                        self.coupling_weights[i,j] = 0

        for i in range(1,6):
            self.coupling_weights[self.n_oscillators_body, i] = weight_body_coupling
            self.coupling_weights[self.n_oscillators_body+1, i+self.n_body_joints] = weight_body_coupling

            for i in range(6,10):
                self.coupling_weights[self.n_oscillators_body+2, i] = weight_body_coupling
                self.coupling_weights[self.n_oscillators_body+3, i+self.n_body_joints] = weight_body_coupling


        print ('Coupling weight')
        print(self.coupling_weights)

    def set_phase_bias(self, parameters):
        """Set phase bias"""
        
        if (parameters.phase_lag is not None):
            lag = parameters.phase_lag
        else:
            lag = 2*pi/self.n_body_joints*np.ones(self.n_joints)
            print('fdgbsfdbdxxgb')
       
        for i in range(self.n_body_joints):
            for j in range(i-1,i+2):
                if (j>=0 and j< self.n_body_joints):
                    self.phase_bias[i,j] = lag[i]*(i-j) 
                    self.phase_bias[i,i + self.n_body_joints] = pi
                    self.phase_bias[i + self.n_body_joints,j+self.n_body_joints] = self.phase_bias[i,j]
                    self.phase_bias[i + self.n_body_joints,i] = pi

        if ((parameters.phase_lag is None) or (len(parameters.phase_lag)!=self.n_joints)):
            lag = pi*np.ones(self.n_joints)
            print('lag leg -> body = pi')

        for i in range(self.n_oscillators_body , self.n_oscillators):
            for j in range (self.n_oscillators_body , self.n_oscillators):
                if((j is not i)): 
                    self.phase_bias[i,j] = pi 
                    if((j+i-2*self.n_oscillators_body)%(self.n_oscillators_legs-1) is 0):
                        self.phase_bias[i,j] = 0

        for i in range(1,6):
            self.phase_bias[self.n_oscillators_body, i] = lag[self.n_body_joints]
            self.phase_bias[self.n_oscillators_body + 1, i+self.n_body_joints] = lag[self.n_body_joints+1]

        for i in range(6,10):
            self.phase_bias[self.n_oscillators_body + 2, i] = lag[self.n_body_joints+2]
            self.phase_bias[self.n_oscillators_body + 3, i+self.n_body_joints] = lag[self.n_body_joints+3]
        
        if(parameters.backward is not None):
           self.phase_bias[:self.n_oscillators_body] = - self.phase_bias[:self.n_oscillators_body]
           
        print ('Phase bias : ')
        print(self.phase_bias)



    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        conv_rate = self.CONV_RATE

        self.rates=conv_rate * np.ones(self.n_oscillators)
        print ('Convergence rates')
        print(self.rates) 


    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes"""
  

        if(parameters.drive<=self.DRIVE_UP_BODY and parameters.drive>= self.DRIVE_DOWN_BODY):
            body_amplitude = 0.065*parameters.drive + 0.196
        else:
            body_amplitude = 0.0

        if(parameters.drive<=self.DRIVE_UP_LEG and parameters.drive>= self.DRIVE_DOWN_LEG):
            legs_amplitude = 0.131*parameters.drive
        else:
            legs_amplitude = 0.0
        
        osc_min =  0.065*self.DRIVE_DOWN_BODY + 0.196
        osc_max = 0.065*self.DRIVE_UP_BODY +0.196

        body_amplitude = body_amplitude*np.ones(self.n_oscillators_body)
        legs_amplitude = legs_amplitude*np.ones(self.n_oscillators_legs)


        if(parameters.amplitude_gradient is not None):
            
            if(len(parameters.amplitude_gradient)!=2):
                raise Exception('There should be 2 parameters to create the gradient not {}'.format(len(parameters.amplitude_gradient)))

            if (min(parameters.amplitude_gradient)<osc_min or min(parameters.amplitude_gradient)>osc_max):
                osc_min =  0.065*self.DRIVE_DOWN_BODY + 0.196
            else:
                osc_min = min(parameters.amplitude_gradient)
            
            if(max(parameters.amplitude_gradient)>osc_max or max(parameters.amplitude_gradient)<osc_min):
                osc_max = 0.065*self.DRIVE_UP_BODY +0.196
            else:
                osc_max = max(parameters.amplitude_gradient)

            if (parameters.amplitude_gradient.index(min(parameters.amplitude_gradient))==0):
                gradient = np.linspace(osc_min, osc_max , self.n_body_joints)
            else:
                gradient = np.linspace(osc_max, osc_min  , self.n_body_joints)

            for i in range(self.n_body_joints):
                body_amplitude[i] = gradient[i]
                body_amplitude[i+self.n_body_joints] = gradient[i]


        if (parameters.amplitudes is not None):
            for i in range(self.n_body_joints):
                body_amplitude[i] = parameters.amplitudes[i]
                body_amplitude[i+self.n_body_joints] = parameters.amplitudes[i]

        # ? if turn > 0, the robot turns right. Hence, The  value of the amplitude of the joint of the left side increase 
        # ? and the one on the right side of the joints decrease and vice-versa with tun < 0
        
        if (parameters.turn is not None):
            if (abs(parameters.turn)>np.min(body_amplitude)):
                print('lower amplitude th')
                turn = np.sign(parameters.turn)*1
                body_amplitude[:self.n_body_joints] = (1+turn)*body_amplitude[:self.n_body_joints] 
                body_amplitude[self.n_body_joints:2*self.n_body_joints]= (1-turn)* body_amplitude[self.n_body_joints:2*self.n_body_joints]
            else:
                print('lower amplitude than the maximum one')
                body_amplitude[:self.n_body_joints] = (1+parameters.turn)*body_amplitude[:self.n_body_joints] 
                body_amplitude[self.n_body_joints:2*self.n_body_joints]= (1-parameters.turn)* body_amplitude[self.n_body_joints:2*self.n_body_joints]

            if (abs(parameters.turn)>np.min(legs_amplitude)):
                turn = np.sign(parameters.turn)*1
                legs_amplitude[:(int)(self.n_legs_joints/2)] =  (1+turn)*legs_amplitude[:(int)(self.n_legs_joints/2)]
                legs_amplitude[(int)(self.n_legs_joints/2):self.n_legs_joints] = (1-turn)*legs_amplitude[(int)(self.n_legs_joints/2):self.n_legs_joints]
            else:
                print('lower amplitude than the maximum one')
                legs_amplitude[:(int)(self.n_legs_joints/2)] =  (1+parameters.turn)*legs_amplitude[:(int)(self.n_legs_joints/2)]
                legs_amplitude[(int)(self.n_legs_joints/2):self.n_legs_joints] = (1-parameters.turn)*legs_amplitude[(int)(self.n_legs_joints/2):self.n_legs_joints]
            

        self.nominal_amplitudes = np.concatenate((body_amplitude,legs_amplitude))

        print ('Nominal amplitudes')
        print(self.nominal_amplitudes )
        #pylog.warning("Nominal amplitudes must be set")