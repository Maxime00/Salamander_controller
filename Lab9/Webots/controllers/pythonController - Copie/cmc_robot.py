"""CMC robot"""

import numpy as np
from network import SalamanderNetwork
from experiment_logger import ExperimentLogger




class SalamanderCMC(object):
    """Salamander robot for CMC"""

    N_BODY_JOINTS = 10
    N_LEGS = 4

    #! Exercise 9g
    DETECTION_WATER = False
    DETECTION_GROUND = True

    #! Exercise 9 extra
    AUTO_DETECTION = False
    THRESHOLD = 3

    def __init__(self, robot, n_iterations, parameters, logs="logs/log.npz", ground_detect = False):
        super(SalamanderCMC, self).__init__()
        self.robot = robot
        timestep = int(robot.getBasicTimeStep())
        self.network = SalamanderNetwork(1e-3*timestep, parameters)


        #! Modification for the 9g:
        self.parameters = parameters

        self.detect = ground_detect
        # Position sensors
        self.position_sensors = [
            self.robot.getPositionSensor('position_sensor_{}'.format(i+1))
            for i in range(self.N_BODY_JOINTS)
        ] + [
            self.robot.getPositionSensor('position_sensor_leg_{}'.format(i+1))
            for i in range(self.N_LEGS)
        ]
        for sensor in self.position_sensors:
            sensor.enable(timestep)

        # GPS
        self.gps = robot.getGPS("fgirdle_gps")
        self.gps.enable(timestep)

        # Get motors
        self.motors_body = [
            self.robot.getMotor("motor_{}".format(i+1))
            for i in range(self.N_BODY_JOINTS)
        ]
        self.motors_legs = [
            self.robot.getMotor("motor_leg_{}".format(i+1))
            for i in range(self.N_LEGS)
        ]

        # Set motors
        for motor in self.motors_body:
            motor.setPosition(0)
            motor.enableForceFeedback(timestep)
            motor.enableTorqueFeedback(timestep)
        for motor in self.motors_legs:
            motor.setPosition(-np.pi/2)
            motor.enableForceFeedback(timestep)
            motor.enableTorqueFeedback(timestep)

        # Iteration counter
        self.iteration = 0

        # Logging
        self.log = ExperimentLogger(
            n_iterations,
            n_links=1,
            n_joints=self.N_BODY_JOINTS+self.N_LEGS,
            filename=logs,
            timestep=1e-3*timestep,
            **parameters
        )

    def log_iteration(self):
        """Log state"""
        self.log.log_link_positions(self.iteration, 0, self.gps.getValues())
        for i, motor in enumerate(self.motors_body):
            # Position
            self.log.log_joint_position(
                self.iteration, i,
                self.position_sensors[i].getValue()
            )
            # # Velocity
            # self.log.log_joint_velocity(
            #     self.iteration, i,
            #     motor.getVelocity()
            # )
            # Command
            self.log.log_joint_cmd(
                self.iteration, i,
                motor.getTargetPosition()
            )
            # Torque
            self.log.log_joint_torque(
                self.iteration, i,
                motor.getTorqueFeedback()
            )
            # Torque feedback
            self.log.log_joint_torque_feedback(
                self.iteration, i,
                motor.getTorqueFeedback()
            )
        for i, motor in enumerate(self.motors_legs):
            # Position
            self.log.log_joint_position(
                self.iteration, 10+i,
                self.position_sensors[10+i].getValue()
            )
            # # Velocity
            # self.log.log_joint_velocity(
            #     self.iteration, i,
            #     motor.getVelocity()
            # )
            # Command
            self.log.log_joint_cmd(
                self.iteration, 10+i,
                motor.getTargetPosition()
            )
            # Torque
            self.log.log_joint_torque(
                self.iteration, 10+i,
                motor.getTorqueFeedback()
            )
            # Torque feedback
            self.log.log_joint_torque_feedback(
                self.iteration, 10+i,
                motor.getTorqueFeedback()
            )

    def step(self):
        """Step"""
        # Increment iteration
        self.iteration += 1
        thresh = 0.3
        
        if (self.gps.getValues()[0]>thresh and self.DETECTION_WATER):
            DETECTION_WATER = False
        
            self.parameters.drive = 4.0
            self.network.parameters.update( self.parameters)
            print('Detection of the water enabled')

        if(self.gps.getValues()[0]<thresh and self.DETECTION_GROUND):
            DETECTION_GROUND = False
            print('Detection of the ground enabled')
            self.parameters.drive = 1.5
            self.network.parameters.update( self.parameters)

        if self.AUTO_DETECTION:
            environment = self.environment_feedback()
            if(environment is not None): # Wait to have enough data to process
    
                if(self.parameters.drive <= self.network.parameters.DRIVE_UP_LEG): 
                    #? Case when the robot has a walking behavior 
                    if (environment<self.THRESHOLD):
                        #! Transition to a swimming behavior
                        print('SWIMMING!!!!!!!!!!!')
                        self.parameters.drive = 4
                        self.network.parameters.update(self.parameters)
                    print('WALKING!!!!!!!!!!!')
                    

                else:
                    #? Case when the robot has a swimming behavior 
                    if (environment>self.THRESHOLD):
                        #! Transition to a walking behavior
                        print('WALKING!!!!!!!!!!!')
                        self.parameters.drive = 2
                        self.network.parameters.update(self.parameters)
                    print('SWIMMING!!!!!!!!!!!')


            #print(self.environment_feedback())


        # Update network
        self.network.step()
        positions = self.network.get_motor_position_output()

        # Update control
        for i in range(self.N_BODY_JOINTS):
            self.motors_body[i].setPosition(positions[i])
        for i in range(self.N_LEGS):
            self.motors_legs[i].setPosition(positions[self.N_BODY_JOINTS+i] - np.pi/2)

        # Log data
        self.log_iteration()

    def environment_feedback(self):
        size_min = 100
        if(self.iteration>=5*size_min):
            results = self.log.joints[(self.iteration-size_min):self.iteration][:,:,4] # Extraction of the torque feedback
        else:
            return None 
            
        new_results = []
        for i in range(len(results)-1):
            new_results.append(results[i+1,:]-results[i,:])

        new_results = np.sum(np.mean(np.abs(new_results),1))
        return new_results


