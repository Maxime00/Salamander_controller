"""Run network without Webots"""

import time
import numpy as np
from math import *
import matplotlib.pyplot as plt
import cmc_pylog as pylog
from network import SalamanderNetwork
from save_figures import save_figures
from parse_args import save_plots
from simulation_parameters import SimulationParameters


def run_network(duration, update=False, drive=0):
    """Run network without Webots and plot results"""
    # Simulation setup
    timestep = 0.1
    times = np.arange(0, duration, timestep)
    n_iterations = len(times)
    parameters = SimulationParameters(
        drive=drive,
        amplitude_gradient=None,#[4.0,2.0] 
        phase_lag=None,
        turn=None,
        backward = None,
    )
    network = SalamanderNetwork(timestep, parameters)
    osc_left = np.arange(10)
    osc_right = np.arange(10, 20)
    osc_legs = np.arange(20, 24)

    # Logs
    phases_log = np.zeros([
        n_iterations,
        len(network.state.phases)
    ])
    phases_log[0, :] = network.state.phases
    amplitudes_log = np.zeros([
        n_iterations,
        len(network.state.amplitudes)
    ])
    amplitudes_log[0, :] = network.state.amplitudes
    freqs_log = np.zeros([
        n_iterations,
        len(network.parameters.freqs)
    ])
    freqs_log[0, :] = network.parameters.freqs
    outputs_log = np.zeros([
        n_iterations,
        len(network.get_motor_position_output())
    ])
    outputs_log[0, :] = network.get_motor_position_output()
    print('tada \n')
    # print((outputs_log).shape)
    # Run network ODE and log data
    tic = time.time()
    for i, _ in enumerate(times[1:]):
        if update:
            network.parameters.update(
                SimulationParameters(
                    drive=2*drive,
                    amplitude_gradient=None,#[4.0,2.0] 
                    phase_lag=None,
                    turn=None,
                    backward = None,
                    # amplitude_gradient=None,

                )
            )
        network.step()
        phases_log[i+1, :] = network.state.phases
        amplitudes_log[i+1, :] = network.state.amplitudes
        outputs_log[i+1, :] = network.get_motor_position_output()
        freqs_log[i+1, :] = network.parameters.freqs
    toc = time.time()
    # Network performance
    pylog.info("Time to run simulation for {} steps: {} [s]".format(
        n_iterations,
        toc - tic
    ))
    print(outputs_log)
    # Implement plots of network results

    pylog.warning("Implement plots")

    plt.subplot(5,1,1)
    plt.plot(times, phases_log%(2*pi))
    plt.xlabel('time (s)')
    plt.ylabel('phases of the joints (rad)')
    plt.title('phases of the joints')
    plt.legend()

    plt.subplot(5,1,2)
    plt.plot(times, outputs_log)
    plt.xlabel('time (s)')
    plt.ylabel('Spinal joint angle (rad)')
    plt.title('Spinal joint angle')
    plt.legend()

    plt.subplot(5,1,3)
    plt.plot(times, amplitudes_log)
    plt.xlabel('time (s)')
    plt.ylabel('Amplitude of the joints')
    plt.title('Amplitude of the joints')
    plt.legend()   


    plt.subplot(5,1,4)
    plt.plot(times, freqs_log)
    plt.xlabel('time (s)')
    plt.ylabel('Intrinsic frequency of the joints (Hz)')
    plt.title('Intrinsic frequency of the joints')
    plt.legend()   


def main(plot):
    """Main"""

    run_network(duration=10, update = False , drive = 1.0)

    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()
    save_figures()


if __name__ == '__main__':
    main(plot=not save_plots())

