""" Lab 4 System Animation"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


class SystemAnimation(object):
    """ SystemAnimation """

    def __init__(
            self,
            res_sys,
            pendulum_sys,
            fps=30,
            handler="Simulation",
            show_parameters=True
    ):
        super(SystemAnimation, self).__init__()

        self.pendulum_sys = pendulum_sys
        self.time = res_sys.time
        self.state = res_sys.state
        self.show_parameters = show_parameters

        self.fps = fps
        self.fig, self.ax = plt.subplots(num=handler)

        self.anims = self.animation_objects()
        t_max = self.time[-1]
        dt = 1 / float(fps)

        self.anim_link = animation.FuncAnimation(
            self.fig, self._animate, np.arange(0, t_max, dt),
            interval=1e3 / float(fps), blit=True
        )
        plt.title(str(handler))
        plt.axis('scaled')
        plt.axis('off')
        limit = 1.15 * self.pendulum_sys.parameters.L
        if limit < 0.5:
            limit = .5
        plt.axis([-limit, limit,
                  -limit, 0.75])
        plt.grid(False)
        return

    def animation_objects(self):
        """ Create and return animation objects """

        blue = (0.0, 0.3, 1.0, 1.0)
        # Pendulum
        pendulum = self.pendulum_sys.pose()
        print(pendulum)
        _length = self.pendulum_sys.parameters.L

        self.line, = self.ax.plot(
            pendulum[:, 0],
            pendulum[:, 1],
            color=blue,
            linewidth=5,
            animated=True
        )
        # Mass
        self.m, = self.ax.plot(
            self.pendulum_sys.origin[0], _length,
            color=blue, marker='o', markersize=12.5, animated=True)
        # Base
        self.ax.plot([-0.5, 0.5], self.pendulum_sys.origin,
                     c='g', linewidth=7.5)
        # Springs/Dampers
        spring_damper_1 = self.ax.plot(pendulum[:, 0] - np.array([_length/2, 0]),
                                       pendulum[:, 1]/2.,
                                       color='r',
                                       linewidth=3.5,
                                       animated=True)[0]

        spring_damper_2 = self.ax.plot(pendulum[:, 0] + np.array([_length/2, 0]),
                                       pendulum[:, 1]/2.,
                                       color='r',
                                       linewidth=3.5,
                                       animated=True)[0]

        # Time
        time = self.ax.text(-0.5, 0.05, "Time: 0.0",
                            fontsize=14, animated=True)

        # Parameters
        if self.show_parameters:
            show_params = self.pendulum_sys.parameters.showParameters
            # build a rectangle in axes coords
            left, width = -0.7, .5
            bottom, height = 0.5, .5
            right = left + width
            top = bottom + height
            params = self.ax.text(right, top,
                                  show_params(),
                                  fontsize=12, animated=False,
                                  horizontalalignment='left',
                                  verticalalignment='top',
                                  transform=self.ax.transAxes)

        return [self.line, self.m] + [spring_damper_1] + \
            [spring_damper_2] + [time]

    @staticmethod
    def animate():
        """Animate System"""
        plt.show()
        return

    def _animate(self, time):
        """ Animation """
        index = np.argmin((self.time - time)**2)
        self.pendulum_sys.state = self.state[index]

        pendulum = self.pendulum_sys.pose()
        _length = self.pendulum_sys.parameters.L

        # Pendulum
        self.anims[0].set_xdata(pendulum[:, 0])
        self.anims[0].set_ydata(pendulum[:, 1])

        # Mass
        self.anims[1].set_xdata([pendulum[1, 0]])
        self.anims[1].set_ydata([pendulum[1, 1]])

        # Springs
        self.anims[2].set_xdata((pendulum[:, 0] - np.array([_length, 0]))/2)
        self.anims[2].set_ydata(pendulum[:, 1]/2)

        self.anims[3].set_xdata((pendulum[:, 0] + np.array([_length, 0]))/2)
        self.anims[3].set_ydata(pendulum[:, 1]/2)

        # Text
        self.anims[4].set_text("Time: {:.1f}".format(self.time[index]))

        return self.anims

