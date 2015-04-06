import math
from numpy import array
from scipy.integrate import ode
from ode_solver import ODESolver
from matplotlib import pyplot as plt
from matplotlib import animation

__author__ = 'davidabrahams'

fig = plt.figure()
ax = plt.axes(xlim=(0, 100), ylim=(-20, 80))
line, = ax.plot([], [], lw=2)


def init():
    line.set_data([], [])
    return line,


def animate(i):
    line.set_data(x[0:i], y[0:i])
    return line,


if __name__ == '__main__':
    solver = ODESolver()
    solns, t = solver.solve_ode()
    x, x_dot = solns[:, 0], solns[:, 1]
    y = solver.get_y_vals(x)
    anim = animation.FuncAnimation(fig, animate, init_func=init, blit=True, interval=solver.dt * 1000)
    plt.show()