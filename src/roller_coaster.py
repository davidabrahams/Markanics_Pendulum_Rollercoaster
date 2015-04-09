import math
from numpy import array
from scipy.integrate import ode
from ode_solver import ODESolver
from matplotlib import pyplot as plt
from matplotlib import animation

__author__ = 'davidabrahams'

fig = plt.figure()
ax = plt.axes()
line, = ax.plot([], [], lw=2)
point, = ax.plot(None, None, 'o', markersize = 10)


def init():
    line.set_data(x, y)
    return line,


def animate(i):
    point.set_data(x[i], y[i])
    return point,


if __name__ == '__main__':
    solver = ODESolver()
    solns, t = solver.solve_ode()
    x, x_dot = solns[:, 0], solns[:, 1]
    y = solver.get_y_vals(x)
    x_range = max(x) - min(x)
    y_range = max(y) - min(y)
    screen_width = 1.2 * max(x_range, y_range)
    x_display = .5 * (max(x) + min(x) - screen_width), .5 * (max(x) + min(x) + screen_width)
    y_display = .5 * (max(y) + min(y) - screen_width), .5 * (max(y) + min(y) + screen_width)
    ax.set_xlim(x_display)
    ax.set_ylim(y_display)
    frame_num = int((solver.t_f - solver.t_0)/solver.dt)
    anim = animation.FuncAnimation(fig, animate, init_func=init, blit=True, frames=frame_num, interval=solver.dt * 1000)
    plt.show()