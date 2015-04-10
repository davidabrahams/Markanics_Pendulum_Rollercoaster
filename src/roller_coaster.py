from numpy import array, linspace, cos, sin
from scipy.integrate import ode
from ode_solver import ODESolver
from matplotlib import pyplot as plt
from matplotlib import animation

__author__ = 'davidabrahams'

fig = plt.figure()
ax_1 = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)
ax_2 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
track, = ax_1.plot([], [], lw=3)
cart, = ax_1.plot(None, None, 'o', markersize=10)
rod, = ax_1.plot([], [], lw=2)
rider, = ax_1.plot(None, None, 'o', markersize=7)
gpe_line, = ax_2.plot([], [], lw=2, label='GPE')
ke_line, = ax_2.plot([], [], lw=2, label='KE Cart')
ke_rider, = ax_2.plot([], [], lw=2, label='KE Rider')
sum_nrg_line, = ax_2.plot([], [], lw=2, label='Total NRG')



def init():
    x_min, x_max = ax_1.get_xlim()
    x_vals = linspace(x_min, x_max, num=200)
    y_vals = solver.get_y_vals(x_vals)
    track.set_data(x_vals, y_vals)
    return track,


def animate(i):
    x1, y1, x2, y2 = x_c[i], y_c[i], x_r[i], y_r[i]
    cart.set_data(x1, y1)
    rod.set_data([x1, x2], [y1, y2])
    rider.set_data(x2, y2)
    gpe_line.set_data(t[0:i], gpe[0:i])
    ke_line.set_data(t[0:i], ke[0:i])
    ke_rider.set_data(t[0:i], ke2[0:i])
    sum_nrg_line.set_data(t[0:i], sum_nrg[0:i])
    return cart, rod, rider, gpe_line, ke_line, ke_rider, sum_nrg_line


if __name__ == '__main__':
    solver = ODESolver()
    solns, t = solver.solve_ode()
    x_c, x_dot, theta, theta_dot = solns[:, 0], solns[:, 1], solns[:, 2], solns[:, 3]
    y_c = solver.get_y_vals(x_c)
    x_r = x_c + solver.l * cos(theta)
    y_r = y_c + solver.l * sin(theta)
    y_dot = solver.f_prime_x(x_c) * x_dot
    gpe = solver.m_c*ODESolver.g*y_c + solver.m_r*ODESolver.g*y_r
    ke = 0.5 * solver.m_c * (x_dot**2 + y_dot**2)
    ke2 = 0.5 * solver.m_r * ((x_dot - solver.l*theta_dot*sin(theta))**2 + (y_dot + solver.l*theta_dot*cos(theta))**2)
    sum_nrg = gpe + ke + ke2
    x_range = max(x_c) - min(x_c)
    y_range = max(y_c) - min(y_c)
    screen_width = 1.2 * max(x_range, y_range)
    x_display = .5 * (max(x_c) + min(x_c) - screen_width), .5 * (max(x_c) + min(x_c) + screen_width)
    y_display = .5 * (max(y_c) + min(y_c) - screen_width), .5 * (max(y_c) + min(y_c) + screen_width)
    ax_1.set_xlim(x_display)
    ax_1.set_ylim(y_display)
    ax_2.set_xlim((min(t), max(t)))
    ax_2.set_ylim((1.2*min(gpe), 1.2*max(gpe)))
    ax_2.legend()
    frame_num = int((solver.t_f - solver.t_0)/solver.dt)
    anim = animation.FuncAnimation(fig, animate, init_func=init, blit=True, frames=frame_num, interval=solver.dt * 1000)
    ax_1.set_aspect('equal')
    plt.show()