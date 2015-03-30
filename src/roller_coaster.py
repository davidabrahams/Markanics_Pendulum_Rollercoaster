import math
from numpy import array
from scipy.integrate import ode
from matplotlib import pyplot as plt
from matplotlib import animation

__author__ = 'davidabrahams'

m = 1
g = 9.8
dt = 1 / 24.0

fig = plt.figure()
ax = plt.axes(xlim=(0, 100), ylim=(-70, 30))
line, = ax.plot([], [], lw=2)


def f_x(x):
    return math.cos(x) - 0.2*x


def f_prime_x(x):
    return -math.sin(x) - 0.2


def f_double_prime(x):
    return -math.cos(x)


def derivs(t, W):
    x, x_dot = W
    x_double_dot = (-g * f_prime_x(x) - f_double_prime(x) * f_prime_x(x) * x_dot ** 2) / (f_prime_x(x) ** 2 + 1)
    return [x_dot, x_double_dot]


def solve_ode():

    x_0 = 0
    x_dot_0 = 1
    initials = [x_0, x_dot_0]

    t_0 = 0
    t_f = 20

    T = []
    Solns = []

    solver = ode(derivs)
    solver.set_initial_value(initials, t_0)
    solver.set_integrator('dopri5')

    while solver.successful() and solver.t < t_f:
        solver.integrate(solver.t + dt)
        T.append(solver.t)
        Solns.append(solver.y)

    return array(Solns), array(T)


def get_y_vals(x_vals):
    return array([f_x(x_val) for x_val in x_vals])


solns, t = solve_ode()
x, x_dot = solns[:, 0], solns[:, 1]
y = get_y_vals(x)


def init():
    line.set_data([], [])
    return line,


def animate(i):
    line.set_data(x[0:i], y[0:i])
    return line,


if __name__ == '__main__':
    anim = animation.FuncAnimation(fig, animate, init_func=init, blit=True, interval=dt * 1000)
    plt.show()