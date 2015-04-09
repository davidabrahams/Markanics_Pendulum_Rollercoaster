import math
from numpy import array
from scipy.integrate import ode

__author__ = 'davidabrahams'


class ODESolver:

    g = 9.8

    def __init__(self):
        self.m = 1
        self.dt = 1 / 24.0
        self.t_0 = 0
        self.t_f = 20

    def f_x(self, x):
        return 10 * math.cos(x) + 0.25 * (x - 10) ** 2


    def f_prime_x(self, x):
        return -10 * math.sin(x) + .5 * (x - 10)


    def f_double_prime(self, x):
        return -10 * math.cos(x) + .5


    def solve_ode(self):

        def derivs(t, W):
            x, x_dot = W
            x_double_dot = (-ODESolver.g * self.f_prime_x(x) - self.f_double_prime(x) * self.f_prime_x(
                x) * x_dot ** 2) / (self.f_prime_x(x) ** 2 + 1)
            return [x_dot, x_double_dot]

        x_0 = 0
        x_dot_0 = 1
        initials = [x_0, x_dot_0]

        T = []
        Solns = []

        solver = ode(derivs)
        solver.set_initial_value(initials, self.t_0)
        solver.set_integrator('dopri5')

        while solver.successful() and solver.t < self.t_f:
            solver.integrate(solver.t + self.dt)
            T.append(solver.t)
            Solns.append(solver.y)

        return array(Solns), array(T)

    def get_y_vals(self, x_vals):
        return array([self.f_x(x_val) for x_val in x_vals])
