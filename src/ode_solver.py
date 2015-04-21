import math
from numpy import array, where, less, greater
from numpy import empty
from numpy import linalg
from numpy import dot
from numpy import sin, cos
from scipy.integrate import ode

__author__ = 'davidabrahams'


class ODESolver:

    g = 9.8

    def __init__(self):
        self.m_c = 2
        self.m_r = 1
        self.dt = 1 / 1000.0
        self.l = 3
        self.t_0 = 0
        self.t_f = 10

    def f_x_s(self, s):

        cond1 = lambda x: x
        cond2 = lambda x: 5 * cos(x - 10 - 0.5 * math.pi) + 10
        cond3 = lambda x: x - 2 * math.pi

        return where(less(s, 10), cond1(s), where(less(s, 10 + 2*math.pi), cond2(s), cond3(s)))

    def f_x_prime_s(self, s):

        cond1 = lambda x: 1
        cond2 = lambda x: -5 * sin(x - 10 - 0.5 * math.pi)
        cond3 = lambda x: 1

        return where(less(s, 10), cond1(s), where(less(s, 10 + 2*math.pi), cond2(s), cond3(s)))

    def f_x_double_prime_s(self, s):

        cond1 = lambda x: 0
        cond2 = lambda x: -5 * cos(x - 10 - 0.5 * math.pi)
        cond3 = lambda x: 0

        return where(less(s, 10), cond1(s), where(less(s, 10 + 2*math.pi), cond2(s), cond3(s)))

    def f_y_s(self, s):

        cond1 = lambda x: 0.2 * (x - 10)**2
        cond2 = lambda x: 5 * sin(x - 10 - 0.5 * math.pi) + 5
        cond3 = lambda x: 0

        return where(less(s, 10), cond1(s), where(less(s, 10 + 2*math.pi), cond2(s), cond3(s)))

    def f_y_prime_s(self, s):

        cond1 = lambda x: 0.4 * (x - 10)
        cond2 = lambda x: 5 * cos(x - 10 - 0.5 * math.pi)
        cond3 = lambda x: 0

        return where(less(s, 10), cond1(s), where(less(s, 10 + 2*math.pi), cond2(s), cond3(s)))

    def f_y_double_prime_s(self, s):

        cond1 = lambda x: 0.4
        cond2 = lambda x: -5 * sin(x - 10 - 0.5 * math.pi)
        cond3 = lambda x: 0

        return where(less(s, 10), cond1(s), where(less(s, 10 + 2*math.pi), cond2(s), cond3(s)))

    def solve_ode(self):

        def derivs(t, W):
            s, s_dot, theta, theta_dot = W
            f_x_p = self.f_x_prime_s(s)
            f_x_pp = self.f_x_double_prime_s(s)
            f_y_p = self.f_y_prime_s(s)
            f_y_pp = self.f_y_double_prime_s(s)
            dydx = f_y_p / f_x_p

            # equations in the form [s_double_dot, theta_double_dot, T, N]
            eqn1 = [f_x_p, 0, -math.cos(theta) / self.m_c, dydx / (self.m_c * math.sqrt(1 + dydx ** 2))]
            rhs1 = [-s_dot**2 * f_x_pp]
            eqn2 = [f_y_p, 0, -math.sin(theta) / self.m_c, -1 / (self.m_c * math.sqrt(1 + dydx ** 2))]
            rhs2 = [-s_dot**2 * f_y_pp - ODESolver.g]
            eqn3 = [f_x_p, -self.l*math.sin(theta), math.cos(theta)/self.m_r, 0]
            rhs3 = [-s_dot**2 * f_x_pp + self.l*theta_dot**2 * math.cos(theta)]
            eqn4 = [f_y_p, self.l*math.cos(theta), math.sin(theta)/self.m_r, 0]
            rhs4 = [-s_dot**2 * f_y_pp + self.l*theta_dot**2 * math.sin(theta) - ODESolver.g]

            M = array([eqn1, eqn2, eqn3, eqn4])
            RHS = array([rhs1, rhs2, rhs3, rhs4])

            solved = dot(linalg.inv(M), RHS)
            s_double_dot = solved[0, 0]
            theta_double_dot = solved[1, 0]

            return [s_dot, s_double_dot, theta_dot, theta_double_dot]

        s_0 = 10 + math.pi + 0.1
        s_dot_0 = 0
        theta_0 = -math.pi / 2
        theta_dot_0 = 0
        initials = [s_0, s_dot_0, theta_0, theta_dot_0]

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

    def get_x_vals(self, s_vals):
        return self.f_x_s(s_vals)


    def get_y_vals(self, s_vals):
        return self.f_y_s(s_vals)
