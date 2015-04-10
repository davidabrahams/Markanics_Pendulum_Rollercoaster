import math
from numpy import array
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
        self.dt = 1 / 24.0
        self.l = 5
        self.t_0 = 0
        self.t_f = 20

    def f_x(self, x):
        return 5 * cos(x) + 0.25 * (x - 10) ** 2


    def f_prime_x(self, x):
        return -5 * sin(x) + .5 * (x - 10)


    def f_double_prime(self, x):
        return -5 * cos(x) + .5


    def solve_ode(self):
        def derivs(t, W):
            x, x_dot, theta, theta_dot = W
            f_p = self.f_prime_x(x)
            f_pp = self.f_double_prime(x)

            # equations in the form [x_double_dot, theta_double_dot, T, N]
            eqn1 = [1, 0, -math.cos(theta) / self.m_c, f_p / (self.m_c * math.sqrt(1 + f_p ** 2))]
            rhs1 = [0]
            eqn2 = [f_p, 0, -math.sin(theta) / self.m_c, -1 / (self.m_c * math.sqrt(1 + f_p ** 2))]
            rhs2 = [-f_pp*x_dot**2 - ODESolver.g]
            eqn3 = [1, -self.l*math.sin(theta), math.cos(theta)/self.m_r, 0]
            rhs3 = [self.l*theta_dot**2 *math.cos(theta)]
            eqn4 = [f_p, self.l*math.cos(theta), math.sin(theta)/self.m_r, 0]
            rhs4 = [-f_pp*x_dot**2 + self.l * theta_dot**2 * math.sin(theta) - ODESolver.g]

            M = array([eqn1, eqn2, eqn3, eqn4])
            RHS = array([rhs1, rhs2, rhs3, rhs4])

            solved = dot(linalg.inv(M), RHS)
            x_double_dot = solved[0, 0]
            theta_double_dot = solved[1, 0]

            return [x_dot, x_double_dot, theta_dot, theta_double_dot]

        x_0 = 0
        x_dot_0 = 0
        theta_0 = -math.pi / 2
        theta_dot_0 = 0
        initials = [x_0, x_dot_0, theta_0, theta_dot_0]

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
