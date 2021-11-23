"""
This script provides a class solving the Burger's equation via numerical
integration using scipy's solve_ivp method. It can be used to generate data
samples of the Burger's equation with Dirichlet boundary condition on both
sides (u = 0).
"""

import numpy as np
from scipy.integrate import solve_ivp


class Simulator:

    def __init__(self, diffusion_coefficient, t_max, t_steps, x_left, x_right,
                 x_steps, train_data):
        """
        Constructor method initializing the parameters for the Burger's
        equation.
        :param diffusion_coefficient: The diffusion coefficient
        :param t_max: Stop time of the simulation
        :param t_steps: Number of simulation steps
        :param x_left: Left end of the 1D simulation field
        :param x_right: Right end of the 1D simulation field
        :param x_steps: Number of spatial steps between x_left and x_right
        """

        # Set class parameters
        self.D = diffusion_coefficient

        self.T = t_max
        self.X0 = x_left
        self.X1 = x_right
        
        self.Nx = x_steps
        self.Nt = t_steps
        
        self.dx = (self.X1 - self.X0)/(self.Nx - 1)
        
        self.x = np.linspace(self.X0 + self.dx, self.X1 - self.dx, self.Nx - 2)
        self.t = np.linspace(0, self.T, self.Nt)
        
        self.train_data = train_data

    def generate_sample(self):
        """
        Single sample generation using the parameters of this simulator.
        :return: The generated sample as numpy array(t, x)
        """

        # Initialize the simulation field
        if self.train_data:
            u0 = -np.sin(np.pi*self.x)
        else:
            u0 = np.sin(np.pi*self.x)

        nx_minus_2 = np.diag(-2*np.ones(self.Nx-2), k=0)
        nx_minus_3 = np.diag(np.ones(self.Nx-3), k=-1)
        nx_plus_3 = np.diag(np.ones(self.Nx-3), k=1)
        
        self.lap = nx_minus_2 + nx_minus_3 + nx_plus_3
        self.lap /= self.dx**2
        # Periodic BC
        # lap[0,-1] = 1/dx**2
        # lap[-1,0] = 1/dx**2

        # Solve Burger's equation
        prob = solve_ivp(self.rc_ode, (0, self.T), u0, t_eval=self.t)
        ode_data = prob.y

        self.sample = np.transpose(ode_data)

        return self.sample

    def rc_ode(self, t, u):
        """
        Solves a given equation for a particular time step.
        :param t: The current time step
        :param u: The equation values to solve
        :return: A finite difference solution
        """
        a_plus = np.maximum(u,0)
        a_min = np.minimum(u,0)
        
        u_left = np.concatenate((np.array([0]),u[:-1]))
        u_right = np.concatenate((u[1:],np.array([0])))

        # Return finite difference
        return self.D*np.matmul(self.lap, u) - (a_plus*(u - u_left)/self.dx + 
                                                a_min*(u_right - u)/self.dx)
