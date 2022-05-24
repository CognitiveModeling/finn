"""
This script provides a class solving the Burger's equation via numerical
integration using scipy's solve_ivp method. It can be used to generate data
samples of the Burger's equation with Dirichlet boundary condition on both
sides (u = 0).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags


class Simulator:

    def __init__(self, diffusion_coefficient, t_max, t_steps, x_left, x_right,
                 x_steps, y_bottom, y_top, y_steps, train_data):
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
        self.Y0 = y_bottom
        self.Y1 = y_top
        
        self.Nx = x_steps
        self.Ny = y_steps
        self.Nt = t_steps
        
        self.dx = (self.X1 - self.X0)/(self.Nx - 1)
        self.dy = (self.Y1 - self.Y0)/(self.Ny - 1)
        
        self.x = np.linspace(self.X0 + self.dx, self.X1 - self.dx, self.Nx)
        self.y = np.linspace(self.Y0 + self.dy, self.Y1 - self.dy, self.Ny)
        self.t = np.linspace(0, self.T, self.Nt)
        
        self.train_data = train_data

    def generate_sample(self):
        """
        Single sample generation using the parameters of this simulator.
        :return: The generated sample as numpy array(t, x)
        """

        # Initialize the simulation field
        X, Y = np.meshgrid(self.x, self.y)
        if self.train_data:
            u0 = -np.sin(np.pi*(X+Y))
        else:
            u0 = -np.sin(np.pi*(X-Y))
        u0 = u0.reshape(self.Nx*self.Ny)

        # Generate arrays as diagonal inputs to the Laplacian matrix
        main_diag = -2*np.ones(self.Nx)/self.dx**2 -2*np.ones(self.Nx)/self.dy**2
        main_diag = np.tile(main_diag, self.Ny)
        
        left_diag = np.ones(self.Nx)
        left_diag[0] = 0
        left_diag = np.tile(left_diag, self.Ny)
        left_diag = left_diag[1:]/self.dx**2
        
        right_diag = np.ones(self.Nx)
        right_diag[-1] = 0
        right_diag = np.tile(right_diag, self.Ny)
        right_diag = right_diag[:-1]/self.dx**2
        
        bottom_diag = np.ones(self.Nx*(self.Ny-1))/self.dy**2
        
        top_diag = np.ones(self.Nx*(self.Ny-1))/self.dy**2
        
        # Generate the sparse Laplacian matrix
        diagonals = [main_diag, left_diag, right_diag, bottom_diag, top_diag]
        offsets = [0, -1, 1, -self.Nx, self.Nx]
        self.lap = diags(diagonals, offsets)

        # Solve Burger's equation
        prob = solve_ivp(self.rc_ode, (0, self.T), u0, t_eval=self.t)
        ode_data = prob.y

        self.sample = np.transpose(ode_data).reshape(-1,self.Ny,self.Nx)

        return self.sample

    def rc_ode(self, t, u):
        """
        Solves a given equation for a particular time step.
        :param t: The current time step
        :param u: The equation values to solve
        :return: A finite difference solution
        """
        
        u = u.reshape(self.Ny,self.Nx)
    
        a_plus = np.maximum(u,0)
        a_min = np.minimum(u,0)
        
        u_left = np.concatenate((np.zeros((u.shape[0],1)),u[:,:-1]),axis=1)
        u_right = np.concatenate((u[:,1:],np.zeros((u.shape[0],1))),axis=1)
        
        u_bottom = np.concatenate((np.zeros((1,u.shape[1])),u[:-1]),axis=0)
        u_top = np.concatenate((u[1:],np.zeros((1,u.shape[1]))),axis=0)
        
        adv = a_plus*(u-u_left)/self.dx + a_min*(u_right-u)/self.dx + \
            a_plus*(u-u_bottom)/self.dy + a_min*(u_top-u)/self.dy
        u = u.reshape(self.Nx*self.Ny)
        
        return self.D*(self.lap@u) - adv.reshape(self.Nx*self.Ny)
