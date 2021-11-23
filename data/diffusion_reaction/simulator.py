"""
This script provides a class solving a diffusion reaction problem via numerical
integration using scipy's solve_ivp method. It can be used to generate data
samples of the diffusion reaction problem with Fitzhugh-Nagumo equation.
"""

import numpy as np
from scipy.integrate import solve_ivp


class Simulator:

    def __init__(self, diffusion_coefficient_u, diffusion_coefficient_v, k,
                 t_max, t_steps, x_left, x_right, x_steps, y_bottom, y_top,
                 y_steps, train_data):
        """
        Constructor method initializing the parameters for the diffusion
        sorption problem.
        :param diffusion_coefficient_u: The diffusion coefficient of u
        :param diffusion_coefficient_u: The diffusion coefficient of v
        :param k: The reaction parameter
        :param t_max: Stop time of the simulation
        :param t_steps: Number of simulation steps
        :param x_left: Left end of the 2D simulation field
        :param x_right: Right end of the 2D simulation field
        :param x_steps: Number of spatial steps between x_left and x_right
        :param y_bottom: bottom end of the 2D simulation field
        :param y_top: top end of the 2D simulation field
        :param y_steps: Number of spatial steps between y_bottom and y_top
        """

        # Set class parameters
        self.Du = diffusion_coefficient_u
        self.Dv = diffusion_coefficient_v
        self.k = k

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
        :return: The generated sample as numpy array(t, x, y)
        """

        # Initialize the simulation field
        X, Y = np.meshgrid(self.x, self.y)
        if self.train_data:
            u0 = np.sin(np.pi*(X+1)/2) * np.sin(np.pi*(Y+1)/2)
        else:
            u0 = np.sin(np.pi*(X+1)/2) * np.sin(np.pi*(Y+1)/2) - 0.5
        u0 = u0.reshape(self.Nx*self.Ny)
        u0 = np.concatenate((u0,u0))

        #
        # Laplacian matrix
        main_diag = -2*np.ones(self.Nx)/self.dx**2 -2*np.ones(self.Nx)/self.dy**2
        main_diag[0] = -1/self.dx**2 -2/self.dy**2
        main_diag[-1] = -1/self.dx**2 -2/self.dy**2
        main_diag = np.tile(main_diag, self.Ny)
        main_diag[:self.Nx] = -2/self.dx**2 -1/self.dy**2
        main_diag[self.Nx*(self.Ny-1):] = -2/self.dx**2 -1/self.dy**2
        main_diag[0] = -1/self.dx**2 -1/self.dy**2
        main_diag[self.Nx-1] = -1/self.dx**2 -1/self.dy**2
        main_diag[self.Nx*(self.Ny-1)] = -1/self.dx**2 -1/self.dy**2
        main_diag[-1] = -1/self.dx**2 -1/self.dy**2
        
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
        
        self.lap = np.diag(main_diag, k=0) +\
            np.diag(left_diag, k=-1) +\
            np.diag(right_diag, k=1) +\
            np.diag(bottom_diag, k=-self.Nx) +\
            np.diag(top_diag, k=self.Nx)
        

        # self.q_u = np.zeros(self.Nx*self.Ny)
        # self.q_v = np.zeros(self.Nx*self.Ny)

        # Solve the diffusion sorption problem
        prob = solve_ivp(self.rc_ode, (0, self.T), u0, t_eval=self.t)
        ode_data = prob.y

        sample_u = np.transpose(ode_data[:self.Nx*self.Ny]).reshape(-1,self.Ny,self.Nx)
        sample_v = np.transpose(ode_data[self.Nx*self.Ny:]).reshape(-1,self.Ny,self.Nx)

        return sample_u, sample_v

    def rc_ode(self, t, y):
        """
        Solves a given equation for a particular time step.
        :param t: The current time step
        :param u: The equation values to solve
        :return: A finite difference solution
        """
        
        # Separate y into u and v
        u = y[:self.Nx*self.Ny]
        v = y[self.Nx*self.Ny:]
       
        # Calculate reaction function for each unknown
        react_u = u - u**3 - self.k - v
        react_v = u - v
       
        # Calculate time derivative for each unknown
        u_t = react_u + self.Du*np.matmul(self.lap,u)
        v_t = react_v + self.Dv*np.matmul(self.lap,v)
        
        # Stack the time derivative into a single array u_t
        u_t = np.concatenate((u_t,v_t))
       
        return u_t
