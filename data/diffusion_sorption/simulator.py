"""
This script provides a class solving a diffusion sorption problem via numerical
integration using scipy's solve_ivp method. It can be used to generate data
samples of the diffusion sorption problem with different boundary conditions
(left Dirichlet, that is constant, right Cauchy).
"""

import numpy as np
from scipy.integrate import solve_ivp


class Simulator:

    def __init__(self, ret_factor_fun, diffusion_coefficient, porosity, rho_s,
                 k_f, n_f, s_max, kl, kd, solubility, t_max, t_steps, x_left,
                 x_right, x_steps, train_data):
        """
        Constructor method initializing the parameters for the diffusion
        sorption problem.
        :param ret_factor_fun: The retardation factor function as string
        :param diffusion_coefficient: The diffusion coefficient [m^2/day]
        :param porosity: The porosity of the medium [-]
        :param rho_s: Dry bulk density [kg/m^3]
        :param k_f: Freundlich K
        :param n_f: Freundlich exponent
        :param s_max: Sorption capacity [m^3/kg]
        :param kl: Half-concentration [kg/m^3]
        :param kd: Partitioning coefficient [m^3/kg]
        :param solubility: The solubility of the quantity
        :param t_max: Stop time of the simulation
        :param t_steps: Number of simulation steps
        :param x_left: Left end of the 1D simulation field
        :param x_right: Right end of the 1D simulation field
        :param x_steps: Number of spatial steps between x_left and x_right
        """

        # Set class parameters
        self.D = diffusion_coefficient

        self.por = porosity
        self.rho_s = rho_s
        self.k_f = k_f
        self.n_f = n_f
        self.s_max = s_max
        self.kl = kl
        self.kd = kd
        self.solubility = solubility

        self.T = t_max
        self.X0 = x_left
        self.X1 = x_right
        
        self.Nx = x_steps
        self.Nt = t_steps
        
        self.dx = (self.X1 - self.X0)/(self.Nx - 1)
        
        self.x = np.linspace(0, self.X1, self.Nx)
        self.t = np.linspace(0, self.T, self.Nt)
        
        self.train_data = train_data

        # Specify the retardation function according to the user input
        if ret_factor_fun == "linear":
            self.retardation = self.retardation_linear
        elif ret_factor_fun == "langmuir":
            self.retardation = self.retardation_langmuir
        elif ret_factor_fun == "freundlich":
            self.retardation = self.retardation_freundlich

    def retardation_linear(self, u):
        """
        Linear retardation factor function.
        :param u: The simulation field
        """
        return 1 + ((1 - self.por)/self.por)*self.rho_s\
                   *self.kd

    def retardation_freundlich(self, u):
        """
        Langmuir retardation factor function.
        :param u: The simulation field
        """
        return 1 + ((1 - self.por)/self.por)*self.rho_s\
                   *self.k_f*self.n_f*(u + 1e-6)**(self.n_f-1)

    def retardation_langmuir(self, u):
        """
        Freundlich retardation factor function.
        :param u: The simulation field
        """
        return 1 + ((1 - self.por)/self.por)*self.rho_s\
                   *((self.s_max*self.kl)/(u + self.kl)**2)

    def generate_sample(self):
        """
        Single sample generation using the parameters of this simulator.
        :return: The generated sample as numpy array(t, x)
        """

        # Initialize the simulation field
        u0 = np.zeros(self.Nx)
        u0 = np.concatenate((u0, u0))

        #
        # Laplacian matrix
        nx = np.diag(-2*np.ones(self.Nx), k=0)
        nx_minus_1 = np.diag(np.ones(self.Nx-1), k=-1)
        nx_plus_1  = np.diag(np.ones(self.Nx-1), k=1)
        
        self.lap = nx + nx_minus_1 + nx_plus_1
        self.lap /= self.dx**2

        self.q = np.zeros(self.Nx)
        self.q_tot = np.zeros(self.Nx)

        # Solve the diffusion sorption problem
        prob = solve_ivp(self.rc_ode, (0, self.T), u0, t_eval=self.t, method="BDF")
        ode_data = prob.y

        sample_c = np.transpose(ode_data[:self.Nx])
        sample_c_tot = np.transpose(ode_data[self.Nx:])

        return sample_c, sample_c_tot

    def rc_ode(self, t, u):
        """
        Solves a given equation for a particular time step.
        :param t: The current time step
        :param u: The equation values to solve
        :return: A finite difference solution
        """
        
        # Separate u into c and c_tot
        c = u[:self.Nx]
        c_tot = u[self.Nx:]
       
        # Calculate left and right BC
        left_BC = self.solubility
        right_BC = (c[-2]-c[-1])/self.dx * self.D
       
        # Calculate time derivative of each unknown
        self.q[0]  = self.D/self.retardation(c[0])/(self.dx**2)*left_BC
        self.q[-1] = self.D/self.retardation(c[-1])/(self.dx**2)*right_BC
        c_t        = self.D/self.retardation(c)*np.matmul(self.lap, c) + self.q
       
        self.q_tot[0]  = self.D*self.por/(self.rho_s/1000)\
                         /(self.dx**2)*left_BC
        self.q_tot[-1] = self.D*self.por/(self.rho_s/1000)\
                         /(self.dx**2)*right_BC
        c_tot_t        = self.D*self.por/(self.rho_s/1000)\
                         *np.matmul(self.lap, c) + self.q_tot
       
        # Stack the time derivative into a single array u_t
        u_t = np.concatenate((c_t, c_tot_t))
       
        return u_t
