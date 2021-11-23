"""
Finite Volume Neural Network implementation with PyTorch.
"""

import torch.nn as nn
import torch as th
from torchdiffeq import odeint
import numpy as np
from itertools import product

__author__ = "Timothy Praditia"

class FINN(nn.Module):
    """
    This is the parent FINN class. This class initializes all sharable parameters
    between different implementations to be inherited to each of the implementation.
    It also contains the initialization of the function_learner and reaction_learner
    NN which learns the constitutive relationships (or the flux multiplier) and
    reaction functions, respectively.
    """
    
    def __init__(self, u, D, BC, device, mode,
                 config, learn_coeff, learn_stencil):
        """
        Constructor.
        
        Inputs:            
        :param u: the unknown variable
        :type u: th.tensor[len(t), Nx, Ny, num_vars]
        
        :param D: diffusion coefficient
        :type D: np.array[num_vars] --- th.tensor is also accepted
        
        :param BC: the boundary condition values. In case of Dirichlet BC, this
        contains the scalar values. In case of Neumann, this contains the flux
        values.
        :type BC: np.array[num_bound, num_vars] --- th.tensor is also accepted
        
        :param layer_sizes: a list of the hidden nodes for each layer (including
        the input and output features)
        :type layer_sizes: list[num_hidden_layers + 2]
        
        :param device: the device to perform simulation
        :type device: th.device
        
        :param mode: mode of simulation ("train" or "test")
        :type mode: str
        
        :param config: configuration of simulation parameters
        :type config: dict
        
        :param learn_coeff: a switch to set diffusion coefficient to be learnable
        :type learn_coeff: bool
        
        :param learn_stencil: a switch to set the numerical stencil to be learnable
        :type learn_stencil: bool
        
        :param bias: a bool value to determine whether to use bias values in the
        function_learner
        :type bias bool
        
        :param sigmoid: a bool value to determine whether to use sigmoid at the
        output layer
        :type sigmoid: bool
        
        Output:
        :return: the full field solution of u from time t0 until t_end
        :rtype: th.tensor[len(t), Nx, Ny, num_vars]

        """
        
        super(FINN, self).__init__()
        
        self.device = device
        self.Nx = u.size()[1]
        self.BC = th.tensor(BC, dtype=th.float).to(device=self.device)
        self.mode = mode
        self.cfg = config
        
        if not learn_coeff:
            self.D = th.tensor(D, dtype=th.float).to(device=self.device)
        else:
            self.D = nn.Parameter(th.tensor(D, dtype=th.float))
        
        if not learn_stencil:
            self.stencil = th.tensor([-1.0, 1.0], dtype=th.float).to(
                device=self.device)
        else:
            self.stencil = th.tensor(
                [th.normal(th.tensor([-1.0]), th.tensor([0.1])),
                 th.normal(th.tensor([1.0]), th.tensor([0.1]))],
                dtype=th.float)
            self.stencil = nn.Parameter(self.stencil)
            
    
    def function_learner(self):
        """
        This function constructs a feedforward NN required for calculation
        of constitutive function (or flux multiplier) as a function of u.
        """
        
        prod_array = np.arange(0,self.max_degree+1)
        poly_degree = list(product(prod_array, repeat=self.num_var))
        poly_degree = np.stack(poly_degree)
        
        keep_prod = []
        
        for i in range(len(poly_degree)):
            if poly_degree[i].sum() <= self.max_degree:
                keep_prod.append(i)
        
        poly_degree = poly_degree[keep_prod]
        self.poly_degree = th.tensor(poly_degree, dtype=th.float).to(device=self.device)
        
        self.coeffs = nn.Parameter(th.randn(len(self.poly_degree),self.num_var))
        
    
    def func_nn(self, u):
        """
        This function constructs a feedforward NN required for calculation
        of constitutive function (or flux multiplier) as a function of u.
        """
        
        mat_size = list(u.shape[:-1])
        mat_size.append(len(self.poly_degree))
        
        inp_mat = th.ones(mat_size).to(device=self.device)
        for i in range(len(self.poly_degree)):
            for j in range(self.num_var):
                inp_mat[...,i] *= u[...,j] ** self.poly_degree[i,j]
        
        inp_mat = th.tensor(inp_mat, dtype=th.float).to(device=self.device)
        
        return th.matmul(inp_mat, self.coeffs)
    

class FINN_Burger(FINN):
    """
    This is the inherited FINN class for the Burger equation implementation.
    This class inherits all parameter from the parent FINN class.
    """
    
    def __init__(self, u, D, BC, dx, device, mode="train",
                 config=None, learn_coeff=False, learn_stencil=False,
                 max_degree=3):
        """
        Constructor.
        
        Inputs:
        Same with the parent FINN class, with the addition of dx (the spatial resolution)
        """
        
        super().__init__(u, D, BC, device, mode, config, learn_coeff,
                         learn_stencil)
        
        self.dx = dx
        self.num_var = 1
        self.max_degree = max_degree
        
        # Initialize the function_learner to learn the first order flux multiplier
        self.function_learner()
    
    """
    TODO: Implement flux kernel for test (if different BC is used)
    """
        
    def flux_kernel(self, t, u):
        """
        This function defines the flux kernel for training, which takes ui and its
        neighbors as inputs, and returns the integrated flux approximation (up to
        second order derivatives)
        """
        
        # Approximate the first order flux multiplier
        a = self.func_nn(u.unsqueeze(-1))
        
        # Apply the ReLU function for upwind scheme to prevent numerical
        # instability
        a_plus = th.relu(a[...,0])
        a_min = -th.relu(-a[...,0])
        
        
        ## Calculate fluxes at the left boundary of control volumes i
        
        # Calculate the flux at the left domain boundary
        left_bound_flux = (self.D*(self.stencil[0]*u[0] +
                            self.stencil[1]*self.BC[0,0]) -\
                            a_plus[0]/self.dx*(-self.stencil[0]*u[0] -
                            self.stencil[1]*self.BC[0,0]))
                            
        # Calculate the fluxes between control volumes i and their left neighbors
        left_neighbors = self.D*(self.stencil[0]*u[1:] +
                            self.stencil[1]*u[:-1]) -\
                            a_plus[1:]/self.dx*(-self.stencil[0]*u[1:] -
                            self.stencil[1]*u[:-1])
                            
        
        # Concatenate the left fluxes
        left_flux = th.cat((left_bound_flux, left_neighbors))
        
        ## Calculate fluxes at the right boundary of control volumes i
        
        # Calculate the flux at the right domain boundary
        right_bound_flux = (self.D*(self.stencil[0]*u[-1] +
                            self.stencil[1]*self.BC[1,0]) -\
                            a_min[-1]/self.dx*(self.stencil[0]*u[-1] +
                            self.stencil[1]*self.BC[1,0]))
                 
        # Calculate the fluxes between control volumes i and their right neighbors
        right_neighbors = self.D*(self.stencil[0]*u[:-1] +
                            self.stencil[1]*u[1:]) -\
                            a_min[:-1]/self.dx*(self.stencil[0]*u[:-1] +
                            self.stencil[1]*u[1:])
        
        # Concatenate the right fluxes
        right_flux = th.cat((right_neighbors, right_bound_flux))
        
        
        # Integrate the fluxes at all boundaries of control volumes i
        flux = left_flux + right_flux
        
        return flux
    
    def state_kernel(self, t, u):
        """
        This function defines the state kernel for training, which takes the
        fluxes as inputs, and returns du/dt)
        """
        
        flux = self.flux_kernel(t, u)
        
        # Since there is no reaction term to be learned, du/dt = fluxes
        state = flux
        
        return state
    
    def forward(self, t, u):
        """
        This function integrates du/dt through time using the Neural ODE method
        """
        
        # The odeint function receives the function state_kernel that calculates
        # du/dt, the initial condition u[0], and the time at which the values of
        # u will be saved t
        pred = odeint(self.state_kernel, u[0], t)
        
        return pred
    
    
class FINN_DiffSorp(FINN):
    """
    This is the inherited FINN class for the diffusion-sorption equation implementation.
    This class inherits all parameter from the parent FINN class.
    """
    def __init__(self, u, D, BC, dx, device, mode="train",
                 config=None, learn_coeff=False, learn_stencil=False, max_degree=3):
        super().__init__(u, D, BC, device, mode, config, learn_coeff,
                         learn_stencil)
        
        """
        Constructor.
        
        Inputs:
        Same with the parent FINN class, with the addition of dx (the spatial resolution)
        """
        
        self.dx = dx
        
        self.num_var = 1
        self.max_degree = max_degree
        
        # Initialize the function_learner to learn the retardation factor function
        self.function_learner()
        
    
    """
    TODO: Implement flux kernel for test (if different BC is used)
    """
        
    def flux_kernel(self, t, u):
        """
        This function defines the flux kernel for training, which takes ui and its
        neighbors as inputs, and returns the integrated flux approximation (up to
        second order derivatives)
        """
        
        # Separate u into c and ct
        c = u[...,0]
        ct = u[...,1]
        
        # Approximate 1/retardation_factor
        # ret = (self.func_nn(c.unsqueeze(-1)) * 10**self.p_exp)[...,0]
        ret = (self.func_nn(c.unsqueeze(-1)))[...,0]
        ret = th.abs(ret)
        
        ## Calculate fluxes at the left boundary of control volumes i
        
        ## For c
        # Calculate the flux at the left domain boundary
        left_bound_flux_c = (self.D[0]/ret[0]*(self.stencil[0]*c[0] +
                            self.stencil[1]*self.BC[0,0])).unsqueeze(0)
        
        
        # Calculate the fluxes between control volumes i and their left neighbors
        left_neighbors_c = self.D[0]/ret[1:]*(self.stencil[0]*c[1:] +
                            self.stencil[1]*c[:-1])
        
        # Concatenate the left fluxes
        left_flux_c = th.cat((left_bound_flux_c, left_neighbors_c))
        
        ## For ct
        # Calculate the flux at the left domain boundary
        left_bound_flux_ct = (self.D[1]*(self.stencil[0]*c[0] +
                            self.stencil[1]*self.BC[0,1])).unsqueeze(0)
              
        # Calculate the fluxes between control volumes i and their left neighbors              
        left_neighbors_ct = self.D[1]*(self.stencil[0]*c[1:] +
                            self.stencil[1]*c[:-1])
        
        # Concatenate the left fluxes
        left_flux_ct = th.cat((left_bound_flux_ct, left_neighbors_ct))
        
        # Stack the left fluxes of c and ct together
        left_flux = th.stack((left_flux_c, left_flux_ct), dim=len(c.size()))
        
        
        ## Calculate fluxes at the right boundary of control volumes i
        
        ## For c
        # Calculate the Cauchy condition for the right domain boundary
        right_BC = self.D[0]*self.dx*(c[-2]-c[-1])
        
        # Calculate the flux at the right domain boundary     
        right_bound_flux_c = (self.D[0]/ret[-1]*(self.stencil[0]*c[-1] +
                            self.stencil[1]*right_BC)).unsqueeze(0)
        
        # Calculate the fluxes between control volumes i and their right neighbors 
        right_neighbors_c = self.D[0]/ret[:-1]*(self.stencil[0]*c[:-1] +
                            self.stencil[1]*c[1:])
                        
        # Concatenate the right fluxes
        right_flux_c = th.cat((right_neighbors_c, right_bound_flux_c))
        
        ## For ct
        # Calculate the flux at the right domain boundary 
        right_bound_flux_ct = (self.D[1]*(self.stencil[0]*c[-1] +
                            self.stencil[1]*right_BC)).unsqueeze(0)
        
        # Calculate the fluxes between control volumes i and their right neighbors
        right_neighbors_ct = self.D[1]*(self.stencil[0]*c[:-1] +
                            self.stencil[1]*c[1:])
        
        # Concatenate the right fluxes
        right_flux_ct = th.cat((right_neighbors_ct, right_bound_flux_ct))
        
        # Stack the right fluxes of c and ct together
        right_flux = th.stack((right_flux_c, right_flux_ct), dim=len(c.size()))
        
        
        # Integrate the fluxes at all boundaries of control volumes i
        flux = left_flux + right_flux
        
        return flux
    
    def state_kernel(self, t, u):
        """
        This function defines the state kernel for training, which takes the
        fluxes as inputs, and returns du/dt)
        """
        
        flux = self.flux_kernel(t, u)
        
        # Since there is no reaction term to be learned, du/dt = fluxes
        state = flux
        
        return state
    
    def forward(self, t, u):
        """
        This function integrates du/dt through time using the Neural ODE method
        """
        
        # The odeint function receives the function state_kernel that calculates
        # du/dt, the initial condition u[0], and the time at which the values of
        # u will be saved t
        pred = odeint(self.state_kernel, u[0], t)
        
        return pred



class FINN_DiffReact(FINN):
    """
    This is the inherited FINN class for the diffusion-reaction equation implementation.
    This class inherits all parameter from the parent FINN class.
    """
    def __init__(self, u, D, BC, dx, dy, device, mode="train",
                 config=None, learn_coeff=False, learn_stencil=False, max_degree=3):
        super().__init__(u, D, BC, device, mode, config, learn_coeff,
                         learn_stencil)
        
        """
        Constructor.
        
        Inputs:
        Same with the parent FINN class, with the addition of dx and dy (the
        spatial resolution)
        """
        
        self.dx = dx
        self.dy = dy
        
        self.Ny = u.size()[2]
        
        self.num_var = 2
        self.max_degree = max_degree
        
        # Initialize the reaction_learner to learn the reaction term
        self.function_learner()
        
    
    """
    TODO: Implement flux kernel for test (if different BC is used)
    """
        
    def flux_kernel(self, t, u):
        """
        This function defines the flux kernel for training, which takes ui and its
        neighbors as inputs, and returns the integrated flux approximation (up to
        second order derivatives)
        """
        
        # Separate u into act and inh
        act = u[...,0]
        inh = u[...,1]
        
        ## Calculate fluxes at the left boundary of control volumes i
        
        ## For act
        # Calculate the flux at the left domain boundary
        left_bound_flux_act = th.cat(self.Ny * [self.BC[0,0].unsqueeze(0)]).unsqueeze(0)
                            
        # Calculate the fluxes between control volumes i and their left neighbors 
        left_neighbors_act = self.D[0]*(self.stencil[0]*act[1:] +
                            self.stencil[1]*act[:-1])
        
        # Concatenate the left fluxes
        left_flux_act = th.cat((left_bound_flux_act, left_neighbors_act))
        
        ## For inh
        # Calculate the flux at the left domain boundary
        left_bound_flux_inh = th.cat(self.Ny * [self.BC[0,1].unsqueeze(0)]).unsqueeze(0)
                
        # Calculate the fluxes between control volumes i and their left neighbors
        left_neighbors_inh = self.D[1]*(self.stencil[0]*inh[1:] +
                            self.stencil[1]*inh[:-1])
        
        # Concatenate the left fluxes
        left_flux_inh = th.cat((left_bound_flux_inh, left_neighbors_inh))
        
        # Stack the left fluxes of act and inh together
        left_flux = th.stack((left_flux_act, left_flux_inh), dim=len(act.size()))
        
        ## Calculate fluxes at the right boundary of control volumes i
        
        ## For act
        # Calculate the flux at the right domain boundary
        right_bound_flux_act = th.cat(self.Ny * [self.BC[1,0].unsqueeze(0)]).unsqueeze(0)
                            
        # Calculate the fluxes between control volumes i and their right neighbors 
        right_neighbors_act = self.D[0]*(self.stencil[0]*act[:-1] +
                            self.stencil[1]*act[1:])
        
        # Concatenate the right fluxes
        right_flux_act = th.cat((right_neighbors_act, right_bound_flux_act))
        
        ## For inh
        # Calculate the flux at the right domain boundary  
        right_bound_flux_inh = th.cat(self.Ny * [self.BC[1,1].unsqueeze(0)]).unsqueeze(0)
           
        # Calculate the fluxes between control volumes i and their right neighbors  
        right_neighbors_inh = self.D[1]*(self.stencil[0]*inh[:-1] +
                            self.stencil[1]*inh[1:])
        
        # Concatenate the right fluxes
        right_flux_inh = th.cat((right_neighbors_inh, right_bound_flux_inh))
        
        # Stack the right fluxes of act and inh together
        right_flux = th.stack((right_flux_act, right_flux_inh), dim=len(act.size()))
        
        ## Calculate fluxes at the bottom boundary of control volumes i
        
        ## For act
        # Calculate the flux at the bottom domain boundary
        bottom_bound_flux_act = th.cat(self.Nx * [self.BC[2,0].unsqueeze(0)]).unsqueeze(-1)
           
        # Calculate the fluxes between control volumes i and their bottom neighbors                   
        bottom_neighbors_act = self.D[0]*(self.stencil[0]*act[:,1:] +
                            self.stencil[1]*act[:,:-1])
        
        # Concatenate the bottom fluxes
        bottom_flux_act = th.cat((bottom_bound_flux_act, bottom_neighbors_act), dim=1)
        
        ## For inh
        # Calculate the flux at the bottom domain boundary
        bottom_bound_flux_inh = th.cat(self.Nx * [self.BC[2,1].unsqueeze(0)]).unsqueeze(-1)
                            
        # Calculate the fluxes between control volumes i and their bottom neighbors
        bottom_neighbors_inh = self.D[1]*(self.stencil[0]*inh[:,1:] +
                            self.stencil[1]*inh[:,:-1])
        
        # Concatenate the bottom fluxes
        bottom_flux_inh = th.cat((bottom_bound_flux_inh, bottom_neighbors_inh),dim=1)
        
        # Stack the bottom fluxes of act and inh together
        bottom_flux = th.stack((bottom_flux_act, bottom_flux_inh), dim=len(act.size()))
        
        ## Calculate fluxes at the bottom boundary of control volumes i
        
        ## For act
        # Calculate the flux at the top domain boundary
        top_bound_flux_act = th.cat(self.Nx * [self.BC[3,0].unsqueeze(0)]).unsqueeze(-1)
                            
        # Calculate the fluxes between control volumes i and their top neighbors
        top_neighbors_act = self.D[0]*(self.stencil[0]*act[:,:-1] +
                            self.stencil[1]*act[:,1:])
        
        # Concatenate the top fluxes
        top_flux_act = th.cat((top_neighbors_act, top_bound_flux_act), dim=1)
        
        ## For inh
        # Calculate the flux at the top domain boundary
        top_bound_flux_inh = th.cat(self.Nx * [self.BC[3,1].unsqueeze(0)]).unsqueeze(-1)
                  
        # Calculate the fluxes between control volumes i and their top neighbors
        top_neighbors_inh = self.D[1]*(self.stencil[0]*inh[:,:-1] +
                            self.stencil[1]*inh[:,1:])
        
        # Concatenate the top fluxes
        top_flux_inh = th.cat((top_neighbors_inh, top_bound_flux_inh), dim=1)
        
        # Stack the top fluxes of act and inh together
        top_flux = th.stack((top_flux_act, top_flux_inh), dim=len(act.size()))
        
        
        # Integrate the fluxes at all boundaries of control volumes i
        flux = left_flux + right_flux + bottom_flux + top_flux
        
        return flux
    
    def state_kernel(self, t, u):
        """
        This function defines the state kernel for training, which takes the
        fluxes as inputs, and returns du/dt)
        """
        
        flux = self.flux_kernel(t, u)
        
        # Add the reaction term to the fluxes term to obtain du/dt
        state = flux + self.func_nn(u)
        
        return state
    
    def forward(self, t, u):
        """
        This function integrates du/dt through time using the Neural ODE method
        """
        
        # The odeint function receives the function state_kernel that calculates
        # du/dt, the initial condition u[0], and the time at which the values of
        # u will be saved t
        pred = odeint(self.state_kernel, u[0], t)
        
        return pred
    
class FINN_AllenCahn(FINN):
    """
    This is the inherited FINN class for the Burger equation implementation.
    This class inherits all parameter from the parent FINN class.
    """
    
    def __init__(self, u, D, BC, dx, device, mode="train",
                 config=None, learn_coeff=False, learn_stencil=False,
                 max_degree=3):
        """
        Constructor.
        
        Inputs:
        Same with the parent FINN class, with the addition of dx (the spatial resolution)
        """
        
        super().__init__(u, D, BC, device, mode, config, learn_coeff,
                         learn_stencil)
        
        self.dx = dx
        self.num_var = 1
        self.max_degree = max_degree
        
        # Initialize the function_learner to learn the reaction term
        self.function_learner()
    
    """
    TODO: Implement flux kernel for test (if different BC is used)
    """
        
    def flux_kernel(self, t, u):
        """
        This function defines the flux kernel for training, which takes ui and its
        neighbors as inputs, and returns the integrated flux approximation (up to
        second order derivatives)
        """
        
        ## Calculate fluxes at the left boundary of control volumes i
        
        # Calculate the flux at the left domain boundary
        left_bound_flux = self.D*(self.stencil[0]*u[0] +
                            self.stencil[1]*u[-1])
                            
        # Calculate the fluxes between control volumes i and their left neighbors
        left_neighbors = self.D*(self.stencil[0]*u[1:] +
                            self.stencil[1]*u[:-1])
        
        # Concatenate the left fluxes
        left_flux = th.cat((left_bound_flux, left_neighbors))
        
        ## Calculate fluxes at the right boundary of control volumes i
        
        # Calculate the flux at the right domain boundary
        right_bound_flux = self.D*(self.stencil[0]*u[-1] +
                            self.stencil[1]*u[0])
                 
        # Calculate the fluxes between control volumes i and their right neighbors
        right_neighbors = self.D*(self.stencil[0]*u[:-1] +
                            self.stencil[1]*u[1:])
        
        # Concatenate the right fluxes
        right_flux = th.cat((right_neighbors, right_bound_flux))
        
        
        # Integrate the fluxes at all boundaries of control volumes i
        flux = left_flux + right_flux
        
        return flux
    
    def state_kernel(self, t, u):
        """
        This function defines the state kernel for training, which takes the
        fluxes as inputs, and returns du/dt)
        """
        
        flux = self.flux_kernel(t, u)
        
        # Since there is no reaction term to be learned, du/dt = fluxes
        state = flux + self.func_nn(u).squeeze()
        
        return state
    
    def forward(self, t, u):
        """
        This function integrates du/dt through time using the Neural ODE method
        """
        
        # The odeint function receives the function state_kernel that calculates
        # du/dt, the initial condition u[0], and the time at which the values of
        # u will be saved t
        pred = odeint(self.state_kernel, u[0], t)
        
        return pred