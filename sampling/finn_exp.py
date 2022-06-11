"""
Finite Volume Neural Network implementation with PyTorch.
"""

import torch.nn as nn
import torch as th
from torchdiffeq import odeint
#from torchdiffeq import odeint_adjoint as odeint


import time


class FINN(nn.Module):
    """
    This is the parent FINN class. This class initializes all sharable parameters
    between different implementations to be inherited to each of the implementation.
    It also contains the initialization of the function_learner and reaction_learner
    NN which learns the constitutive relationships (or the flux multiplier) and
    reaction functions, respectively.
    """
    
    def __init__(self, u, D, BC, layer_sizes, device, mode,
                 config, learn_coeff, learn_stencil, bias, sigmoid):
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
        self.layer_sizes = layer_sizes
        self.mode = mode
        self.cfg = config
        self.bias = bias
        self.sigmoid = sigmoid
        
        if not learn_coeff:
            self.D = th.tensor(D, dtype=th.float, device=self.device)
        else:
            self.D = nn.Parameter(th.tensor(D, dtype=th.float,
                                            device=self.device))
        
        if not learn_stencil:
            self.stencil = th.tensor([-1.0, 1.0], dtype=th.float,
                                     device=self.device)
        else:
            self.stencil = th.tensor(
                [th.normal(th.tensor([-1.0]), th.tensor([0.1])),
                 th.normal(th.tensor([1.0]), th.tensor([0.1]))],
                dtype=th.float, device=self.device)
            self.stencil = nn.Parameter(self.stencil)
    
    def function_learner(self):
        """
        This function constructs a feedforward NN required for calculation
        of constitutive function (or flux multiplier) as a function of u.
        """
        layers = list()
        
        for layer_idx in range(len(self.layer_sizes) - 1):
            layer = nn.Linear(
                in_features=self.layer_sizes[layer_idx],
                out_features=self.layer_sizes[layer_idx + 1],
                bias=self.bias
                ).to(device=self.device)
            layers.append(layer)
        
            if layer_idx < len(self.layer_sizes) - 2 or not self.sigmoid:
                layers.append(nn.Tanh())
            else:
                # Use sigmoid function to keep the values strictly positive
                # (all outputs have the same sign)
                layers.append(nn.Sigmoid())
        
        return nn.Sequential(*nn.ModuleList(layers))
    

class Wrapper(nn.Module):
    def __init__(self, f):

        super().__init__()
        self.f = f

    def forward(self, t, u):
        return self.f(t, u)
    
    
class FINN_DiffSorp(FINN):
    """
    This is the inherited FINN class for the diffusion-sorption equation implementation.
    This class inherits all parameter from the parent FINN class.
    """
    def __init__(self, u, D, BC, dx, layer_sizes, device, mode="train",
                 config=None, learn_coeff=False, learn_stencil=False, bias=True,
                 sigmoid=True, cauchy_mult=1.0, use_exp=True, cauchy=True):
        
        super().__init__(u, D, BC, layer_sizes, device, mode, config, learn_coeff,
                         learn_stencil, bias, sigmoid)
        
        """
        Constructor.
        
        Inputs:
        Same with the parent FINN class, with the addition of dx (the spatial resolution)
        """
        
        self.dx = dx
        
        # Initialize the function_learner to learn the retardation factor function
        self.func_nn = self.function_learner().to(device=self.device)
        # Initialize the multiplier of the retardation factor function (denormalization)
        if use_exp:
            self.p_exp = nn.Parameter(th.tensor([0.0],dtype=th.float))
        # Initialize the multiplier for the Cauchy boundary condition
        self.cauchy_mult = cauchy_mult
        self.use_exp = use_exp
        
        self.cauchy = cauchy
        
    
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
        if self.use_exp:
            ret = (self.func_nn(c.unsqueeze(-1)) * 10**self.p_exp)[...,0]
        else:
            ret = (self.func_nn(c.unsqueeze(-1)))[...,0]
        
        ## Calculate fluxes at the left boundary of control volumes i
        
        ## For c
        # Calculate the flux at the left domain boundary
        left_bound_flux_c = (self.D[0]*ret[0]*(self.stencil[0]*c[0] +
                            self.stencil[1]*self.BC[0,0])).unsqueeze(0)
        
        
        # Calculate the fluxes between control volumes i and their left neighbors
        left_neighbors_c = self.D[0]*ret[1:]*(self.stencil[0]*c[1:] +
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
        
        if self.cauchy:
            # Calculate the Cauchy condition for the right domain boundary
            right_BC = self.D[0]*self.dx*(c[-2]-c[-1])*self.cauchy_mult
            
            # Calculate the flux at the right domain boundary     
            right_bound_flux_c = (self.D[0]*ret[-1]*(self.stencil[0]*c[-1] +
                            self.stencil[1]*right_BC)).unsqueeze(0)
            
            # Calculate the flux at the right domain boundary 
            right_bound_flux_ct = (self.D[1]*(self.stencil[0]*c[-1] +
                            self.stencil[1]*right_BC)).unsqueeze(0)
        else:
            right_bound_flux_c = th.zeros(1).to(device=self.device)
            right_bound_flux_ct = th.zeros(1).to(device=self.device)
        
        # No flux at the right domain boundary     
        
        # Calculate the fluxes between control volumes i and their right neighbors 
        right_neighbors_c = self.D[0]*ret[:-1]*(self.stencil[0]*c[:-1] +
                            self.stencil[1]*c[1:])
                        
        # Concatenate the right fluxes
        right_flux_c = th.cat((right_neighbors_c, right_bound_flux_c))
        
        ## For ct
        # No flux at the right domain boundary
        
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
