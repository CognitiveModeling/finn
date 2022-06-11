"""
Finite Volume Neural Network implementation with PyTorch.
"""

import torch.nn as nn
import torch as th
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.distributions import Normal

import os
import init
import set_module
import pandas as pd
import sys
sys.path.append(os.path.abspath("../models"))
from utils.configuration import Configuration

import time


class Bayes_Layer(nn.Module):
    """
        This is the building block layer for bayesian NODE's
    """

    def __init__(self, input_features, output_features, prior_var=1.):

        super().__init__()

        # set input and output dimensions
        self.input_features = input_features
        self.output_features = output_features

        # initialize mu and rho parameters for the weights of the layer
        self.w_mu = nn.Parameter(Normal(0., 0.1).expand(
            (output_features, input_features)).sample())
        self.w_rho = nn.Parameter(Normal(-3., 0.1).expand(
            (output_features, input_features)).sample())

        # initialize mu and rho parameters for the layer's bias
        self.b_mu = nn.Parameter(Normal(0., 0.1).expand(
            (output_features,)).sample())
        self.b_rho = nn.Parameter(Normal(-3., 0.1).expand(
            (output_features,)).sample())

        # initialize weight samples
        # (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.b = None

        # initialize prior distribution for all of the weights and biases
        self.prior = th.distributions.Normal(0, prior_var)
        
    def sample_weights(self):
        """This functionality is implemented here in order to assure
        that the weights are sampled before any time forward propagation
        """
        # sample weights
        self.w = Normal(self.w_mu, th.log(1+th.exp(self.w_rho))).rsample()
        
        # sample bias
        self.b = Normal(self.b_mu, th.log(1+th.exp(self.b_rho))).rsample()
        
        # log prior
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = th.sum(w_log_prior) + th.sum(b_log_prior)

        # log posterior
        self.w_post = Normal(
            self.w_mu.data, th.log(1+th.exp(self.w_rho)))
        self.b_post = Normal(
            self.b_mu.data, th.log(1+th.exp(self.b_rho)))
        self.log_post = self.w_post.log_prob(
            self.w).sum() + self.b_post.log_prob(self.b).sum()

    def forward(self, input):
        """
          Standard linear forward propagation
        """
        return F.linear(input, self.w, self.b)


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
        
        layers = []

        for layer_idx in range(len(self.layer_sizes) - 1):
            layer = Bayes_Layer(
                input_features=self.layer_sizes[layer_idx],
                output_features=self.layer_sizes[layer_idx + 1]
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
    def __init__(self, u, D, BC, dx, layer_sizes, device, data, noise_tol, mode="train",
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
        
        self.variational_layers = [x for x in self.func_nn.children()
                                   if type(x) == Bayes_Layer]
        
        self.data = data
        
        self.noise_tol = noise_tol
        
        self.cfg = config
        
        self.inp_func = th.linspace(0,2,501).unsqueeze(-1).to(device=self.device)
        
    
    def sample_weights(self):
        """This functionality is implemented here in order to assure
        that the weights are sampled before any time forward propagation
        """
        for layer in self.variational_layers:
            layer.sample_weights()
        
        
    def log_prior(self):
        """Calculates the log prior for
         all the variational layers
        Returns:
            torch.Tensor: log-prior for all the layers as a scalar
        """
        log_prior = th.stack([x.log_prior for x in self.variational_layers])
        return th.mean(log_prior)

    def log_post(self):
        """Calculates the log posterior for all the
         variational layers.
        Returns:
            torch.Tensor: log-posterior for all the layers as a scalar
        """
        log_post = th.stack([x.log_post for x in self.variational_layers])
        return th.mean(log_post)
    
        
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
        
        # # Calculate the flux at the right domain boundary     
        # right_bound_flux_c = (self.D[0]*ret[-1]*(self.stencil[0]*c[-1] +
        #                     self.stencil[1]*right_BC)).unsqueeze(0)
        
        # Calculate the fluxes between control volumes i and their right neighbors 
        right_neighbors_c = self.D[0]*ret[:-1]*(self.stencil[0]*c[:-1] +
                            self.stencil[1]*c[1:])
                        
        # Concatenate the right fluxes
        right_flux_c = th.cat((right_neighbors_c, right_bound_flux_c))
        
        ## For ct
        # # Calculate the flux at the right domain boundary 
        # right_bound_flux_ct = (self.D[1]*(self.stencil[0]*c[-1] +
        #                     self.stencil[1]*right_BC)).unsqueeze(0)
        
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
        self.sample_weights()
        pred = odeint(self.state_kernel, u[0], t)
        
        return pred
    
    def sample_elbo(self, t, u, samples=5, kl_weight=0.01):
        """Method for calculating the negative expected lower bound
        Args:
            y0 (torch.Tensor): initial condition of the differential
                equation.
            t (torch.Tensor): timesteps where we want to perform the
                solution with the ode solver.
            y (torch.Tensor): true trajectory(target) of the differential
                equation.
            samples (int): for how many samples to perform the monte carlo
                approximation of the ELBO
        Returns:
            torch.Tensor: Scalar of the ELBO loss
        """
        outputs = th.empty((samples,) + (u.shape[0],)).to(device=self.device)
        log_priors = th.empty(samples)
        log_posts = th.empty(samples)
        log_likes = th.empty(samples).to(device=self.device)

        for i in range(samples):
            # Calculate predicted breakthrough curve and likelihood
            pred = self.forward(t=t, u=u).to(device=self.cfg.device)
            cauchy_mult = self.cfg.cauchy_mult * self.cfg.D_eff[0] * self.cfg.dx
            outputs[i] = ((pred[:,-2,0] - pred[:,-1,0]) * cauchy_mult).squeeze()
            log_likes[i] = Normal(1e3*self.data, self.noise_tol).log_prob(
                1e3*outputs[i]).mean()
            
            # Calculate prior and physical prior
            ret = (1/self.func_nn(self.inp_func)).squeeze()
            log_prior_phys = Normal(th.zeros(len(ret)-1).to(device=self.device), 1e-5).log_prob(
                th.relu(ret[1:]-ret[:-1])).mean()
            log_priors[i] = self.log_prior() + log_prior_phys
            
            log_posts[i] = self.log_post()
            

        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()

        elbo_loss = kl_weight * (log_post - log_prior) - log_like
        
        return elbo_loss



# Train the model

# Load the user configurations
config = Configuration("config.json")

# Initialization of configurations and set modules for different core samples
params = init.Initialize(config)
core1_cfg = set_module.Set_Module('data_core1.xlsx', params)
core2_cfg = set_module.Set_Module('data_core2.xlsx', params)
core2b_cfg = set_module.Set_Module('data_core2_long.xlsx', params)

# Read the core #2 data from the Excel file
data_core2 = pd.read_excel('data_core2.xlsx', index_col=None, header=None)
breakthrough_data_core2 = (th.FloatTensor(data_core2[1]) / 1000).to(config.general.device)
breakthrough_time_core2 = th.FloatTensor(data_core2[0]).to(config.general.device)
x_core2 = th.linspace(0,core2_cfg.X,core2_cfg.Nx, dtype=th.float).to(config.general.device)

# Initialize the model to be trained using data from core sample #2
u = th.zeros(core2_cfg.Nt,core2_cfg.Nx,2).to(config.general.device)
model = FINN_DiffSorp(
        u = u,
        D = core2_cfg.D_eff,
        BC = np.array([[core2_cfg.solubility, core2_cfg.solubility], [0.0, 0.0]]),
        dx = core2_cfg.dx,
        layer_sizes = params.config.model.layer_sizes,
        device = params.device,
        data=breakthrough_data_core2,
        noise_tol=config.sampling.noise_tol,
        mode="train",
        config=core2_cfg,
        learn_coeff=False,
        cauchy_mult=core2_cfg.cauchy_mult,
        use_exp=False
    ).to(config.general.device)

optimizer = th.optim.Adam(model.parameters(),lr=0.001)

def closure():
    optimizer.zero_grad()
    loss = model.sample_elbo(breakthrough_time_core2, u)
    
    loss.backward()
    
    print(loss.item())
    return loss

best_loss = np.inf

save_file = 'var_inf_ensemble.pt'

for itr in range(2000):
    loss = optimizer.step(closure)
    print(loss.item())
    print(itr)
    if loss.item() < best_loss:
        th.save(model.state_dict(), save_file)
        best_loss = loss.item()
    
model.load_state_dict(th.load(save_file))

out = []
for i in range(config.sampling.num_sample):
    pred = model(breakthrough_time_core2, u)
    cauchy_mult = core2_cfg.cauchy_mult * core2_cfg.D_eff[0] * core2_cfg.dx
    out.append(((pred[:,-2,0] - pred[:,-1,0]) * cauchy_mult).squeeze())
    print(i)

out = th.stack(out)
np.save('core2.npy', np.array(out.detach().cpu()))

# Plot the trained model with core sample #2
core2_physmodel = pd.read_excel('data_core2.xlsx', sheet_name=2, index_col=None, header=None)
core2_physmodel = th.FloatTensor(core2_physmodel[1])/1000

fig, ax = plt.subplots(1, 4, figsize=(12,3))

breakthrough_pred = th.stack(out).detach().cpu().numpy()
mean_pred = np.mean(breakthrough_pred,axis=0)
lower_pred = np.quantile(breakthrough_pred,0.05,axis=0)
upper_pred = np.quantile(breakthrough_pred,0.95,axis=0)


h = ax[0].scatter(breakthrough_time_core2.cpu(), 1000*breakthrough_data_core2.cpu(),
                  color="red", label="Data",s=10)
ax[0].plot(breakthrough_time_core2.cpu(), 1000*mean_pred,linewidth=2,label="FINN")
ax[0].fill_between(breakthrough_time_core2.cpu(), 1000*lower_pred, 1000*upper_pred, alpha=0.3)
ax[0].plot(breakthrough_time_core2.cpu(), 1000*core2_physmodel, linestyle="--", linewidth=2, label="Physical Model")
ax[0].set_title("Core #2", fontsize=20)
ax[0].set_xlabel("$t$", fontsize=20)
ax[0].set_ylabel("$u$", fontsize=20)
ax[0].axes.set_xticks([0,20,40])
ax[0].axes.set_yticks([0,2.5,5])
ax[0].set_xticklabels([0,20,40], fontsize=20)
ax[0].set_yticklabels([0,2.5,5], fontsize=20)


# Plot the trained model with core sample #1
data_core1 = pd.read_excel('data_core1.xlsx', index_col=None, header=None)
breakthrough_data_core1 = (th.FloatTensor(data_core1[1]) / 1000).to(config.general.device)
breakthrough_time_core1 = th.FloatTensor(data_core1[0]).to(config.general.device)
x_core1 = th.linspace(0,core1_cfg.X,core1_cfg.Nx, dtype=th.float).to(config.general.device)

# Adjust the model's known parameter
u = th.zeros(core1_cfg.Nt,core1_cfg.Nx,2).to(config.general.device)
model.D = core1_cfg.D_eff
model.BC = np.array([[core1_cfg.solubility, core1_cfg.solubility], [0.0, 0.0]])
model.dx = core1_cfg.dx
model.cauchy_mult = core1_cfg.cauchy_mult

# Calculate prediction for core #1 using samples
out = []
for i in range(config.sampling.num_sample):
    pred = model(breakthrough_time_core1, u)
    cauchy_mult = core1_cfg.cauchy_mult * core1_cfg.D_eff[0] * core1_cfg.dx
    out.append(((pred[:,-2,0] - pred[:,-1,0]) * cauchy_mult).squeeze())
    print(i)

out = th.stack(out)
np.save('core1.npy', np.array(out.detach().cpu()))

core1_physmodel = pd.read_excel('data_core1.xlsx', sheet_name=2, index_col=None, header=None)
core1_physmodel = th.FloatTensor(core1_physmodel[1])/1000

breakthrough_pred = th.stack(out).detach().cpu().numpy()
mean_pred = np.mean(breakthrough_pred,axis=0)
lower_pred = np.quantile(breakthrough_pred,0.05,axis=0)
upper_pred = np.quantile(breakthrough_pred,0.95,axis=0)


h = ax[1].scatter(breakthrough_time_core1.cpu(), 1000*breakthrough_data_core1.cpu(),
                  color="red",s=10)
ax[1].plot(breakthrough_time_core1.cpu(), 1000*mean_pred,linewidth=2)
ax[1].fill_between(breakthrough_time_core1.cpu(), 1000*lower_pred, 1000*upper_pred, alpha=0.3)
ax[1].plot(breakthrough_time_core1.cpu(), 1000*core1_physmodel, linestyle="--", linewidth=2)
ax[1].set_title("Core #1", fontsize=20)
ax[1].set_xlabel("$t$", fontsize=20)
ax[1].set_ylabel("$u$", fontsize=20)
ax[1].axes.set_xticks([0,20,40])
ax[1].axes.set_yticks([0,2.5,5])
ax[1].set_xticklabels([0,20,40], fontsize=20)
ax[1].set_yticklabels([0,2.5,5], fontsize=20)

# Test the trained model with core sample #2B
data_core2b = pd.read_excel('data_core2_long.xlsx', index_col=None, header=None)
profile_data_core2b = th.FloatTensor(data_core2b[1]) / 1000
profile_x_core2b = th.FloatTensor(data_core2b[0])
time_core2b = th.linspace(0.0, core2b_cfg.T, 101)
x_core2b = th.linspace(0,core2b_cfg.X,core2b_cfg.Nx, dtype=th.float).to(config.general.device)

# Adjust the model's known parameter
u = th.zeros(core2b_cfg.Nt,core2b_cfg.Nx,2).to(config.general.device)
model.D = core2b_cfg.D_eff
model.BC = np.array([[core2b_cfg.solubility, core2b_cfg.solubility], [0.0, 0.0]])
model.dx = core2b_cfg.dx
model.cauchy = False

# Calculate prediction for core #2b using samples
out = []
for i in range(config.sampling.num_sample):
    pred = model(time_core2b, u)
    out.append(pred[-1,...,1])
    print(i)

out = th.stack(out)
np.save('core2b.npy', np.array(out.detach().cpu()))

data_core2b_physmodel = pd.read_excel('data_core2_long.xlsx', sheet_name=2, index_col=None, header=None)
core2b_physmodel = th.FloatTensor(data_core2b_physmodel[1])/1000
x_core2b_physmodel = th.FloatTensor(data_core2b_physmodel[0])

profile_pred = out.detach().cpu().numpy()
mean_pred = np.mean(profile_pred,axis=0)
lower_pred = np.quantile(profile_pred,0.05,axis=0)
upper_pred = np.quantile(profile_pred,0.95,axis=0)


h = ax[2].scatter(profile_x_core2b.cpu(), 1000*profile_data_core2b.cpu(),
                  color="red",s=10)
ax[2].plot(x_core2b.cpu(), 1000*mean_pred,linewidth=2)
ax[2].fill_between(x_core2b.cpu(), 1000*lower_pred, 1000*upper_pred, alpha=0.3)
ax[2].plot(x_core2b_physmodel.cpu(), 1000*core2b_physmodel, linestyle="--", linewidth=2)
ax[2].set_title("Core #2B", fontsize=20)
ax[2].set_xlabel("$x$", fontsize=20)
ax[2].set_ylabel("$u_t$", fontsize=20)
ax[2].axes.set_xticks([0,0.05,0.1])
ax[2].axes.set_yticks([0,300,600])
ax[2].set_xticklabels([0,0.05,0.1], fontsize=20)
ax[2].set_yticklabels([0,300,600], fontsize=20)

## Retardation

u = th.linspace(0,2,501).unsqueeze(-1)

ret = []

for i in range(config.sampling.num_sample):
    model.sample_weights()
    ret.append(1/model.func_nn(u.to(device=config.general.device)).squeeze())

ret = th.stack(ret).detach().cpu().numpy()
np.save('ret.npy', ret)
u = np.array(u.squeeze().detach())

ret_phys = np.loadtxt("retardation_phys.txt")
ret_phys = ret_phys[::4]

ret_pred = ret
mean_pred = np.mean(ret_pred,axis=0)
lower_pred = np.quantile(ret_pred,0.05,axis=0)
upper_pred = np.quantile(ret_pred,0.95,axis=0)

h = ax[3].plot(1000*u, mean_pred, linewidth=2)
ax[3].fill_between(1000*u, lower_pred, upper_pred, alpha=0.3)
ax[3].plot(1000*u, ret_phys, linestyle="--", linewidth=2)
ax[3].set_title("Retardation factor", fontsize=20)
ax[3].set_xlabel("$u$", fontsize=20)
ax[3].set_ylabel("$R(u)$", fontsize=20)
ax[3].axes.set_xticks([0,1000,2000])
ax[3].axes.set_yticks([3.0,4.0,5.0])
ax[3].set_xticklabels([0,1000,2000], fontsize=20)
ax[3].set_yticklabels([3.0,4.0,5.0], fontsize=20)

fig.legend(loc=8, ncol=3, fontsize=20)
plt.tight_layout(rect=[0,0.15,1,1])
fig.set_rasterized(True)

fig_name = 'var_inf_ensemble.pdf'
plt.savefig(fig_name)
fig_name = 'var_inf_ensemble.png'
plt.savefig(fig_name)