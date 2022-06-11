# -*- coding: utf-8 -*-
"""
Created on March 2021
@author: Timothy Praditia

This script is the main file to train and test FINN with experimental dataset
"""

import torch
import numpy as np
import os
import init
import set_module
import train
import pandas as pd
import sys
sys.path.append(os.path.abspath("../models"))
from finn_exp import FINN_DiffSorp
from utils.configuration import Configuration
from sampler import Metropolis_Hastings, MALA, Barker
import pickle
import matplotlib.pyplot as plt

# Load the user configurations
config = Configuration("config.json")

# Initialization of configurations and set modules for different core samples
params = init.Initialize(config)
core1_cfg = set_module.Set_Module('data_core1.xlsx', params)
core2_cfg = set_module.Set_Module('data_core2.xlsx', params)
core2b_cfg = set_module.Set_Module('data_core2_long.xlsx', params)

# Initialize the model to be trained using data from core sample #2
u = torch.zeros(core2_cfg.Nt,core2_cfg.Nx,2).to(config.general.device)
model = FINN_DiffSorp(
        u = u,
        D = core2_cfg.D_eff,
        BC = np.array([[core2_cfg.solubility, core2_cfg.solubility], [0.0, 0.0]]),
        dx = core2_cfg.dx,
        layer_sizes = params.config.model.layer_sizes,
        device = params.device,
        mode="train",
        learn_coeff=False,
        cauchy_mult=core2_cfg.cauchy_mult,
        use_exp=False
    ).to(config.general.device)

model_temp = FINN_DiffSorp(
        u = u,
        D = core2_cfg.D_eff,
        BC = np.array([[core2_cfg.solubility, core2_cfg.solubility], [0.0, 0.0]]),
        dx = core2_cfg.dx,
        layer_sizes = params.config.model.layer_sizes,
        device = params.device,
        mode="train",
        learn_coeff=False,
        cauchy_mult=core2_cfg.cauchy_mult,
        use_exp=False
    ).to(config.general.device)

# Read the core #2 data from the Excel file
data_core2 = pd.read_excel('data_core2.xlsx', index_col=None, header=None)
breakthrough_data_core2 = (torch.FloatTensor(data_core2[1]) / 1000).to(config.general.device)
breakthrough_time_core2 = torch.FloatTensor(data_core2[0]).to(config.general.device)
x_core2 = torch.linspace(0,core2_cfg.X,core2_cfg.Nx, dtype=torch.float).to(config.general.device)

if not config.sampling.random_init:
    if config.sampling.train:
        u0 = torch.zeros(core2_cfg.Nt,core2_cfg.Nx,2).to(config.general.device)
        train.run_training(x_core2, breakthrough_time_core2, u0, breakthrough_data_core2, model, core2_cfg)

    model.load_state_dict(torch.load(os.path.join(os.path.abspath(""),
                                      "checkpoints",
                                      config.model.name,
                                      config.model.name + ".pt"),
                                     map_location=config.general.device))
    burn_in = 1
else:
    burn_in = 201


if config.sampling.sampler == "mh":
    sampler = Metropolis_Hastings(model, model_temp, config.sampling.step_size,
                                     config.sampling.noise_tol, core2_cfg,
                                     breakthrough_data_core2)
    
elif config.sampling.sampler == "mala":
    sampler = MALA(model, model_temp, config.sampling.step_size,
                                     config.sampling.noise_tol, core2_cfg,
                                     breakthrough_data_core2)
     
elif config.sampling.sampler == "barker":
    sampler = Barker(model, model_temp, config.sampling.step_size,
                                     config.sampling.noise_tol, core2_cfg,
                                     breakthrough_data_core2)

if config.sampling.sample:
    sampler.sample(config.sampling.num_sample, u, breakthrough_time_core2)

# Load the saved samples
save_path = os.path.abspath("") + "/" + config.sampling.name + ".pickle"
pickle_file = open(save_path, "rb")
sampler = pickle.load(pickle_file)
pickle_file.close()

# Plot the trained model with core sample #2
core2_physmodel = pd.read_excel('data_core2.xlsx', sheet_name=2, index_col=None, header=None)
core2_physmodel = torch.FloatTensor(core2_physmodel[1])/1000

fig, ax = plt.subplots(1, 4, figsize=(12,3))

breakthrough_pred = torch.stack(sampler.output_sample[burn_in:]).detach().cpu().numpy()
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
breakthrough_data_core1 = (torch.FloatTensor(data_core1[1]) / 1000).to(config.general.device)
breakthrough_time_core1 = torch.FloatTensor(data_core1[0]).to(config.general.device)
x_core1 = torch.linspace(0,core1_cfg.X,core1_cfg.Nx, dtype=torch.float).to(config.general.device)

# Adjust the model's known parameter
u = torch.zeros(core1_cfg.Nt,core1_cfg.Nx,2).to(config.general.device)
model.D = core1_cfg.D_eff
model.BC = np.array([[core1_cfg.solubility, core1_cfg.solubility], [0.0, 0.0]])
model.dx = core1_cfg.dx
model.cauchy_mult = core1_cfg.cauchy_mult

# Calculate prediction for core #1 using samples
pred = []
for i in range(burn_in, len(sampler.param_sample)):
    torch.nn.utils.vector_to_parameters(sampler.param_sample[i], model.parameters())
    output = model(t=breakthrough_time_core1, u=u).to(device=config.general.device)
        
    # Calculate predicted breakthrough curve
    cauchy_mult = core1_cfg.cauchy_mult * core1_cfg.D_eff[0] * core1_cfg.dx
    breakthrough_pred = ((output[:,-2,0] - output[:,-1,0]) * cauchy_mult).squeeze()
    
    pred.append(breakthrough_pred)
    print(i)
    
sampler.core1_sample = pred

save_path = os.path.abspath("") + "/" + config.sampling.name + ".pickle"
pickle_file = open(save_path, "wb")
pickle.dump(sampler, pickle_file)
pickle_file.close()

core1_physmodel = pd.read_excel('data_core1.xlsx', sheet_name=2, index_col=None, header=None)
core1_physmodel = torch.FloatTensor(core1_physmodel[1])/1000

breakthrough_pred = torch.stack(pred).detach().cpu().numpy()
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
profile_data_core2b = torch.FloatTensor(data_core2b[1]) / 1000
profile_x_core2b = torch.FloatTensor(data_core2b[0])
time_core2b = torch.linspace(0.0, core2b_cfg.T, 101)
x_core2b = torch.linspace(0,core2b_cfg.X,core2b_cfg.Nx, dtype=torch.float).to(config.general.device)

# Adjust the model's known parameter
u = torch.zeros(core2b_cfg.Nt,core2b_cfg.Nx,2).to(config.general.device)
model.D = core2b_cfg.D_eff
model.BC = np.array([[core2b_cfg.solubility, core2b_cfg.solubility], [0.0, 0.0]])
model.dx = core2b_cfg.dx
model.cauchy = False

# Calculate prediction for core #2b using samples
pred = []
for i in range(burn_in, len(sampler.param_sample)):
    torch.nn.utils.vector_to_parameters(sampler.param_sample[i], model.parameters())
    output = model(t=time_core2b, u=u).to(device=config.general.device)
        
    pred.append(output[-1,...,1])
    print(i)

sampler.core2b_sample = pred

save_path = os.path.abspath("") + "/" + config.sampling.name + ".pickle"
pickle_file = open(save_path, "wb")
pickle.dump(sampler, pickle_file)
pickle_file.close()

data_core2b_physmodel = pd.read_excel('data_core2_long.xlsx', sheet_name=2, index_col=None, header=None)
core2b_physmodel = torch.FloatTensor(data_core2b_physmodel[1])/1000
x_core2b_physmodel = torch.FloatTensor(data_core2b_physmodel[0])

profile_pred = torch.stack(pred).detach().cpu().numpy()
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

u = np.linspace(0,2,501)
ret = torch.stack(sampler.func_sample[burn_in:]).detach().cpu().numpy()
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

fig_name = config.sampling.name + '.pdf'
plt.savefig(fig_name)

# Calculate effective sample size

import tensorflow_probability as tfp

params = torch.stack(sampler.param_sample[burn_in:])
ess_params = tfp.mcmc.effective_sample_size(params.detach().cpu()).numpy()
print(f'ESS Params = {ess_params.mean()}')

pred = torch.stack(sampler.output_sample[burn_in:])
ess_pred = tfp.mcmc.effective_sample_size(pred.detach().cpu()).numpy()
print(f'ESS Predictions = {ess_pred.mean()}')

