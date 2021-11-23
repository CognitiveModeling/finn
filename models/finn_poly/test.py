#! env/bin/python3

"""
Main file for testing (evaluating) a FINN model
"""

import numpy as np
import torch as th
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys

sys.path.append("..")
from utils.configuration import Configuration
from finn import FINN_Burger, FINN_DiffSorp, FINN_DiffReact, FINN_AllenCahn

_author_ = "Matthias Karlbauer, Timothy Praditia"


def run_testing(visualize=False, model_number=None):

    # Load the user configurations
    config = Configuration("config.json")
    
    # Append the model number to the name of the model
    if model_number is None:
        model_number = config.model.number
    config.model.name = config.model.name + "_" + str(model_number).zfill(2)

    # Print some information to console
    print("Model name:", config.model.name)

    # Hide the GPU(s) in case the user specified to use the CPU in the config
    # file
    if config.general.device == "CPU":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    root_path = os.path.abspath("../../data")
    data_path = os.path.join(root_path, config.data.type, config.data.name)
    
    # Set device on GPU if specified in the configuration file, else CPU
    device = th.device(config.general.device)
    
    
    if config.data.type == "burger":
        # Load samples, together with x, y, and t series
        t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                      dtype=th.float).to(device=device)
        x = np.load(os.path.join(data_path, "x_series.npy"))
        u = th.tensor(np.load(os.path.join(data_path, "sample.npy")),
                             dtype=th.float).to(device=device)
        
        u[1:] = u[1:] + th.normal(th.zeros_like(u[1:]),th.ones_like(u[1:])*config.data.noise)
        
        dx = x[1]-x[0]
    
        # Initialize and set up the model
        model = FINN_Burger(
            u = u,
            D = np.array([1.0]),
            BC = np.array([[0.0], [0.0]]),
            dx = dx,
            device = device,
            mode="test",
            learn_coeff=True
        ).to(device=device)
    
    
    elif config.data.type == "diffusion_sorption":
        # Load samples, together with x, y, and t series
        t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                      dtype=th.float).to(device=device)
        x = np.load(os.path.join(data_path, "x_series.npy"))
        sample_c = th.tensor(np.load(os.path.join(data_path, "sample_c.npy")),
                             dtype=th.float).to(device=device)
        sample_ct = th.tensor(np.load(os.path.join(data_path, "sample_ct.npy")),
                             dtype=th.float).to(device=device)
        
        dx = x[1]-x[0]
        u = th.stack((sample_c, sample_ct), dim=len(sample_c.shape))
        u[1:] = u[1:] + th.normal(th.zeros_like(u[1:]),th.ones_like(u[1:])*config.data.noise)
        
        # Initialize and set up the model
        if "test" in config.data.name:
            bc = np.array([[0.7, 0.7], [0.0, 0.0]])
        else:
            bc = np.array([[1.0, 1.0], [0.0, 0.0]])
            
        model = FINN_DiffSorp(
            u = u,
            D = np.array([0.5, 0.1]),
            BC = bc,
            dx = dx,
            device = device,
            mode="test",
            learn_coeff=True
        ).to(device=device)
    
    elif config.data.type == "diffusion_reaction":
        # Load samples, together with x, y, and t series
        t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                      dtype=th.float).to(device=device)
        x = np.load(os.path.join(data_path, "x_series.npy"))
        y = np.load(os.path.join(data_path, "y_series.npy"))
        sample_u = th.tensor(np.load(os.path.join(data_path, "sample_u.npy")),
                             dtype=th.float).to(device=device)
        sample_v = th.tensor(np.load(os.path.join(data_path, "sample_v.npy")),
                             dtype=th.float).to(device=device)
        
        dx = x[1]-x[0]
        dy = y[1]-y[0]
        
        u = th.stack((sample_u, sample_v), dim=len(sample_u.shape))
        u[1:] = u[1:] + th.normal(th.zeros_like(u[1:]),th.ones_like(u[1:])*config.data.noise)
    
        # Initialize and set up the model
        model = FINN_DiffReact(
            u = u,
            D = np.array([5E-4/(dx**2), 1E-3/(dx**2)]),
            BC = np.zeros((4,2)),
            dx = dx,
            dy = dy,
            device = device,
            mode="test",
            learn_coeff=True
        ).to(device=device)
        
    elif config.data.type == "allen_cahn":
        # Load samples, together with x, y, and t series
        t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                      dtype=th.float).to(device=device)
        x = np.load(os.path.join(data_path, "x_series.npy"))
        u = th.tensor(np.load(os.path.join(data_path, "sample.npy")),
                             dtype=th.float).to(device=device)
        
        u[1:] = u[1:] + th.normal(th.zeros_like(u[1:]),th.ones_like(u[1:])*config.data.noise)
        
        dx = x[1]-x[0]
    
        # Initialize and set up the model
        model = FINN_AllenCahn(
            u = u,
            D = np.array([0.6]),
            BC = np.array([[0.0], [0.0]]),
            dx = dx,
            device = device,
            mode="test",
            learn_coeff=True
        ).to(device=device)
    

    # Count number of trainable parameters
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Trainable model parameters: {pytorch_total_params}\n")

    # Load the trained weights from the checkpoints into the model
    model.load_state_dict(th.load(os.path.join(os.path.abspath(""),
                                               "checkpoints",
                                               config.model.name,
                                               config.model.name+".pt")))
    
    model.eval()

    # Initialize the criterion (loss)
    criterion = nn.MSELoss()

    #
    # Forward data through the model
    u_hat = model(t=t, u=u).detach().cpu()
    u = u.cpu()
    t = t.cpu()
    
    pred = np.array(u_hat)
    labels = np.array(u)

    # Compute error
    mse = criterion(u_hat, u).item()
    print(f"MSE: {mse}")
    
    #
    # Visualize the data
    if config.data.type == "burger" and visualize:
        
        fig, ax = plt.subplots(2, 1, figsize=(3,5))
        h = ax[0].imshow(np.transpose(u_hat), interpolation='nearest',
                      extent=[0, 2,
                              -1, 1],
                      origin='lower', aspect='auto')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(h, cax=cax)
        h.set_clim(u.min(), u.max())
        ax[0].set_title("Burgers'")
        ax[0].set_ylabel("$x$")
        ax[0].set_xlabel("$t$")
        ax[0].axes.set_xticks([0, 0.5, 1.0])
        ax[0].axes.set_yticks([-1, 0, 1])
        
        h = ax[1].plot(x, u[-1], "ro-", markersize=2, linewidth = 0.5, label="Data")
        h = ax[1].plot(x, u_hat[-1],label="Prediction")
        ax[1].set_title("t = 1")
        ax[1].set_ylabel("$u$")
        ax[1].set_xlabel("$x$")
        ax[1].axes.set_xticks([-1, 0, 1])
        ax[1].axes.set_yticks([-0.5, 0, 0.5])
        
        fig.legend(loc=8, ncol=2)
        plt.tight_layout(rect=[0,0.05,1,1])
        plt.savefig('polyfinn_burger_train.pdf')
    
    elif config.data.type == "diffusion_sorption" and visualize:
        
        fig, ax = plt.subplots(2, 1, figsize=(3,5))
        h = ax[0].imshow(np.transpose(u_hat[...,0]), interpolation='nearest',
                    extent=[0, 10000,
                            0, 1],
                      origin='lower', aspect='auto')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(h, cax=cax, ticks=[0, 0.3, 0.6])
        h.set_clim(u.min(), u.max())
        ax[0].set_title("Diffusion sorption")
        ax[0].set_ylabel("$x$")
        ax[0].set_xlabel("$t$")
        ax[0].axes.set_xticks([0, 10000])
        ax[0].axes.set_yticks([0.0, 0.5, 1.0])
        ax[0].set_xticklabels([0, 10000])
        ax[0].set_yticklabels([0.0, 0.5, 1.0])
        
        h = ax[1].plot(x, u[-1,...,0], "ro-", markersize=2, linewidth = 0.5, label="Data")
        h = ax[1].plot(x, u_hat[-1,...,0],label="Prediction")
        ax[1].set_title("t = 2500")
        ax[1].set_ylabel("$u$")
        ax[1].set_xlabel("$x$")
        ax[1].axes.set_xticks([0, 0.5, 1])
        ax[1].axes.set_yticks([0, 0.5, 1])
        ax[1].set_xticklabels([0, 0.5, 1])
        ax[1].set_yticklabels([0, 0.5, 1])
        
        fig.legend(loc=8, ncol=2)
        plt.tight_layout(rect=[0,0.05,1,1])
        plt.savefig('polyfinn_diff_sorp_train.pdf')
        
    elif config.data.type == "diffusion_reaction" and visualize:
        
        fig, ax = plt.subplots(2, 1, figsize=(3,5))
        h = ax[0].imshow(np.transpose(u_hat[-1,...,0]), interpolation='nearest',
                    extent=[-1, 1,
                            -1, 1],
                      origin='lower', aspect='auto')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(h, cax=cax, ticks=[-0.5,0,0.5])
        h.set_clim(u[-1].min(), u[-1].max())
        ax[0].set_title("Diffusion reaction")
        ax[0].set_ylabel("$y$")
        ax[0].set_xlabel("$x$")
        ax[0].axes.set_xticks([-1, 0, 1])
        ax[0].axes.set_yticks([-1, 0, 1])
        ax[0].set_xticklabels([-1, 0, 1])
        ax[0].set_yticklabels([-1, 0, 1])
        
        h = ax[1].plot(x, u[-1,...,49//2,0], "ro-", markersize=2, linewidth = 0.5, label="Data")
        h = ax[1].plot(x, u_hat[-1,...,49//2,0],label="Prediction")
        ax[1].set_title("t = 10, y = 0")
        ax[1].set_ylabel("$u_1$")
        ax[1].set_xlabel("$x$")
        ax[1].axes.set_xticks([-1,0,1])
        ax[1].axes.set_yticks([-0.5,0,0.5])
        ax[1].set_xticklabels([-1, 0, 1])
        ax[1].set_yticklabels([-0.5, 0, 0.5])
        
        fig.legend(loc=8, ncol=2)
        plt.tight_layout(rect=[0,0.05,1,1])
        plt.savefig('polyfinn_diff_react_train.pdf')
        
    elif config.data.type == "allen_cahn" and visualize:
        fig, ax = plt.subplots(2, 1, figsize=(3,5))
        h = ax[0].imshow(np.transpose(u_hat), interpolation='nearest',
                      extent=[0, 0.5,
                              -1, 1],
                      origin='lower', aspect='auto')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(h, cax=cax)
        h.set_clim(u.min(), u.max())
        ax[0].set_title("Allen-Cahn")
        ax[0].set_ylabel("$x$")
        ax[0].set_xlabel("$t$")
        ax[0].axes.set_xticks([0, 0.25, 0.5])
        ax[0].axes.set_yticks([-1, 0, 1])
        
        h = ax[1].plot(x, u[-1], "ro-", markersize=2, linewidth = 0.5, label="Data")
        h = ax[1].plot(x, u_hat[-1],label="Prediction")
        ax[1].set_title("t = 0.5")
        ax[1].set_ylabel("$u$")
        ax[1].set_xlabel("$x$")
        ax[1].axes.set_xticks([-1, 0, 1])
        ax[1].axes.set_yticks([-1.0, -0.5, 0, 0.5])
        
        fig.legend(loc=8, ncol=2)
        plt.tight_layout(rect=[0,0.05,1,1])
        plt.savefig('polyfinn_allen_cahn_train.pdf')
    
    return pred, labels


def animate_1d(t, axis1, axis2, field, field_hat):
    """
    Data animation function animating an image over time.
    :param t: The current time step
    :param axis: The matplotlib image object
    :param field: The data field
    :return: The matplotlib image object updated with the current time step's
        image date
    """
    axis1.set_ydata(field[:, t])
    axis2.set_ydata(field_hat[:, t])


def animate_2d(t, im1, im2, u_hat, u):
    """
    Data animation function animating an image over time.
    :param t: The current time step
    :param axis: The matplotlib image object
    :param field: The data field
    :return: The matplotlib image object updated with the current time step's
        image date
    """
    im1.set_array(u_hat[t,:,:].squeeze().t().detach())
    im2.set_array(u[t,:,:].squeeze().t().detach())



if __name__ == "__main__":
    th.set_num_threads(1)
    
    pred, u = run_testing(visualize=True)

    print("Done.")