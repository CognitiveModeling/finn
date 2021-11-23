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
import time

sys.path.append("..")
from utils.configuration import Configuration
from finn import FINN_Burger, FINN_DiffSorp, FINN_DiffReact, FINN_AllenCahn


def run_testing(print_progress=False, visualize=False, model_number=None):

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
            layer_sizes = config.model.layer_sizes,
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
            layer_sizes = config.model.layer_sizes,
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
            layer_sizes = config.model.layer_sizes,
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
            D = np.array([0.3]),
            BC = np.array([[0.0], [0.0]]),
            dx = dx,
            layer_sizes = config.model.layer_sizes,
            device = device,
            mode="train",
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
    time_start = time.time()
    with th.no_grad():
        u_hat = model(t=t, u=u)
    if print_progress:
        print(f"Forward pass took: {time.time() - time_start} seconds.")
    u_hat = u_hat.detach().cpu()
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
        u_hat = np.transpose(u_hat)
        u = np.transpose(u)
    
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    
        # u(t, x) over space
        h = ax[0].imshow(u, interpolation='nearest', 
                      extent=[t.min(), t.max(),
                              x.min(), x.max()],
                      origin='upper', aspect='auto')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
    
        ax[0].set_xlim(0, t.max())
        ax[0].set_ylim(x.min(), x.max())
        ax[0].legend(loc="upper right")
        ax[0].set_xlabel('$t$')
        ax[0].set_ylabel('$x$')
        ax[0].set_title('$u(t,x)$', fontsize = 10)
        
        h = ax[1].imshow(u_hat, interpolation='nearest', 
                      extent=[t.min(), t.max(),
                              x.min(), x.max()],
                      origin='upper', aspect='auto')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
    
        ax[1].set_xlim(0, t.max())
        ax[1].set_ylim(x.min(), x.max())
        ax[1].legend(loc="upper right")
        ax[1].set_xlabel('$t$')
        ax[1].set_ylabel('$x$')
        ax[1].set_title('$u(t,x)$', fontsize = 10)
        
        # u(t, x) over time
        fig, ax = plt.subplots()
        line1, = ax.plot(x, u[:, 0], 'b-', linewidth=2, label='Exact')
        line2, = ax.plot(x, u_hat[:, 0], 'ro', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')    
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([-1.1, 1.1])
    
        anim = animation.FuncAnimation(fig,
                                       animate_1d,
                                       frames=len(t),
                                       fargs=(line1, line2, u, u_hat),
                                       interval=20)
        plt.tight_layout()
        plt.draw()
        plt.show()
    
    
    elif config.data.type == "diffusion_sorption" and visualize:
        u_hat = np.transpose(u_hat[...,0])
        u = np.transpose(u[...,0])
    
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    
        # u(t, x) over space
        h = ax[0].imshow(u, interpolation='nearest', 
                      extent=[t.min(), t.max(),
                              x.min(), x.max()],
                      origin='upper', aspect='auto')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
    
        ax[0].set_xlim(0, t.max())
        ax[0].set_ylim(x.min(), x.max())
        ax[0].legend(loc="upper right")
        ax[0].set_xlabel('$t$')
        ax[0].set_ylabel('$x$')
        ax[0].set_title('$u(t,x)$', fontsize = 10)
        
        h = ax[1].imshow(u_hat, interpolation='nearest', 
                      extent=[t.min(), t.max(),
                              x.min(), x.max()],
                      origin='upper', aspect='auto')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
    
        ax[1].set_xlim(0, t.max())
        ax[1].set_ylim(x.min(), x.max())
        ax[1].legend(loc="upper right")
        ax[1].set_xlabel('$t$')
        ax[1].set_ylabel('$x$')
        ax[1].set_title('$u(t,x)$', fontsize = 10)
        
        # u(t, x) over time
        fig, ax = plt.subplots()
        line1, = ax.plot(x, u[:, 0], 'b-', linewidth=2, label='Exact')
        line2, = ax.plot(x, u_hat[:, 0], 'ro', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')    
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([-1.1, 1.1])
    
        anim = animation.FuncAnimation(fig,
                                       animate_1d,
                                       frames=len(t),
                                       fargs=(line1, line2, u, u_hat),
                                       interval=20)
        plt.tight_layout()
        plt.draw()
        plt.show()
        
        plt.figure()
        plt.plot(x,u_hat[:,-1])
        plt.scatter(x,u[:,-1])
        
    elif config.data.type == "diffusion_reaction" and visualize:
    
        # Plot u over space
        fig, ax = plt.subplots(2, 1, figsize=(6, 10))
    
        im1 = ax[0].imshow(u_hat[-1,:,:,0].squeeze().t().detach(), interpolation='nearest', 
                     extent=[x.min(), x.max(),
                             y.min(), y.max()],
                     origin='lower', aspect='auto')
        fig.colorbar(im1, ax=ax[0])
        im1.set_clim(u[-1,:,:,0].min(), u[-1,:,:,0].max())
        im2 = ax[1].imshow(u[-1,:,:,0].squeeze().t().detach(), interpolation='nearest', 
                     extent=[x.min(), x.max(),
                             y.min(), y.max()],
                     origin='lower', aspect='auto')
        fig.colorbar(im2, ax=ax[1])
        im2.set_clim(u[-1,:,:,0].min(), u[-1,:,:,0].max())
        ax[0].set_xlabel("$x$")
        ax[0].set_ylabel("$y$")
        ax[0].set_title('$u(x,y) predicted$', fontsize = 10)
        ax[1].set_xlabel("$x$")
        ax[1].set_ylabel("$y$")
        ax[1].set_title('$u(x,y) data$', fontsize = 10)
        plt.show()
        
        # Animate through time
        anim = animation.FuncAnimation(fig,
                                        animate_2d,
                                        frames=len(t),
                                        fargs=(im1, im2, u_hat[...,0], u[...,0]),
                                        interval=20)
        
        plt.tight_layout()
        plt.show()
        
        # Plot v over space
        fig, ax = plt.subplots(2, 1, figsize=(6, 10))
    
        im1 = ax[0].imshow(u_hat[-1,:,:,1].squeeze().t().detach(), interpolation='nearest', 
                     extent=[x.min(), x.max(),
                             y.min(), y.max()],
                     origin='lower', aspect='auto')
        fig.colorbar(im1, ax=ax[0])
        im1.set_clim(u[-1,:,:,1].min(), u[-1,:,:,1].max())
        im2 = ax[1].imshow(u[-1,:,:,1].squeeze().t().detach(), interpolation='nearest', 
                     extent=[x.min(), x.max(),
                             y.min(), y.max()],
                     origin='lower', aspect='auto')
        fig.colorbar(im2, ax=ax[1])
        im2.set_clim(u[-1,:,:,1].min(), u[-1,:,:,1].max())
        ax[0].set_xlabel("$x$")
        ax[0].set_ylabel("$y$")
        ax[0].set_title('$v(x,y) predicted$', fontsize = 10)
        ax[1].set_xlabel("$x$")
        ax[1].set_ylabel("$y$")
        ax[1].set_title('$v(x,y) data$', fontsize = 10)
        
        # Animate through time
        anim = animation.FuncAnimation(fig,
                                        animate_2d,
                                        frames=len(t),
                                        fargs=(im1, im2, u_hat[...,1], u[...,1]),
                                        interval=20)
        
        plt.tight_layout()
        plt.draw()
        plt.show()
        
    elif config.data.type == "allen_cahn" and visualize:
        u_hat = np.transpose(u_hat)
        u = np.transpose(u)
    
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    
        # u(t, x) over space
        h = ax[0].imshow(u, interpolation='nearest', 
                      extent=[t.min(), t.max(),
                              x.min(), x.max()],
                      origin='upper', aspect='auto')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
    
        ax[0].set_xlim(0, t.max())
        ax[0].set_ylim(x.min(), x.max())
        ax[0].legend(loc="upper right")
        ax[0].set_xlabel('$t$')
        ax[0].set_ylabel('$x$')
        ax[0].set_title('$u(t,x)$', fontsize = 10)
        
        h = ax[1].imshow(u_hat, interpolation='nearest', 
                      extent=[t.min(), t.max(),
                              x.min(), x.max()],
                      origin='upper', aspect='auto')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
    
        ax[1].set_xlim(0, t.max())
        ax[1].set_ylim(x.min(), x.max())
        ax[1].legend(loc="upper right")
        ax[1].set_xlabel('$t$')
        ax[1].set_ylabel('$x$')
        ax[1].set_title('$u(t,x)$', fontsize = 10)
        
        # u(t, x) over time
        fig, ax = plt.subplots()
        line1, = ax.plot(x, u[:, -1], 'b-', linewidth=2, label='Exact')
        line2, = ax.plot(x, u_hat[:, -1], 'ro', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')    
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([-1.1, 1.1])
    
        anim = animation.FuncAnimation(fig,
                                        animate_1d,
                                        frames=len(t),
                                        fargs=(line1, line2, u, u_hat),
                                        interval=20)
        plt.tight_layout()
        plt.draw()
        plt.show()
    
    return model


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
    
    model = run_testing(print_progress=True, visualize=True)

    print("Done.")