#! env/bin/python3

"""
Main file for testing (evaluating) a model
"""

import numpy as np
import torch as th
import torch.nn as nn
import glob
import os
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys

sys.path.append("..")
from utils.configuration import Configuration
import utils.helper_functions as helpers
from phydnet import ConvLSTM, PhyCell, EncoderRNN


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
    
    if config.model.small:
        input_dim = 32
        hidden_dims = [32]
        n_layers_convcell = 1
    else:
        input_dim = 64
        hidden_dims = [128,128,64]
        n_layers_convcell = 3
    
    if config.data.type == "burger":
        # Load samples, together with x, y, and t series
        t = np.load(os.path.join(data_path, "t_series.npy"))
        x = np.load(os.path.join(data_path, "x_series.npy"))
        u = th.tensor(np.load(os.path.join(data_path, "sample.npy")),
                             dtype=th.float).to(device=device)
        
        u = u + th.normal(th.zeros_like(u),th.ones_like(u)*config.data.noise)
        
        bc = th.tensor([[[0.0, 0.0]]]).to(device)
        
        sample_length = u.shape[0]
        input_tensor = u[:sample_length//2].unsqueeze(0).unsqueeze(-2).to(device=device)
        target_tensor = u[sample_length//2:].unsqueeze(0).unsqueeze(-2).to(device=device)
    
        # Initialize and set up the model
        phycell  =  PhyCell(input_shape=(input_tensor.shape[-1]//4+1),
                            input_dim=input_dim,
                            F_hidden_dims=[7],
                            n_layers=1,
                            kernel_size=7,
                            device=device) 
        
        convcell =  ConvLSTM(input_shape=(input_tensor.shape[-1]//4+1),
                             input_dim=input_dim,
                             hidden_dims=hidden_dims,
                             n_layers=n_layers_convcell,
                             kernel_size=3,
                             device=device)
        
        model  = EncoderRNN(phycell,
                              convcell,
                              input_channels=1,
                              input_dim=(input_tensor.shape[-1],),
                              _1d=True,
                              bc=bc,
                              device=device,
                              sigmoid=False,
                              small=config.model.small)
        
        constraints = th.zeros((7,7)).to(device)
        ind = 0
        for i in range(0,7):
            constraints[ind,i] = 1
            ind +=1
        
    
    elif config.data.type == "diffusion_sorption":
        # Load samples, together with x, y, and t series
        t = np.load(os.path.join(data_path, "t_series.npy"))
        x = np.load(os.path.join(data_path, "x_series.npy"))
        sample_c = th.tensor(np.load(os.path.join(data_path, "sample_c.npy")),
                             dtype=th.float).to(device=device)
        sample_ct = th.tensor(np.load(os.path.join(data_path, "sample_ct.npy")),
                             dtype=th.float).to(device=device)
        
        u = th.stack((sample_c, sample_ct), dim=1)
        
        u = u + th.normal(th.zeros_like(u),th.ones_like(u)*config.data.noise)

        sample_length = u.shape[0]
        input_tensor = u[:sample_length//2].unsqueeze(0).to(device=device)
        target_tensor = u[sample_length//2:].unsqueeze(0).to(device=device)
        
        if "test" in config.data.name:
            bc = th.tensor([[[0.7, 0.0], [0.7, 0.0]]]).to(device)
        else:
            bc = th.tensor([[[1.0, 0.0], [1.0, 0.0]]]).to(device)
        
        # Initialize and set up the model
        phycell  =  PhyCell(input_shape=(input_tensor.shape[-1]//4+1),
                            input_dim=input_dim,
                            F_hidden_dims=[7],
                            n_layers=1,
                            kernel_size=7,
                            device=device) 
        
        convcell =  ConvLSTM(input_shape=(input_tensor.shape[-1]//4+1),
                             input_dim=input_dim,
                             hidden_dims=hidden_dims,
                             n_layers=n_layers_convcell,
                             kernel_size=3,
                             device=device)
        
        model  = EncoderRNN(phycell,
                              convcell,
                              input_channels=2,
                              input_dim=(input_tensor.shape[-1],),
                              _1d=True,
                              bc=bc,
                              device=device,
                              sigmoid=True,
                              small=config.model.small)
        
        constraints = th.zeros((7,7)).to(device)
        ind = 0
        for i in range(0,7):
            constraints[ind,i] = 1
            ind +=1  
    
    elif config.data.type == "diffusion_reaction":
        # Load samples, together with x, y, and t series
        t = np.load(os.path.join(data_path, "t_series.npy"))
        x = np.load(os.path.join(data_path, "x_series.npy"))
        y = np.load(os.path.join(data_path, "y_series.npy"))
        sample_u = th.tensor(np.load(os.path.join(data_path, "sample_u.npy")),
                             dtype=th.float).to(device=device)
        sample_v = th.tensor(np.load(os.path.join(data_path, "sample_v.npy")),
                             dtype=th.float).to(device=device)
        
        
        u = th.stack((sample_u, sample_v), dim=1)
        u = u + th.normal(th.zeros_like(u),th.ones_like(u)*config.data.noise)
        
        sample_length = u.shape[0]
        input_tensor = u[:sample_length//2].unsqueeze(0).to(device=device)
        target_tensor = u[sample_length//2:].unsqueeze(0).to(device=device)
        
        # Initialize and set up the model
        bc = th.tensor([[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]).to(device=device)

        phycell  =  PhyCell(input_shape=(input_tensor.shape[-2]//4+1,
                                         input_tensor.shape[-1]//4+1),
                            input_dim=input_dim,
                            F_hidden_dims=[49],
                            n_layers=1,
                            kernel_size=(7,7),
                            device=device) 
        
        convcell =  ConvLSTM(input_shape=(input_tensor.shape[-2]//4+1,
                                          input_tensor.shape[-1]//4+1),
                             input_dim=input_dim, 
                             hidden_dims=hidden_dims,
                             n_layers=n_layers_convcell,
                             kernel_size=(3,3),
                             device=device)
        
        model  = EncoderRNN(phycell,
                              convcell,
                              input_channels=2,
                              input_dim=(input_tensor.shape[-2],input_tensor.shape[-1]),
                              _1d=False,
                              bc=bc,
                              device=device,
                              sigmoid=False,
                              small=config.model.small)
        
        constraints = th.zeros((49,7,7)).to(device)
        ind = 0
        for i in range(0,7):
            for j in range(0,7):
                constraints[ind,i,j] = 1
                ind +=1
                
    elif config.data.type == "allen_cahn":
        # Load samples, together with x, y, and t series
        t = np.load(os.path.join(data_path, "t_series.npy"))
        x = np.load(os.path.join(data_path, "x_series.npy"))
        u = th.tensor(np.load(os.path.join(data_path, "sample.npy")),
                             dtype=th.float).to(device=device)
        
        u = u + th.normal(th.zeros_like(u),th.ones_like(u)*config.data.noise)
        
        bc = th.tensor([[[0.0, 0.0]]]).to(device)
        
        sample_length = u.shape[0]
        input_tensor = u[:sample_length//2].unsqueeze(0).unsqueeze(-2).to(device=device)
        target_tensor = u[sample_length//2:].unsqueeze(0).unsqueeze(-2).to(device=device)
    
        # Initialize and set up the model
        phycell  =  PhyCell(input_shape=(input_tensor.shape[-1]//4+1),
                            input_dim=input_dim,
                            F_hidden_dims=[7],
                            n_layers=1,
                            kernel_size=7,
                            device=device) 
        
        convcell =  ConvLSTM(input_shape=(input_tensor.shape[-1]//4+1),
                             input_dim=input_dim,
                             hidden_dims=hidden_dims,
                             n_layers=n_layers_convcell,
                             kernel_size=3,
                             device=device)
        
        model  = EncoderRNN(phycell,
                              convcell,
                              input_channels=1,
                              input_dim=(input_tensor.shape[-1],),
                              _1d=True,
                              bc=bc,
                              device=device,
                              sigmoid=False,
                              small=config.model.small)
        
        constraints = th.zeros((7,7)).to(device)
        ind = 0
        for i in range(0,7):
            constraints[ind,i] = 1
            ind +=1

    elif config.data.type == "burger_2d":
        # Load samples, together with x, y, and t series
        t = np.load(os.path.join(data_path, "t_series.npy"))
        x = np.load(os.path.join(data_path, "x_series.npy"))
        y = np.load(os.path.join(data_path, "y_series.npy"))
        u = th.tensor(np.load(os.path.join(data_path, "sample.npy")),
                             dtype=th.float).to(device=device)
        
        u = u + th.normal(th.zeros_like(u),th.ones_like(u)*config.data.noise)
        
        sample_length = u.shape[0]
        input_tensor = u[:sample_length//2].unsqueeze(0).unsqueeze(-3).to(device=device)
        target_tensor = u[sample_length//2:].unsqueeze(0).unsqueeze(-3).to(device=device)
        
        # Initialize and set up the model
        bc = th.tensor([[[0.0, 0.0, 0.0, 0.0]]]).to(device=device)

        phycell  =  PhyCell(input_shape=(input_tensor.shape[-2]//4+1,
                                         input_tensor.shape[-1]//4+1),
                            input_dim=input_dim,
                            F_hidden_dims=[49],
                            n_layers=1,
                            kernel_size=(7,7),
                            device=device) 
        
        convcell =  ConvLSTM(input_shape=(input_tensor.shape[-2]//4+1,
                                          input_tensor.shape[-1]//4+1),
                             input_dim=input_dim, 
                             hidden_dims=hidden_dims,
                             n_layers=n_layers_convcell,
                             kernel_size=(3,3),
                             device=device)
        
        model  = EncoderRNN(phycell,
                              convcell,
                              input_channels=1,
                              input_dim=(input_tensor.shape[-2],input_tensor.shape[-1]),
                              _1d=False,
                              bc=bc,
                              device=device,
                              sigmoid=False,
                              small=config.model.small)
        
        constraints = th.zeros((49,7,7)).to(device)
        ind = 0
        for i in range(0,7):
            for j in range(0,7):
                constraints[ind,i,j] = 1
                ind +=1

    # Count number of trainable parameters
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Trainable model parameters: {pytorch_total_params}\n")

    # Load the trained weights from the checkpoints into the model
    model_path = os.path.join(os.path.abspath(""),
                              "checkpoints",
                              config.model.name,
                              config.model.name+".pt")
    model.load_state_dict(th.load(model_path))
    model.eval()

    time_start = time.time()
    with th.no_grad():
    
        input_length = input_tensor.size()[1]
        target_length = target_tensor.size()[1]
    
        predictions = []
        for ei in range(input_length-1):
            encoder_output, encoder_hidden, output_image,_,_  = model(input_tensor[:,ei], (ei==0))
            predictions.append(output_image.cpu())
                
        decoder_input = input_tensor[:,-1,:,:] # first decoder input= last image of input sequence
    
        for di in range(target_length):
            decoder_output, decoder_hidden, output_image,_,_ = model(decoder_input, False, False)
            decoder_input = output_image
            predictions.append(output_image.cpu())
    
        input = input_tensor.cpu().numpy()
        target = target_tensor.cpu().numpy()
            
        target = np.concatenate((input,target),axis=1)
        target = target[:,1:]
            
        predictions =  np.stack(predictions) # (nt, batch_size, channels, Nx, Ny)
        predictions = predictions.swapaxes(0,1)  # (batch_size, nt, channels, Nx, Ny)
      
    if print_progress:
      print(f"Forward pass took: {time.time() - time_start} seconds.")

    mse = np.mean((predictions - target)**2)
    print(f"MSE: {mse}")
        
    #
    # Visualize the data
    if config.data.type == "burger" and visualize:
        
        u_hat = np.transpose(predictions.squeeze())
        u = np.transpose(target.squeeze())
    
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
                                       frames=len(t) - 1,
                                       fargs=(line1, line2, u, u_hat),
                                       interval=20)
        plt.tight_layout()
        plt.draw()
        plt.show()
     
    elif config.data.type == "diffusion_sorption" and visualize:
        u_hat = np.transpose(predictions.squeeze().swapaxes(1,-1)[...,0])
        u = np.transpose(target.squeeze().swapaxes(1,-1)[...,0])
    
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
                                       frames=len(t) - 1,
                                       fargs=(line1, line2, u, u_hat),
                                       interval=20)
        plt.tight_layout()
        plt.draw()
        plt.show()
        
    elif config.data.type == "diffusion_reaction" and visualize:
        
        u_hat = predictions.squeeze().swapaxes(1,-1).swapaxes(-3,-2)
        u = target.squeeze().swapaxes(1,-1).swapaxes(-3,-2)
    
        # Plot u over space
        fig, ax = plt.subplots(2, 1, figsize=(6, 10))
    
        im1 = ax[0].imshow(u_hat[-1,:,:,0].squeeze().transpose(), interpolation='nearest', 
                     extent=[x.min(), x.max(),
                             y.min(), y.max()],
                     origin='lower', aspect='auto')
        fig.colorbar(im1, ax=ax[0])
        im1.set_clim(u[-1,:,:,0].min(), u[-1,:,:,0].max())
        im2 = ax[1].imshow(u[-1,:,:,0].squeeze().transpose(), interpolation='nearest',
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
                                        frames=len(t) - 1,
                                        fargs=(im1, im2, u_hat[...,0], u[...,0]),
                                        interval=20)
        
        plt.tight_layout()
        plt.show()
        
        # Plot v over space
        fig, ax = plt.subplots(2, 1, figsize=(6, 10))
    
        im1 = ax[0].imshow(u_hat[-1,:,:,1].squeeze().transpose(), interpolation='nearest',
                     extent=[x.min(), x.max(),
                             y.min(), y.max()],
                     origin='lower', aspect='auto')
        fig.colorbar(im1, ax=ax[0])
        im1.set_clim(u[-1,:,:,1].min(), u[-1,:,:,1].max())
        im2 = ax[1].imshow(u[-1,:,:,1].squeeze().transpose(), interpolation='nearest', 
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
        
        u_hat = np.transpose(predictions.squeeze())
        u = np.transpose(target.squeeze())
    
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
                                       frames=len(t) - 1,
                                       fargs=(line1, line2, u, u_hat),
                                       interval=20)
        plt.tight_layout()
        plt.draw()
        plt.show()

    elif config.data.type == "burger_2d" and visualize:
        
        u_hat = predictions.squeeze()
        u = target.squeeze()
    
        # Plot u over space
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
        im1 = ax[0].imshow(u_hat[-1].squeeze().transpose(), interpolation='nearest', 
                     extent=[x.min(), x.max(),
                             y.min(), y.max()],
                     origin='lower', aspect='auto')
        fig.colorbar(im1, ax=ax[0])
        im1.set_clim(u[-1].min(), u[-1].max())
        im2 = ax[1].imshow(u[-1].squeeze().transpose(), interpolation='nearest',
                     extent=[x.min(), x.max(),
                             y.min(), y.max()],
                     origin='lower', aspect='auto')
        fig.colorbar(im2, ax=ax[1])
        im2.set_clim(u[-1].min(), u[-1].max())
        ax[0].set_xlabel("$x$")
        ax[0].set_ylabel("$y$")
        ax[0].set_title('$u(x,y) predicted$', fontsize = 10)
        ax[1].set_xlabel("$x$")
        ax[1].set_ylabel("$y$")
        ax[1].set_title('$u(x,y) data$', fontsize = 10)
        #plt.show()
        
        # Animate through time
        anim = animation.FuncAnimation(fig,
                                        animate_2d,
                                        frames=len(t) - 1,
                                        fargs=(im1, im2, u_hat, u),
                                        interval=20)
        
        plt.tight_layout()
        plt.draw()
        plt.show()

    return mse


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
    im1.set_array(u_hat[t,:,:].squeeze().transpose())
    im2.set_array(u[t,:,:].squeeze().transpose())


if __name__ == "__main__":
    th.set_num_threads(1)
    
    run_testing(print_progress=True, visualize=True)

    print("Done.")