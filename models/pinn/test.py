#! env/bin/python3

"""
Main file for testing (evaluating) a model
"""

import numpy as np
import torch as th
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import time

sys.path.append("..")
from utils.configuration import Configuration
from pinn import *


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

    # Set device on GPU if specified in the configuration file, else CPU
    device = th.device(config.general.device)
    
    # Initialize and set up model and data
    if config.data.type == "burger":
        model = PINN_Burger(
            layer_sizes=config.model.layer_sizes,
            device=device
        )

        # Set up the test data
        data_path = os.path.join(os.path.abspath("../../data"),
                                 config.data.type,
                                 config.data.name)
        sample_u = np.load(os.path.join(data_path, "sample.npy"))
        t_series = np.load(os.path.join(data_path, "t_series.npy"))
        x_series = np.load(os.path.join(data_path, "x_series.npy"))

        (ts, xs) = np.array(
            np.meshgrid(t_series, x_series, indexing="ij"),
            dtype=np.float32
        )
        t_tensor = th.unsqueeze(th.tensor(ts.flatten()), 1).to(device=device)
        x_tensor = th.unsqueeze(th.tensor(xs.flatten()), 1).to(device=device)

    elif config.data.type == "diffusion_sorption":
        model = PINN_DiffSorp(
            layer_sizes=config.model.layer_sizes,
            device=device,
            config=config
        )

        # Set up the test data
        data_path = os.path.join(os.path.abspath("../../data"),
                                 config.data.type,
                                 config.data.name)
        sample_c = np.load(os.path.join(data_path, "sample_c.npy"))
        sample_ct = np.load(os.path.join(data_path, "sample_ct.npy"))
        sample_c_ct = np.stack((sample_c, sample_ct), axis=2)
        t_series = np.load(os.path.join(data_path, "t_series.npy"))
        x_series = np.load(os.path.join(data_path, "x_series.npy"))

        (ts, xs) = np.array(
            np.meshgrid(t_series, x_series, indexing="ij"),
            dtype=np.float32
        )
        t_tensor = th.unsqueeze(th.tensor(ts.flatten()), 1).to(device=device)
        x_tensor = th.unsqueeze(th.tensor(xs.flatten()), 1).to(device=device)

    elif config.data.type == "diffusion_reaction":
        model = PINN_DiffReact(
            layer_sizes=config.model.layer_sizes,
            device=device,
            config=config
        )

        # Set up the test data
        data_path = os.path.join(os.path.abspath("../../data"),
                                 config.data.type,
                                 config.data.name)
        sample_u = np.load(os.path.join(data_path, "sample_u.npy"))
        sample_v = np.load(os.path.join(data_path, "sample_v.npy"))
        t_series = np.load(os.path.join(data_path, "t_series.npy"))
        x_series = np.load(os.path.join(data_path, "x_series.npy"))
        y_series = np.load(os.path.join(data_path, "y_series.npy"))

        (ts, xs, ys) = np.array(
            np.meshgrid(t_series, x_series, y_series, indexing="ij"),
            dtype=np.float32
        )
        t_tensor = th.unsqueeze(th.tensor(ts.flatten()), 1).to(device=device)
        x_tensor = th.unsqueeze(th.tensor(xs.flatten()), 1).to(device=device)
        y_tensor = th.unsqueeze(th.tensor(ys.flatten()), 1).to(device=device)
    
    elif config.data.type == "allen_cahn":
        model = PINN_AllenCahn(
            layer_sizes=config.model.layer_sizes,
            device=device
        )

        # Set up the test data
        data_path = os.path.join(os.path.abspath("../../data"),
                                 config.data.type,
                                 config.data.name)
        sample_u = np.load(os.path.join(data_path, "sample.npy"))
        t_series = np.load(os.path.join(data_path, "t_series.npy"))
        x_series = np.load(os.path.join(data_path, "x_series.npy"))

        (ts, xs) = np.array(
            np.meshgrid(t_series, x_series, indexing="ij"),
            dtype=np.float32
        )
        t_tensor = th.unsqueeze(th.tensor(ts.flatten()), 1).to(device=device)
        x_tensor = th.unsqueeze(th.tensor(xs.flatten()), 1).to(device=device)

    elif config.data.type == "burger_2d":
        model = PINN_Burger2D(
            layer_sizes=config.model.layer_sizes,
            device=device
        )

        # Set up the test data
        data_path = os.path.join(os.path.abspath("../../data"),
                                 config.data.type,
                                 config.data.name)
        sample_u = np.load(os.path.join(data_path, "sample.npy"))
        t_series = np.load(os.path.join(data_path, "t_series.npy"))
        x_series = np.load(os.path.join(data_path, "x_series.npy"))
        y_series = np.load(os.path.join(data_path, "y_series.npy"))

        (ts, xs, ys) = np.array(
            np.meshgrid(t_series, x_series, y_series, indexing="ij"),
            dtype=np.float32
        )
        t_tensor = th.unsqueeze(th.tensor(ts.flatten()), 1).to(device=device)
        x_tensor = th.unsqueeze(th.tensor(xs.flatten()), 1).to(device=device)
        y_tensor = th.unsqueeze(th.tensor(ys.flatten()), 1).to(device=device)

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
    # Forward data through the model and compute the error
    if config.data.type == "burger":
        u_hat, _ = model.forward(t=t_tensor, x=x_tensor)
        u_hat = u_hat.view(len(t_series), len(x_series)).detach().cpu().numpy()
        
        pred = u_hat
        labels = sample_u
        
        mse_u = np.mean(np.square(sample_u - u_hat))
        mse = mse_u

    elif config.data.type == "diffusion_sorption":
        c_hat, ct_hat, _, _ = model.forward(t=t_tensor, x=x_tensor)
        c_hat = c_hat.view(
            len(t_series), len(x_series)
        ).detach().cpu().numpy()
        ct_hat = ct_hat.view(
            len(t_series), len(x_series)
        ).detach().cpu().numpy()
        
        pred = np.stack((c_hat,ct_hat),axis=-1)
        labels = np.stack((sample_c, sample_ct),axis=-1)
        
        mse_c = np.mean(np.square(sample_c - c_hat))
        mse_ct = np.mean(np.square(sample_ct - ct_hat))
        mse = (mse_c + mse_ct)/2

    elif config.data.type == "diffusion_reaction":
        
        """
        # if "ext" in config.data.name:
        # Partition the batch number due to extremely high memory usage
        u_hat = th.zeros(len(t_series)*len(x_series)*len(y_series),1)
        v_hat = th.zeros(len(t_series)*len(x_series)*len(y_series),1)
        for i in range(len(t_series)):
            t_inp = t_tensor[i*len(x_series)*len(y_series):(i+1)*len(x_series)*len(y_series)]
            x_inp = x_tensor[i*len(x_series)*len(y_series):(i+1)*len(x_series)*len(y_series)]
            y_inp = y_tensor[i*len(x_series)*len(y_series):(i+1)*len(x_series)*len(y_series)]
            u_hat[i*len(x_series)*len(y_series):(i+1)*len(x_series)*len(y_series)], \
                v_hat[i*len(x_series)*len(y_series):(i+1)*len(x_series)*len(y_series)], \
                    _, _ = model.forward(t=t_inp, x=x_inp, y=y_inp)
        """

        time_start = time.time()
        u_hat = th.zeros(len(t_series)*len(x_series)*len(y_series),1)
        v_hat = th.zeros(len(t_series)*len(x_series)*len(y_series),1)
        u_hat, v_hat, _, _ = model.forward(t=t_tensor, x=x_tensor, y=y_tensor)
        if print_progress:
            print(f"Forward pass took: {time.time() - time_start} seconds.")
        
        u_hat = u_hat.view(
            len(t_series), len(x_series), len(y_series)
        ).detach().cpu().numpy()
        v_hat = v_hat.view(
            len(t_series), len(x_series), len(y_series)
        ).detach().cpu().numpy()
        
        pred = np.stack((u_hat,v_hat),axis=-1)
        labels = np.stack((sample_u, sample_v),axis=-1)

        mse_u = np.mean(np.square(sample_u - u_hat))
        mse_v = np.mean(np.square(sample_v - v_hat))
        mse = (mse_u + mse_v)/2
    
    elif config.data.type == "allen_cahn":
        u_hat, _ = model.forward(t=t_tensor, x=x_tensor)
        u_hat = u_hat.view(len(t_series), len(x_series)).detach().cpu().numpy()
        
        pred = u_hat
        labels = sample_u
        
        mse_u = np.mean(np.square(sample_u - u_hat))
        mse = mse_u

    elif config.data.type == "burger_2d":
        
        """
        # if "ext" in config.data.name:
        # Partition the batch number due to extremely high memory usage
        u_hat = th.zeros(len(t_series)*len(x_series)*len(y_series),1)
        v_hat = th.zeros(len(t_series)*len(x_series)*len(y_series),1)
        for i in range(len(t_series)):
            t_inp = t_tensor[i*len(x_series)*len(y_series):(i+1)*len(x_series)*len(y_series)]
            x_inp = x_tensor[i*len(x_series)*len(y_series):(i+1)*len(x_series)*len(y_series)]
            y_inp = y_tensor[i*len(x_series)*len(y_series):(i+1)*len(x_series)*len(y_series)]
            u_hat[i*len(x_series)*len(y_series):(i+1)*len(x_series)*len(y_series)], \
                v_hat[i*len(x_series)*len(y_series):(i+1)*len(x_series)*len(y_series)], \
                    _, _ = model.forward(t=t_inp, x=x_inp, y=y_inp)
        """

        time_start = time.time()
        u_hat = th.zeros(len(t_series)*len(x_series)*len(y_series),1)
        u_hat, _ = model.forward(t=t_tensor, x=x_tensor, y=y_tensor)
        if print_progress:
            print(f"Forward pass took: {time.time() - time_start} seconds.")
        
        u_hat = u_hat.view(
            len(t_series), len(x_series), len(y_series)
        ).detach().cpu().numpy()
        
        pred = u_hat
        labels = sample_u

        mse = np.mean(np.square(sample_u - u_hat))

    print(f"MSE: {mse}")

    #
    # Visualize the data
    if visualize:
        plt.style.use("dark_background")

        # Plot over space and time
        fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        if config.data.type == "burger":

            im1 = ax[0].imshow(
                np.transpose(sample_u), interpolation='nearest',
                origin='lower', aspect='auto'
            )
            fig.colorbar(im1, ax=ax[0])
            im2 = ax[1].imshow(
                np.transpose(u_hat), interpolation='nearest',
                origin='lower', aspect='auto'
            )
            fig.colorbar(im2, ax=ax[1])

            ax[0].set_xlabel("t")
            ax[0].set_ylabel("x")
            ax[1].set_xlabel("t")


        elif config.data.type == "diffusion_sorption":

            im1 = ax[0].imshow(
                np.transpose(sample_c), interpolation='nearest',
                origin='lower', aspect='auto'
            )
            fig.colorbar(im1, ax=ax[0])
            im2 = ax[1].imshow(
                np.transpose(c_hat), interpolation='nearest',
                origin='lower', aspect='auto'
            )
            fig.colorbar(im2, ax=ax[1])

            ax[0].set_xlabel("t")
            ax[0].set_ylabel("x")
            ax[1].set_xlabel("t")
            

        elif config.data.type == "diffusion_reaction":
            
            im1 = ax[0].imshow(
                np.transpose(sample_u[-1]), interpolation='nearest',
                origin='lower', aspect='auto', vmin=-0.4,
                vmax=0.4
            )
            fig.colorbar(im1, ax=ax[0])
            im2 = ax[1].imshow(
                np.transpose(u_hat[-1]), interpolation='nearest',
                origin='lower', aspect='auto', vmin=-0.4,
                vmax=0.4
            )
            fig.colorbar(im2, ax=ax[1])

            ax[0].set_xlabel("x")
            ax[0].set_ylabel("y")
            ax[1].set_xlabel("x")
        
        elif config.data.type == "allen_cahn":

            im1 = ax[0].imshow(
                np.transpose(sample_u), interpolation='nearest',
                origin='lower', aspect='auto'
            )
            fig.colorbar(im1, ax=ax[0])
            im2 = ax[1].imshow(
                np.transpose(u_hat), interpolation='nearest',
                origin='lower', aspect='auto'
            )
            fig.colorbar(im2, ax=ax[1])

            ax[0].set_xlabel("t")
            ax[0].set_ylabel("x")
            ax[1].set_xlabel("t")

        elif config.data.type == "burger_2d":
            
            im1 = ax[0].imshow(
                np.transpose(sample_u[-1]), interpolation='nearest',
                origin='lower', aspect='auto', vmin=-0.4,
                vmax=0.4
            )
            fig.colorbar(im1, ax=ax[0])
            im2 = ax[1].imshow(
                np.transpose(u_hat[-1]), interpolation='nearest',
                origin='lower', aspect='auto', vmin=-0.4,
                vmax=0.4
            )
            fig.colorbar(im2, ax=ax[1])

            ax[0].set_xlabel("x")
            ax[0].set_ylabel("y")
            ax[1].set_xlabel("x")
        
        
        ax[0].set_title("Ground Truth")
        ax[1].set_title("Network Output")


        anim = animation.FuncAnimation(
                fig,
                animate,
                frames=len(t_series),
                fargs=(im1, im2, sample_u, u_hat),
                interval=20
            )
        plt.show()
    
    #
    # Compute and return statistics
    mse = np.mean(np.square(pred - labels))

    return mse, pred, labels


def animate(t, im1, im2, sample_u, u_hat):
    """
    Data animation function animating an image over time.
    :param t: The current time step
    :param axis: The matplotlib image object
    :param field: The data field
    :return: The matplotlib image object updated with the current time step's
        image date
    """
    im1.set_array(sample_u[t])
    im2.set_array(u_hat[t])



if __name__ == "__main__":
    th.set_num_threads(1)
    
    pred, u = run_testing(print_progress=True, visualize=True)
    
    print("Done.")