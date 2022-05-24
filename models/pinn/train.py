#! env/bin/python3

"""
Main file for training a model
"""

import numpy as np
import torch as th
import os
import time
from threading import Thread
import sys

sys.path.append("..")
from utils.configuration import Configuration
import utils.helper_functions as helpers
from pinn import *


def run_training(print_progress=False, model_number=None):

    th.set_num_threads(1)

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

    # Initialize and set up the model
    if config.data.type == "burger":
        model = PINN_Burger(
            layer_sizes=config.model.layer_sizes,
            device=device
        )
    elif config.data.type == "diffusion_sorption":
        model = PINN_DiffSorp(
            layer_sizes=config.model.layer_sizes,
            device=device,
            config=config
        )
    elif config.data.type == "diffusion_reaction":
        model = PINN_DiffReact(
            layer_sizes=config.model.layer_sizes,
            device=device,
            config=config
        )
    elif config.data.type == "allen_cahn":
        model = PINN_AllenCahn(
            layer_sizes=config.model.layer_sizes,
            device=device
        )
    elif config.data.type == "burger_2d":
        model = PINN_Burger2D(
            layer_sizes=config.model.layer_sizes,
            device=device
        )

    # Count number of trainable parameters
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("Trainable model parameters:", pytorch_total_params)

    # If desired, restore the network by loading the weights saved in the .pt
    # file
    if config.training.continue_training:
        print('Restoring model (that is the network\'s weights) from file...')
        model.load_state_dict(th.load(os.path.join(os.path.abspath(""),
                                      "checkpoints",
                                      config.model.name,
                                      config.model.name + ".pt")))
        model.train()

    #
    # Set up an optimizer and the criterion (loss)
    optimizer = th.optim.LBFGS(model.parameters(),
                               lr=config.training.learning_rate,
                               line_search_fn="strong_wolfe")
    # If using mini-batch training, we cannot use LBFGS optimizer
    if config.training.batch_size < config.validation.batch_size:
        optimizer = th.optim.Adam(model.parameters(),
                                  lr=config.training.learning_rate)

    print(optimizer)

    #
    # Set up lists to save and store the epoch errors
    epoch_errors_train = []
    epoch_errors_valid = []
    best_train = np.infty
    best_valid = np.infty

    #
    # Set up the training and validation datasets and -loaders
    dataloader_train = helpers.build_dataloader(
        config=config, mode="train", batch_size=config.training.batch_size
    )
    dataloader_valid = helpers.build_dataloader(
        config=config, mode="val", batch_size=config.validation.batch_size
    )

    """
    TRAINING
    """

    a = time.time()
    
    sample = next(iter(dataloader_train))

    # Move data to device and separate the sample in its subcomponents
    sample = sample.to(device=device)
    if config.data.type == "burger":
        t, x, u, _, _ = th.split(sample, 1, dim=1)
    elif config.data.type == "diffusion_sorption":
        t, x, c, ct, _, _ = th.split(sample, 1, dim=1)
    elif config.data.type == "diffusion_reaction":
        t, x, y, u, v, _, _, _ = th.split(sample, 1, dim=1)
    elif config.data.type == "allen_cahn":
        t, x, u, _, _ = th.split(sample, 1, dim=1)
    elif config.data.type == "burger_2d":
        t, x, y, u, _, _, _ = th.split(sample, 1, dim=1)
        
    sample_valid = next(iter(dataloader_valid))

    # Move data to device and separate the sample in its subcomponents
    sample_valid = sample_valid.to(device=device)
    if config.data.type == "burger":
        t_valid, x_valid, u_valid, _, _ = th.split(sample_valid, 1, dim=1)
    elif config.data.type == "diffusion_sorption":
        t_valid, x_valid, c_valid, ct_valid, _, _ = th.split(sample_valid, 1, dim=1)
    elif config.data.type == "diffusion_reaction":
        t_valid, x_valid, y_valid, u_valid, v_valid, _, _, _ = th.split(sample_valid, 1, dim=1)
    elif config.data.type == "allen_cahn":
        t_valid, x_valid, u_valid, _, _ = th.split(sample_valid, 1, dim=1)
    elif config.data.type == "burger_2d":
        t_valid, x_valid, y_valid, u_valid, _, _, _ = th.split(sample_valid, 1, dim=1)

    #
    # Start the training and iterate over all epochs
    for epoch in range(config.training.epochs):

        epoch_start_time = time.time()
        
        # Define the closure function that consists of resetting the
        # gradient buffer, loss function calculation, and backpropagation
        # It is necessary for LBFGS optimizer, because it requires multiple
        # function evaluations
        def closure():
            # Set the model to train mode
            model.train()
            
            # Reset the optimizer to clear data from previous iterations
            optimizer.zero_grad()

            # Forward propagate and calculate loss function
            if config.data.type == "burger":
                u_hat, f_hat = model.forward(t=t, x=x)
                mse_u = th.mean(th.square(u_hat - u))
                mse_f = th.mean(th.square(f_hat))
                mse = mse_u + mse_f

            elif config.data.type == "diffusion_sorption":
                c_hat, ct_hat, f_hat, g_hat = model.forward(t=t, x=x)
                mse_c = th.mean(th.square(c_hat - c))
                mse_ct = th.mean(th.square(ct_hat - ct))
                mse_f = th.mean(th.square(f_hat))
                mse_g = th.mean(th.square(g_hat))
                mse = mse_c + mse_ct + mse_f + mse_g

            elif config.data.type == "diffusion_reaction":
                u_hat, v_hat, f_hat, g_hat = model.forward(t=t, x=x, y=y)
                mse_u = th.mean(th.square(u_hat - u))
                mse_v = th.mean(th.square(v_hat - v))
                mse_f = th.mean(th.square(f_hat))
                mse_g = th.mean(th.square(g_hat))
                mse = mse_u + mse_v + mse_f + mse_g
                
            elif config.data.type == "allen_cahn":
                u_hat, f_hat = model.forward(t=t, x=x)
                mse_u = th.mean(th.square(u_hat - u))
                mse_f = th.mean(th.square(f_hat))
                mse = mse_u + mse_f

            elif config.data.type == "burger_2d":
                u_hat, f_hat = model.forward(t=t, x=x, y=y)
                mse_u = th.mean(th.square(u_hat - u))
                mse_f = th.mean(th.square(f_hat))
                mse = mse_u + mse_f

            mse.backward()
            
            #print(mse_u.item(), mse_f.item())
            
            return mse
                
        optimizer.step(closure)
        
        # Extract the MSE value from the closure function
        mse = closure()

        epoch_errors_train.append(mse.item())

        # Create a plus or minus sign for the training error
        train_sign = "(-)"
        if epoch_errors_train[-1] < best_train:
            train_sign = "(+)"
            best_train = epoch_errors_train[-1]

        #
        # VALIDATION
        
        sequence_errors_valid = []
        
        if config.data.type == "burger":
            u_hat_valid, f_hat_valid = model.forward(t=t_valid, x=x_valid)
            mse_u_valid = th.mean(th.square(u_hat_valid - u_valid))
            mse_f_valid = th.mean(th.square(f_hat_valid))
            mse_valid = mse_u_valid + mse_f_valid

        elif config.data.type == "diffusion_sorption":
            c_hat_valid, ct_hat_valid, f_hat_valid, g_hat_valid = model.forward(t=t_valid, x=x_valid)
            mse_c_valid = th.mean(th.square(c_hat_valid - c_valid))
            mse_ct_valid = th.mean(th.square(ct_hat_valid - ct_valid))
            mse_f_valid = th.mean(th.square(f_hat_valid))
            mse_g_valid = th.mean(th.square(g_hat_valid))
            mse_valid = mse_c_valid + mse_ct_valid + mse_f_valid + mse_g_valid
            
        elif config.data.type == "diffusion_reaction":
            u_hat_valid, v_hat_valid, f_hat_valid, g_hat_valid = model.forward(t=t_valid, x=x_valid, y=y_valid)
            mse_u_valid = th.mean(th.square(u_hat_valid - u_valid))
            mse_v_valid = th.mean(th.square(v_hat_valid - v_valid))
            mse_f_valid = th.mean(th.square(f_hat_valid))
            mse_g_valid = th.mean(th.square(g_hat_valid))
            mse_valid = mse_u_valid + mse_v_valid + mse_f_valid + mse_g_valid
        
        elif config.data.type == "allen_cahn":
            u_hat_valid, f_hat_valid = model.forward(t=t_valid, x=x_valid)
            mse_u_valid = th.mean(th.square(u_hat_valid - u_valid))
            mse_f_valid = th.mean(th.square(f_hat_valid))
            mse_valid = mse_u_valid + mse_f_valid

        elif config.data.type == "burger_2d":
            u_hat_valid, f_hat_valid = model.forward(t=t_valid, x=x_valid, y=y_valid)
            mse_u_valid = th.mean(th.square(u_hat_valid - u_valid))
            mse_f_valid = th.mean(th.square(f_hat_valid))
            mse_valid = mse_u_valid + mse_f_valid
            
        epoch_errors_valid.append(mse_valid.item())
            
        # Save the model to file (if desired)
        if config.training.save_model \
            and mse_valid < best_valid:
            
            # Start a separate thread to save the model
            thread = Thread(target=helpers.save_model_to_file(
                model_src_path=os.path.abspath(""),
                config=config,
                epoch=epoch,
                epoch_errors_train=epoch_errors_train,
                epoch_errors_valid=epoch_errors_valid,
                net=model))
            thread.start()

        # Create a plus or minus sign for the validation error
        valid_sign = "(-)"
        if epoch_errors_valid[-1] < best_valid:
            best_valid = epoch_errors_valid[-1]
            valid_sign = "(+)"
        
        #
        # Print progress to the console
        print(f"Epoch {str(epoch+1).zfill(int(np.log10(config.training.epochs))+1)}/{str(config.training.epochs)} took {str(np.round(time.time() - epoch_start_time, 2)).ljust(5, '0')} seconds. \t\tAverage epoch training error: {train_sign}{str(np.round(epoch_errors_train[-1], 10)).ljust(12, ' ')} \t\tValidation error: {valid_sign}{str(np.round(epoch_errors_valid[-1], 10)).ljust(12, ' ')}")

    b = time.time()
    print('\nTraining took ' + str(np.round(b - a, 2)) + ' seconds.\n\n')


if __name__ == "__main__":
    
    run_training(print_progress=True)

    print("Done.")