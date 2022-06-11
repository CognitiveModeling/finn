#! env/bin/python3

"""
Main file for training a model with FINN
"""

import numpy as np
import torch as th
import torch.nn as nn
import os
import time
from threading import Thread
import sys

sys.path.append("../models")
from utils.configuration import Configuration
import utils.helper_functions as helpers



def run_training(x, t, u0, data, model, cfg, print_progress=True):

    # Set device on GPU if specified in the configuration file, else CPU
    # device = helpers.determine_device()
    config = cfg.config
    device = config.general.device
    model = model.to(device)
    x = x.to(device)
    t = t.to(device)
    u0 = u0.to(device)
    data = data.to(device)
    
    # Count number of trainable parameters
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    
    if print_progress:
        print("Trainable model parameters:", pytorch_total_params)

    # If desired, restore the network by loading the weights saved in the .pt
    # file
    if config.training.continue_training:
        if print_progress: 
            print('Restoring model (that is the network\'s weights) from file...')
        model.load_state_dict(th.load(os.path.join(os.path.abspath(""),
                                      "checkpoints",
                                      config.model.name,
                                      config.model.name + ".pt")))
        model.train()

    #
    # Set up an optimizer and the criterion (loss)
    optimizer = th.optim.LBFGS(model.parameters(),
                                lr=config.training.learning_rate)

    criterion = nn.MSELoss(reduction="sum")

    #
    # Set up lists to save and store the epoch errors
    epoch_errors_train = []
    
    best_train = np.infty
    
    """
    TRAINING
    """

    a = time.time()

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
            u_hat = model(t=t, u=u0)
            
            # Calculate predicted breakthrough curve
            cauchy_mult = cfg.cauchy_mult * cfg.D_eff[0] * cfg.dx
            pred = ((u_hat[:,-2,0] - u_hat[:,-1,0]) * cauchy_mult).squeeze()
            
            loss_data = criterion(1e3 * pred, 1e3 * data)
            
            # Calculate physical loss based on retardation factor monotonicity
            inp_temp = th.linspace(0,2,501).unsqueeze(-1).to(device)
            ret_temp = 1/(model.func_nn(inp_temp))[...,0]
            
            loss_phys = 100*th.mean(th.relu(ret_temp[1:]-ret_temp[:-1]))
            
            loss = loss_data + loss_phys
            
            loss.backward()
            
            if print_progress:
                print(loss_data.item(), loss_phys.item(), loss.item())
                
            return loss
        
        optimizer.step(closure)
            
        # Extract the MSE value from the closure function
        loss = closure()
        
        epoch_errors_train.append(loss.item())

        # Create a plus or minus sign for the training error
        train_sign = "(-)"
        if epoch_errors_train[-1] < best_train:
            train_sign = "(+)"
            best_train = epoch_errors_train[-1]
            # Save the model to file (if desired)
            if config.training.save_model:
                # Start a separate thread to save the model
                thread = Thread(target=helpers.save_model_to_file(
                    model_src_path=os.path.abspath(""),
                    config=config,
                    epoch=epoch,
                    epoch_errors_train=epoch_errors_train,
                    epoch_errors_valid=epoch_errors_train,
                    net=model))
                thread.start()

        #
        # Print progress to the console
        if print_progress:
            print(f"Epoch {str(epoch+1).zfill(int(np.log10(config.training.epochs))+1)}/{str(config.training.epochs)} took {str(np.round(time.time() - epoch_start_time, 2)).ljust(5, '0')} seconds. \t\tAverage epoch training error: {train_sign}{str(np.round(epoch_errors_train[-1], 10)).ljust(12, ' ')}")

    b = time.time()
    if print_progress:
        print('\nTraining took ' + str(np.round(b - a, 2)) + ' seconds.\n\n')