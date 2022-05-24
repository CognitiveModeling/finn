#! env/bin/python3

"""
Main file for training a model
"""

import numpy as np
import torch as th
import torch.nn as nn
import os
import time
from threading import Thread
import random
import sys

sys.path.append("..")
from utils.configuration import Configuration
import utils.helper_functions as helpers
from fno import FNO1d, FNO2d


def run_training(print_progress=True, model_number=None):

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
       # Load samples
        u = th.tensor(np.load(os.path.join(data_path, "sample.npy")),
                             dtype=th.float).to(device=device)
        
        u = u + th.normal(th.zeros_like(u),th.ones_like(u)*config.data.noise)

        input_tensor = u.permute(1,0).unsqueeze(0)[...,:10].unsqueeze(-1).to(device=device)
        target_tensor = u.permute(1,0).unsqueeze(0).unsqueeze(-1).to(device=device)
        
        x = th.tensor(np.load(os.path.join(data_path, "x_series.npy")),
                             dtype=th.float).to(device=device)
        t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                             dtype=th.float).to(device=device)
        
        grid = x.unsqueeze(-1).unsqueeze(0)
        
        bc = th.tensor([[[0.0, 0.0]]]).to(device)
        
        # Initialize and set up the model
        model = FNO1d(num_channels=1, bc=bc).to(device=device)
    
    elif config.data.type == "diffusion_sorption":
        # Load samples
        sample_c = th.tensor(np.load(os.path.join(data_path, "sample_c.npy")),
                             dtype=th.float).to(device=device)
        sample_ct = th.tensor(np.load(os.path.join(data_path, "sample_ct.npy")),
                             dtype=th.float).to(device=device)
        
        u = th.stack((sample_c, sample_ct), dim=1)
        
        u = u + th.normal(th.zeros_like(u),th.ones_like(u)*config.data.noise)

        input_tensor = u.permute(2,0,1).unsqueeze(0)[...,:10,:].to(device=device)
        target_tensor = u.permute(2,0,1).unsqueeze(0).to(device=device)
        
        x = th.tensor(np.load(os.path.join(data_path, "x_series.npy")),
                             dtype=th.float).to(device=device)
        t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                             dtype=th.float).to(device=device)
        
        grid = x.unsqueeze(-1).unsqueeze(0)
        
        bc = th.tensor([[[1.0, 0.0], [1.0, 0.0]]]).to(device)
        
        # Initialize and set up the model
        model = FNO1d(num_channels=2, bc=bc).to(device=device)
    
    elif config.data.type == "diffusion_reaction":
        # Load samples
        sample_u = th.tensor(np.load(os.path.join(data_path, "sample_u.npy")),
                             dtype=th.float).to(device=device)
        sample_v = th.tensor(np.load(os.path.join(data_path, "sample_v.npy")),
                             dtype=th.float).to(device=device)
        
        u = th.stack((sample_u, sample_v), dim=1)
        
        u = u + th.normal(th.zeros_like(u),th.ones_like(u)*config.data.noise)

        input_tensor = u.permute(2,3,0,1).unsqueeze(0)[...,:10,:].to(device=device)
        target_tensor = u.permute(2,3,0,1).unsqueeze(0).to(device=device)
        
        x = th.tensor(np.load(os.path.join(data_path, "x_series.npy")),
                             dtype=th.float).to(device=device)
        y = th.tensor(np.load(os.path.join(data_path, "y_series.npy")),
                             dtype=th.float).to(device=device)
        t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                             dtype=th.float).to(device=device)
        
        X, Y = th.meshgrid(x,y)
        
        grid = th.stack((X.unsqueeze(0), Y.unsqueeze(0)), dim=-1)
        
        bc = th.tensor([[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]).to(device=device)
        
        # Initialize and set up the model
        model = FNO2d(num_channels=2, bc=bc).to(device=device)
        
    elif config.data.type == "allen_cahn":
       # Load samples
        u = th.tensor(np.load(os.path.join(data_path, "sample.npy")),
                             dtype=th.float).to(device=device)
        
        u = u + th.normal(th.zeros_like(u),th.ones_like(u)*config.data.noise)

        input_tensor = u.permute(1,0).unsqueeze(0)[...,:10].unsqueeze(-1).to(device=device)
        target_tensor = u.permute(1,0).unsqueeze(0).unsqueeze(-1).to(device=device)
        
        x = th.tensor(np.load(os.path.join(data_path, "x_series.npy")),
                             dtype=th.float).to(device=device)
        t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                             dtype=th.float).to(device=device)
        
        grid = x.unsqueeze(-1).unsqueeze(0)
        
        bc = th.tensor([[[0.0, 0.0]]]).to(device)
        
        # Initialize and set up the model
        model = FNO1d(num_channels=1, bc=bc).to(device=device)

    elif config.data.type == "burger_2d":
        # Load samples
        u = th.tensor(np.load(os.path.join(data_path, "sample.npy")),
                             dtype=th.float).to(device=device)
        
        u = u + th.normal(th.zeros_like(u),th.ones_like(u)*config.data.noise)

        sample_length = u.shape[0]
        input_tensor = u.permute(1,2,0).unsqueeze(0)[...,:10].unsqueeze(-1).to(device=device)
        target_tensor = u.permute(1,2,0).unsqueeze(0).unsqueeze(-1).to(device=device)
        
        x = th.tensor(np.load(os.path.join(data_path, "x_series.npy")),
                             dtype=th.float).to(device=device)
        y = th.tensor(np.load(os.path.join(data_path, "y_series.npy")),
                             dtype=th.float).to(device=device)
        t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                             dtype=th.float).to(device=device)
        
        X, Y = th.meshgrid(x,y)
        
        grid = th.stack((X.unsqueeze(0), Y.unsqueeze(0)), dim=-1)
        
        bc = th.tensor([[[0.0, 0.0, 0.0, 0.0]]]).to(device=device)
        
        # Initialize and set up the model
        model = FNO2d(num_channels=1, bc=bc).to(device=device)
    
    if print_progress:
        
      # Count number of trainable parameters
      pytorch_total_params = sum(
          p.numel() for p in model.parameters() if p.requires_grad
      )
      print("Total trainable model parameters:", pytorch_total_params)

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
    optimizer = th.optim.Adam(model.parameters(),
                                lr=config.training.learning_rate)

    criterion = nn.MSELoss(reduction="mean")

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
            optimizer.zero_grad()
            # input_tensor : th.Size([batch_size, Nx, Ny, Nt, num_channels])
            # target_tensor : th.Size([batch_size, Nx, Ny, Nt, num_channels])
            
            # Use the first 10 time steps for initial condition
            pred = target_tensor[...,:10,:]
            inp_shape = list(input_tensor.shape)
            inp_shape = inp_shape[:-2]
            inp_shape.append(-1)
            xx = input_tensor
            
            # Loop through the time steps
            for i in range(10, len(t)):
                # Reshape the input tensor to collect all channels together
                # at the last dimension
                inp_temp = xx.reshape(inp_shape)
                # Forward run of the model
                pred_temp = model(inp_temp,grid)
                # Update the prediction tensor
                pred = th.cat((pred, pred_temp), dim=-2)
                # Update the input tensor with the prediction
                xx = th.cat((xx[...,1:,:], pred_temp), dim=-2)
            
            mse = criterion(pred, target_tensor)
            
            mse.backward()
            
            print(mse.item())
            
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


if __name__ == "__main__":
    th.set_num_threads(1)
    run_training()

    print("Done.")