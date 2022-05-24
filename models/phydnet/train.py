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
from phydnet import ConvLSTM, PhyCell, EncoderRNN
from constrain_moments import K2M


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
    
    if config.model.small:
        input_dim = 32
        hidden_dims = [32]
        n_layers_convcell = 1
    else:
        input_dim = 64
        hidden_dims = [128,128,64]
        n_layers_convcell = 3

    if config.data.type == "burger":
        # Load samples
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
        # Load samples
        sample_c = th.tensor(np.load(os.path.join(data_path, "sample_c.npy")),
                             dtype=th.float).to(device=device)
        sample_ct = th.tensor(np.load(os.path.join(data_path, "sample_ct.npy")),
                             dtype=th.float).to(device=device)
        
        u = th.stack((sample_c, sample_ct), dim=1)
        
        u = u + th.normal(th.zeros_like(u),th.ones_like(u)*config.data.noise)

        sample_length = u.shape[0]
        input_tensor = u[:sample_length//2].unsqueeze(0).to(device=device)
        target_tensor = u[sample_length//2:].unsqueeze(0).to(device=device)
        
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
        # Load samples
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
        # Load samples
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
        # Load samples
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
    
    if print_progress:
      print(phycell)
      print(convcell)
      print(model)
        
      # Count number of trainable parameters
      pytorch_total_params = sum(
          p.numel() for p in phycell.parameters() if p.requires_grad
      )
      print("PhyCell parameters:", pytorch_total_params)
      pytorch_total_params = sum(
          p.numel() for p in convcell.parameters() if p.requires_grad
      )
      print("ConvLSTM parameters:", pytorch_total_params)
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
        teacher_forcing_ratio = np.maximum(0 , 1 - epoch * 0.003)
        
        # Define the closure function that consists of resetting the
        # gradient buffer, loss function calculation, and backpropagation
        # It is necessary for LBFGS optimizer, because it requires multiple
        # function evaluations
        def closure():
            optimizer.zero_grad()
            # input_tensor : torch.Size([batch_size, input_length, channels, cols, rows])
            input_length  = input_tensor.size(1)
            target_length = target_tensor.size(1)
            mse = 0
            for ei in range(input_length-1): 
                encoder_output, encoder_hidden, output_image,_,_ = model(input_tensor[:,ei], (ei==0) )
                mse += criterion(output_image,input_tensor[:,ei+1])
        
            decoder_input = input_tensor[:,-1,:,:] # first decoder input = last image of input sequence
            
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False 
            for di in range(target_length):
                decoder_output, decoder_hidden, output_image,_,_ = model(decoder_input)
                target = target_tensor[:,di]
                mse += criterion(output_image,target)
                if use_teacher_forcing:
                    decoder_input = target # Teacher forcing    
                else:
                    decoder_input = output_image
        
            # Moment regularization  # encoder.phycell.cell_list[0].F.conv1.weight # size (nb_filters,in_channels,7,7)
            k2m = K2M([7]).to(device)
            for b in range(0,model.phycell.cell_list[0].input_dim):
                filters = model.phycell.cell_list[0].F.conv1.weight[:,b,:] # (nb_filters,7,7)     
                m = k2m(filters.double()) 
                m  = m.float()   
                mse += criterion(m, constraints) # constrains is a precomputed matrix   
            mse.backward()
            
            return mse / target_tensor.size(1)
        
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