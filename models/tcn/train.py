import numpy as np
import torch as th
import torch.nn as nn
import time
import glob
import os
import matplotlib.pyplot as plt
from threading import Thread
import sys

sys.path.append("..")
from utils.configuration import Configuration
import utils.helper_functions as helpers
from tcn import TemporalConvNet


def run_training(print_progress=False, model_number=None):

    # Set a random seed for varying weight initializations
    th.seed()

    th.set_num_threads(1)

    # Load the user configurations
    config = Configuration("config.json")

    # Append the model number to the name of the model
    if model_number is None:
        model_number = config.model.number
    config.model.name = config.model.name + "_" + str(model_number).zfill(2)
    
    print("Model name:", config.model.name)

    # Hide the GPU(s) in case the user specified to use the CPU in the config
    # file
    if config.general.device == "CPU":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    time_start = time.time()

    # setting device on GPU if available, else CPU
    device = helpers.determine_device()

    # Initialize and set up the network
    model = TemporalConvNet(config=config).to(device=device)

    if print_progress:
        # Count number of trainable parameters
        pytorch_total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print("Trainable model parameters:", pytorch_total_params)

    # If desired, restore the network by loading the weights saved in the .pt
    # file
    if config.training.continue_training:
        print("Restoring model (that is the network\"s weights) from file...")
        model.load_state_dict(th.load(os.path.join(os.path.abspath(""),
                                      "checkpoints",
                                      config.model.name,
                                      config.model.name + ".pt")))
        model.train()

    #
    # Set up the optimizer and the criterion (loss)
    optimizer = th.optim.LBFGS(model.parameters(),
                                lr=config.training.learning_rate)
    criterion = nn.MSELoss(reduction="mean")


    #
    # Load data depending on the task
    if config.data.type == "burger":
        data_path = os.path.join("../../data/",
                                 config.data.type,
                                 config.data.name,
                                 "sample.npy")
        data = np.array(np.load(data_path), dtype=np.float32)
        data = np.expand_dims(data, axis=0)

    elif config.data.type == "diffusion_sorption":
        data_path_base = os.path.join("../../data/",
                                      config.data.type,
                                      config.data.name)
        data_path_c = os.path.join(data_path_base, "sample_c.npy")
        data_path_ct = os.path.join(data_path_base, "sample_ct.npy")
        data_c = np.array(np.load(data_path_c), dtype=np.float32)
        data_ct = np.array(np.load(data_path_ct), dtype=np.float32)
        data = np.stack((data_c, data_ct), axis=0)

    elif config.data.type == "diffusion_reaction":
        data_path_base = os.path.join("../../data/",
                                      config.data.type,
                                      config.data.name)
        data_path_u = os.path.join(data_path_base, "sample_u.npy")
        data_path_v = os.path.join(data_path_base, "sample_v.npy")
        data_u = np.array(np.load(data_path_u), dtype=np.float32)
        data_v = np.array(np.load(data_path_v), dtype=np.float32)
        data = np.stack((data_u, data_v), axis=0)
        
    elif config.data.type == "allen_cahn":
        data_path = os.path.join("../../data/",
                                 config.data.type,
                                 config.data.name,
                                 "sample.npy")
        data = np.array(np.load(data_path), dtype=np.float32)
        data = np.expand_dims(data, axis=0)

    elif config.data.type == "burger_2d":
        data_path = os.path.join("../../data/",
                                 config.data.type,
                                 config.data.name,
                                 "sample.npy")
        data = np.array(np.load(data_path), dtype=np.float32)
        data = np.expand_dims(data, axis=0)
    
    # Set up the training and validation datasets and -loaders
    data_train = th.tensor(
        data[:, :config.training.t_stop],
        device=device
    ).unsqueeze(0)
    data_valid = th.tensor(
        data[:, config.validation.t_start:config.validation.t_stop],
        device=device
    ).unsqueeze(0)

    #
    # Set up lists to save and store the epoch errors
    epoch_errors_train = []
    epoch_errors_valid = []
    best_train = np.infty
    best_valid = np.infty
    
    """
    TRAINING
    """

    a = time.time()

    #
    # Start the training and iterate over all epochs
    for epoch in range(config.training.epochs):

        epoch_start_time = time.time()

        # Separate the data into network inputs and labels
        net_input = data_train[:, :, :-1]
        net_label = data_train[:, :, 1:]
        
        def closure():
            # Set the model to train mode
            model.train()
                
            # Reset the optimizer to clear data from previous iterations
            optimizer.zero_grad()

            # Forward the input through the network
            net_outputs = model.forward(x=net_input)
            
            # Comput the error
            mse = criterion(net_outputs, net_label)
            
            mse.backward()
                
            return mse
        
        # Backpropagate the error and perform a weight update
        optimizer.step(closure)
            
        # Extract the MSE value from the closure function
        mse = closure()

        epoch_errors_train.append(mse.item())

        train_sign = "(-)"
        if epoch_errors_train[-1] < best_train:
            best_train = epoch_errors_train[-1]
            train_sign = "(+)"

        #
        # Validation

        # Separate the data into network inputs and labels
        net_input = data_valid[:, :, :-1]
        net_label = data_valid[:, :, 1:]

        # Forward the input through the network
        net_outputs = model.forward(x=net_input)
        
        # Comput the error
        mse = criterion(net_outputs, net_label)
        epoch_errors_valid.append(mse.item())

        # Save the model to file (if desired)
        if config.training.save_model \
            and mse.item() < best_valid:
            
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
        if print_progress:
            print(f"Epoch {str(epoch+1).zfill(int(np.log10(config.training.epochs))+1)}/{str(config.training.epochs)} took {str(np.round(time.time() - epoch_start_time, 2)).ljust(5, '0')} seconds. \t\tAverage epoch training error: {train_sign}{str(np.round(epoch_errors_train[-1], 10)).ljust(12, ' ')} \t\tValidation error: {valid_sign}{str(np.round(epoch_errors_valid[-1], 10)).ljust(12, ' ')}")

    if print_progress:
        b = time.time()
        print("\nTraining took " + str(np.round(b - a, 2)) + " seconds.\n\n")


if __name__ == "__main__":
    run_training(print_progress=True)

    print("Done.")
