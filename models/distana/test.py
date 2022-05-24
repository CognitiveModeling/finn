import numpy as np
import torch as th
import time
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

sys.path.append("..")
from utils.configuration import Configuration
import utils.helper_functions as helpers
from distana import DISTANA


def run_testing(print_progress=False, visualize=False, model_number=None):

    th.set_num_threads(1)
    
    # Load the user configurations
    config = Configuration("config.json")

    # Append the model number to the name of the model
    if model_number is None:
        model_number = config.model.number
    config.model.name = config.model.name + "_" + str(model_number).zfill(2)

    # Hide the GPU(s) in case the user specified to use the CPU in the config
    # file
    if config.general.device == "CPU":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Set device on GPU if specified in the configuration file, else CPU
    device = helpers.determine_device()

    # Initialize and set up the network
    model = DISTANA(config=config, device=device).to(device=device)

    if print_progress:
        # Count number of trainable parameters
        pytorch_total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print("Trainable model parameters:", pytorch_total_params)

        # Restore the network by loading the weights saved in the .pt file
        print("Restoring model (that is the network\"s weights) from file...")

    model.load_state_dict(th.load(os.path.join(os.path.abspath(""),
                                  "checkpoints",
                                  config.model.name,
                                  config.model.name + ".pt")))
    model.eval()

    """
    TESTING
    """

    #
    # Load the data depending on the task
    if config.data.type == "burger":
        data_path = os.path.join("../../data/",
                                 config.data.type,
                                 config.data.name,
                                 "sample.npy")
        data = np.array(np.load(data_path), dtype=np.float32)
        data = np.expand_dims(data, axis=1)

    elif config.data.type == "diffusion_sorption":        
        data_path_base = os.path.join("../../data/",
                                      config.data.type,
                                      config.data.name)
        data_path_c = os.path.join(data_path_base, "sample_c.npy")
        data_path_ct = os.path.join(data_path_base, "sample_ct.npy")
        data_c = np.array(np.load(data_path_c), dtype=np.float32)
        data_ct = np.array(np.load(data_path_ct), dtype=np.float32)
        data = np.stack((data_c, data_ct), axis=1)

    elif config.data.type == "diffusion_reaction":
        data_path_base = os.path.join("../../data/",
                                      config.data.type,
                                      config.data.name)
        data_path_u = os.path.join(data_path_base, "sample_u.npy")
        data_path_v = os.path.join(data_path_base, "sample_v.npy")
        data_u = np.array(np.load(data_path_u), dtype=np.float32)
        data_v = np.array(np.load(data_path_v), dtype=np.float32)
        data = np.stack((data_u, data_v), axis=1)
        
    elif config.data.type == "allen_cahn":
        data_path = os.path.join("../../data/",
                                 config.data.type,
                                 config.data.name,
                                 "sample.npy")
        data = np.array(np.load(data_path), dtype=np.float32)
        data = np.expand_dims(data, axis=1)

    elif config.data.type == "burger_2d":
        data_path = os.path.join("../../data/",
                                 config.data.type,
                                 config.data.name,
                                 "sample.npy")
        data = np.array(np.load(data_path), dtype=np.float32)
        data = np.expand_dims(data, axis=1)

    # Set up the training and validation datasets and -loaders
    data_test = th.tensor(data,
                          device=device).unsqueeze(1)
    sequence_length = len(data_test) - 1

    # Evaluate the network for the given test data

    # Separate the data into network inputs and labels
    net_inputs = th.clone(data_test[:-1])
    net_labels = th.clone(data_test[1:])
    
    # Set up an array of zeros to store the network outputs
    net_outputs = th.zeros(size=(sequence_length,
                                 config.testing.batch_size,                                 
                                 config.model.dynamic_channels[-1],
                                 *config.model.field_size),
                           device=device)
    state_list = None

    # Iterate over the remaining sequence of the training example and perform a
    # forward pass
    time_start = time.time()
    for t in range(len(net_inputs)):

        if t < config.testing.teacher_forcing_steps:
            # Teacher forcing
            net_input = net_inputs[t]
        else:
            # Closed loop
            net_input = net_outputs[t - 1]

        # Feed the boundary data also in closed loop if desired
        if config.testing.feed_boundary_data:
            net_input[:, :, 0] = net_inputs[t, :, :, 0]
            net_input[:, :, -1] = net_inputs[t, :, :, -1]

        net_output, state_list = model.forward(input_tensor=net_input,
                                               cur_state_list=state_list)
        net_outputs[t] = net_output[-1]

    if print_progress:
        forward_pass_duration = time.time() - time_start
        print("Forward pass took:", forward_pass_duration, "seconds.")

    # Convert the PyTorch network output tensor into a numpy array
    net_outputs = net_outputs.cpu().detach().numpy()[:, 0, 0]
    net_labels = net_labels.cpu().detach().numpy()[:, 0, 0]

    #
    # Visualize the data
    if visualize:
        plt.style.use("dark_background")

        # Plot over space and time
        fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        if config.data.type == "burger" or\
           config.data.type == "diffusion_sorption" or\
           config.data.type == "allen_cahn":
            
            im1 = ax[0].imshow(
                np.transpose(net_labels), interpolation='nearest',
                cmap='rainbow', origin='lower', aspect='auto', vmin=-0.4,
                vmax=0.4
            )
            fig.colorbar(im1, ax=ax[0])
            im2 = ax[1].imshow(
                np.transpose(net_outputs), interpolation='nearest',
                cmap='rainbow', origin='lower', aspect='auto', vmin=-0.4,
                vmax=0.4
            )
            fig.colorbar(im2, ax=ax[1])

            ax[0].set_xlabel("t")
            ax[0].set_ylabel("x")
            ax[1].set_xlabel("t")

        elif config.data.type == "diffusion_reaction":
            
            im1 = ax[0].imshow(
                np.transpose(net_labels[..., 0]), interpolation='nearest',
                cmap='rainbow', origin='lower', aspect='auto', vmin=-0.4,
                vmax=0.4
            )
            fig.colorbar(im1, ax=ax[0])
            im2 = ax[1].imshow(
                np.transpose(net_outputs[..., 0]), interpolation='nearest',
                cmap='rainbow', origin='lower', aspect='auto', vmin=-0.4,
                vmax=0.4
            )
            fig.colorbar(im2, ax=ax[1])

            ax[0].set_xlabel("x")
            ax[0].set_ylabel("y")
            ax[1].set_xlabel("x")

        elif config.data.type == "burger_2d":
            
            im1 = ax[0].imshow(
                np.transpose(net_labels[-1, ...]), interpolation='nearest',
                cmap='rainbow', origin='lower', aspect='auto', vmin=-0.4,
                vmax=0.4
            )
            fig.colorbar(im1, ax=ax[0])
            im2 = ax[1].imshow(
                np.transpose(net_outputs[-1, ...]), interpolation='nearest',
                cmap='rainbow', origin='lower', aspect='auto', vmin=-0.4,
                vmax=0.4
            )
            fig.colorbar(im2, ax=ax[1])

            ax[0].set_xlabel("x")
            ax[0].set_ylabel("y")
            ax[1].set_xlabel("x")


        ax[0].set_title("Ground Truth")
        ax[1].set_title("Network Output")


        if config.data.type == "diffusion_reaction"\
            or config.data.type == "burger_2d":
            anim = animation.FuncAnimation(
                fig,
                animate,
                frames=sequence_length,
                fargs=(im1, im2, net_labels, net_outputs),
                interval=20
            )

        plt.show()

    #
    # Compute and return statistics
    mse = np.mean(np.square(net_outputs - net_labels))

    return net_outputs, net_labels


def animate(t, im1, im2, net_labels, net_outputs):
    """
    Data animation function animating an image over time.
    :param t: The current time step
    :param axis: The matplotlib image object
    :param field: The data field
    :return: The matplotlib image object updated with the current time step's
        image date
    """
    im1.set_array(net_labels[t])
    im2.set_array(net_outputs[t])


if __name__ == "__main__":
    mse = run_testing(print_progress=True, visualize=True)

    print(f"MSE: {mse}")
    exit()
    
    u_hat = []
    
    for i in range(10):
        pred, u = run_testing(visualize=False, model_number=i)
        pred = np.expand_dims(pred,-1)
        u_hat.append(pred)
        
    u_hat = np.stack(u_hat)
    
    np.save("distana_test.npy", u_hat)

    print("Done.")