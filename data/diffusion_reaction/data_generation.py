"""
This script creates a certain amount of data for a diffusion reaction problem.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from simulator import Simulator


##############
# PARAMETERS #
##############

TRAIN_DATA = True

#
# Diffusion sorption specifications
DIFFUSION_COEFFICIENT_U = 1E-3
DIFFUSION_COEFFICIENT_V = 5E-3
K = 5E-3

#
# Simulation
if TRAIN_DATA:
    T_MAX = 50
    T_STEPS = 501
else:
    T_MAX = 7.5
    T_STEPS = 101

X_LEFT = -1.0
X_RIGHT = 1.0
X_STEPS = 49
Y_BOTTOM = -1.0
Y_TOP = 1.0
Y_STEPS = 49

#
# Dataset
DATAPOINTS_INITIAL = 50
DATAPOINTS_BOUNDARY = 200
DATAPOINTS_COLLOCATION = 20000
DATASET_NAME = "data"
SAVE_DATA = True
VISUALIZE_DATA = False


#############
# FUNCTIONS #
#############

def generate_sample(simulator, visualize, save_data, root_path):
    """
    This function generates a data sample, visualizes it if desired and saves
    the data to file if desired.
    :param simulator: The simulator object for data creation
    :param visualize: Boolean indicating whether to visualize the data
    :param save_data: Boolean indicating whether to write the data to file
    :param root_path: The root path of this script
    """

    
    print("Generating data...")
    # Generate a data sample
    sample_u, sample_v = simulator.generate_sample()
    
    if TRAIN_DATA:
        
        # Randomly draw indices for initial, boundary and collocation points
        idcs_init, idcs_bound = draw_indices(
            simulator=simulator,
            n_init=DATAPOINTS_INITIAL,
            n_bound=DATAPOINTS_BOUNDARY,
            n_colloc=DATAPOINTS_COLLOCATION
        )
        
    
        # If specified, visualize the sample
        if visualize:
            visualize_sample(sample_u=sample_u,
                             sample_v=sample_v,
                             simulator=simulator)
    
        # If specified, write the sample to file
        if save_data:
            
            write_data_to_file(
                root_path=root_path,
                simulator=simulator,
                sample_u=sample_u,
                sample_v=sample_v
            )
        
            # List for tuples as train/val/test data
            data_tuples = []
                
            # Concatenate all indices and add their data tuples to the list
            all_idcs = np.concatenate((idcs_init, idcs_bound), axis=0)
            for pair in all_idcs:
                data_tuples.append(create_data_tuple_init_bound(
                    sample_u=sample_u,
                    sample_v=sample_v,
                    pair=pair,
                    simulator=simulator
                ))
            data_tuples.extend(np.transpose(create_data_tuple_colloc(
                sample_u=sample_u, sample_v=sample_v, simulator=simulator
            )))
        
            write_tuples_to_file(root_path, data_tuples, mode="train")

        # Training data (validation)
        
        # Randomly draw indices for initial, boundary and collocation points
        idcs_init, idcs_bound = draw_indices(
            simulator=simulator,
            n_init=DATAPOINTS_INITIAL,
            n_bound=DATAPOINTS_BOUNDARY,
            n_colloc=DATAPOINTS_COLLOCATION
        )
        
        # If specified, write the sample to file
        if save_data:
            
            # List for tuples as train/val/test data
            data_tuples = []
                
            # Concatenate all indices and add their data tuples to the list
            all_idcs = np.concatenate((idcs_init, idcs_bound), axis=0)
            for pair in all_idcs:
                data_tuples.append(create_data_tuple_init_bound(
                    sample_u=sample_u,
                    sample_v=sample_v,
                    pair=pair,
                    simulator=simulator
                ))
                
            data_tuples.extend(np.transpose(create_data_tuple_colloc(
                sample_u=sample_u, sample_v=sample_v, simulator=simulator
            )))
        
            write_tuples_to_file(root_path, data_tuples, mode="val")
        
    else:
        
        # If specified, write the sample to file
        if save_data:
        
            write_data_to_file(
                root_path=root_path,
                simulator=simulator,
                sample_u=sample_u,
                sample_v=sample_v
            )

    # If specified, visualize the sample
    if visualize:
        visualize_sample(sample_u=sample_u,
                         sample_v=sample_v,
                         simulator=simulator)


def draw_indices(simulator, n_init, n_bound, n_colloc):
    """
    Randomly chooses a specified number of points from the spatiotemporal
    sample for the initial and boundary conditions as well as collocation
    points.
    :param simulator: The simulator that created the sample
    :param n_init: Number of initial points at t=0
    :param n_bound: Number of boundary points at x_left, x_right, y_top and y_bottom
    :param n_colloc: Number of collocation points
    :return: The three-dimensional index arrays(t, x, y)
    """

    rng = np.random.default_rng()

    idcs_init = np.zeros((n_init, 3), dtype=np.int)
    idcs_init[:, 0] = 0
    idcs_init[:, 1] = rng.choice(len(simulator.x),
                                  size=n_init)
    idcs_init[:, 2] = rng.choice(len(simulator.y),
                                  size=n_init)

    idcs_bound = np.zeros((n_bound, 3), dtype=np.int)
    idcs_bound[:n_bound//4, 0] = rng.choice(len(simulator.t)//5 + 1,
                                  size=n_bound//4)
    idcs_bound[:n_bound//4, 1] = 0
    idcs_bound[:n_bound//4, 2] = rng.choice(len(simulator.y),
                                  size=n_bound//4)
    
    idcs_bound[n_bound//4: 2*n_bound//4, 0] = rng.choice(len(simulator.t)//5 + 1,
                                  size=2*n_bound//4 - n_bound//4)
    idcs_bound[n_bound//4: 2*n_bound//4, 1] = len(simulator.x) - 1
    idcs_bound[n_bound//4: 2*n_bound//4, 2] = rng.choice(len(simulator.y),
                                  size=2*n_bound//4 - n_bound//4)
    
    idcs_bound[2*n_bound//4: 3*n_bound//4, 0] = rng.choice(len(simulator.t)//5 + 1,
                                  size=3*n_bound//4 - 2*n_bound//4)
    idcs_bound[2*n_bound//4: 3*n_bound//4, 1] = rng.choice(len(simulator.x),
                                  size=3*n_bound//4 - 2*n_bound//4)
    idcs_bound[2*n_bound//4: 3*n_bound//4, 2] = 0
    
    idcs_bound[3*n_bound//4:, 0] = rng.choice(len(simulator.t)//5 + 1,
                                  size=n_bound - 3*n_bound//4)
    idcs_bound[3*n_bound//4:, 1] = rng.choice(len(simulator.x),
                                  size=n_bound - 3*n_bound//4)
    idcs_bound[3*n_bound//4:, 1] = len(simulator.y) - 1

    return idcs_init, idcs_bound


def write_data_to_file(root_path, simulator, sample_u, sample_v):
    """
    Writes the given data to the according directory in .npy format.
    :param root_path: The root_path of the script
    :param simulator: The simulator that created the data
    :param sample_u: The sample to be written to file (activator)
    :param sample_v: The sample to be written to file (inhibitor)
    """
    
    if TRAIN_DATA:
    
        # Create the data directory for the training data if it does not yet exist
        data_path = os.path.join(root_path, DATASET_NAME+"_train")
        os.makedirs(data_path, exist_ok=True)
        
        # Write the t- and x-series data along with the sample to file
        np.save(file=os.path.join(data_path, "t_series.npy"),
                arr=simulator.t[:len(simulator.t)//5 + 1])
        np.save(file=os.path.join(data_path, "x_series.npy"), arr=simulator.x)
        np.save(file=os.path.join(data_path, "y_series.npy"), arr=simulator.y)
        np.save(file=os.path.join(data_path, "sample_u.npy"),
                arr=sample_u[:len(simulator.t)//5 + 1])
        np.save(file=os.path.join(data_path, "sample_v.npy"),
                arr=sample_v[:len(simulator.t)//5 + 1])
            
        # Create the data directory for the extrapolation data if it does not yet exist
        data_path = os.path.join(root_path, DATASET_NAME+"_ext")
        os.makedirs(data_path, exist_ok=True)
        
        # Write the t- and x-series data along with the sample to file
        np.save(file=os.path.join(data_path, "t_series.npy"), arr=simulator.t)
        np.save(file=os.path.join(data_path, "x_series.npy"), arr=simulator.x)
        np.save(file=os.path.join(data_path, "y_series.npy"), arr=simulator.y)
        np.save(file=os.path.join(data_path, "sample_u.npy"), arr=sample_u)
        np.save(file=os.path.join(data_path, "sample_v.npy"), arr=sample_v)
    
    else:

        # Create the data directory if it does not yet exist
        data_path = os.path.join(root_path, DATASET_NAME+"_test")
        os.makedirs(data_path, exist_ok=True)
    
        # Write the t- and x-series data along with the sample to file
        np.save(file=os.path.join(data_path, "t_series.npy"), arr=simulator.t)
        np.save(file=os.path.join(data_path, "x_series.npy"), arr=simulator.x)
        np.save(file=os.path.join(data_path, "y_series.npy"), arr=simulator.y)
        np.save(file=os.path.join(data_path, "sample_u.npy"), arr=sample_u)
        np.save(file=os.path.join(data_path, "sample_v.npy"), arr=sample_v)

def write_tuples_to_file(root_path, data_tuples, mode):
    """
    Writes the data tuples to the according directory in .npy format for
    training and validation of PINN
    :param root_path: The root_path of the script
    :param data_tuples: Array of the train/val tuples
    :param mode: Any of "train" or "val"
    
    """
    
    data_path = os.path.join(root_path, DATASET_NAME+"_train")
    os.makedirs(os.path.join(data_path, mode), exist_ok=True)
    
    # Iterate over the data_tuples and write them to separate files
    for idx, data_tuple in enumerate(data_tuples):
        
        name = f"{mode}_{str(idx).zfill(5)}.npy"
        np.save(file=os.path.join(data_path, mode, name), arr=data_tuple)

def create_data_tuple_init_bound(simulator, sample_u, sample_v, pair):
    """
    Creates a tuple (t, x, y, sample_u, sample_v, t_idx, x_idx, y_idx), where t is the
    time step, x and y are the spatial coordinate, sample_u and sample_v are the 
    desired model output, and t_idx, x_idx and y_idx are the indices in the
    sample for t, x and y.
    :param simulator: The simulator that generated the sample
    :param sample_u: The data sample (activator)
    :param sample_v: The data sample (inhibitor)
    :param pair: The index pair of the current data points
    
    :return: Tuple (t, x, y, sample_u, sample_v, t_idx, x_idx, y_idx)
    """
    t_idx, x_idx, y_idx = pair
    u = sample_u[t_idx, x_idx, y_idx]
    v = sample_v[t_idx, x_idx, y_idx]
    t, x, y = simulator.t[t_idx], simulator.x[x_idx], simulator.y[y_idx]
    
    return np.array((t, x, y, u, v, t_idx, x_idx, y_idx), dtype=np.float32)

def create_data_tuple_colloc(simulator, sample_u, sample_v):
    """
    Creates a tuple (t, x, y, sample_u, sample_v, t_idx, x_idx, y_idx), where t is the
    time step, x and y are the spatial coordinate, sample_u and sample_v are the 
    desired model output, and t_idx, x_idx and y_idx are the indices in the
    sample for t, x and y.
    :param simulator: The simulator that generated the sample
    :param sample_u: The data sample (activator)
    :param sample_v: The data sample (inhibitor)
    
    :return: Tuple (t, x, y, sample_u, sample_v, t_idx, x_idx, y_idx)
    """
    t = np.arange(len(simulator.t)//2 + 1)
    x = np.arange(len(simulator.x))
    y = np.arange(len(simulator.y))
    
    t, x, y = np.meshgrid(t,x,y)
    
    pair = np.hstack((t.flatten()[:,None],x.flatten()[:,None],y.flatten()[:,None]))
    idx = np.random.choice(len(pair), DATAPOINTS_COLLOCATION , replace=False)
    
    t_idx = pair[idx,0]
    x_idx = pair[idx,1]
    y_idx = pair[idx,2]
    
    u = sample_u[t_idx, x_idx, y_idx]
    v = sample_v[t_idx, x_idx, y_idx]
    
    t, x, y = simulator.t[t_idx], simulator.x[x_idx], simulator.y[y_idx]
    
    return np.array((t, x, y, u, v, t_idx, x_idx, y_idx), dtype=np.float32)


def visualize_sample(sample_u, sample_v, simulator):
    """
    Method to visualize a single sample. Code taken and modified from
    https://github.com/maziarraissi/PINNs
    :param sample_u: The actual data sample (activator) for visualization
    :param sample_v: The actual data sample (inhibitor) for visualization
    :param simulator: The simulator used for data generation
    """

    sample_u_init = np.transpose(sample_u[0])
    sample_v_init = np.transpose(sample_v[0])

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # u(x,y) initial
    im1 = ax[0].imshow(sample_u_init, interpolation='nearest', cmap='rainbow', 
                  extent=[simulator.x.min(), simulator.x.max(),
                          simulator.y.min(), simulator.y.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax)
    im1.set_clim(sample_u.min(), sample_u.max())

    ax[0].set_xlim(simulator.x.min(), simulator.x.max())
    ax[0].set_ylim(simulator.y.min(), simulator.y.max())
    ax[0].set_xlabel('$x$')
    ax[0].set_ylabel('$y$')
    ax[0].set_title('$u(x,y)$', fontsize = 10)
    
    
    # v(x,y) initial
    im2 = ax[1].imshow(sample_v_init, interpolation='nearest', cmap='rainbow', 
                  extent=[simulator.x.min(), simulator.x.max(),
                          simulator.y.min(), simulator.y.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax=cax)
    im2.set_clim(sample_v.min(), sample_v.max())

    ax[1].set_xlim(simulator.x.min(), simulator.x.max())
    ax[1].set_ylim(simulator.y.min(), simulator.y.max())
    ax[1].set_xlabel('$x$')
    ax[1].set_ylabel('$y$')
    ax[1].set_title('$v(x,y)$', fontsize = 10)
    
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(simulator.t),
        fargs=(im1, im2, sample_u, sample_v),
        interval=20
    )
    
    plt.tight_layout()
    plt.draw()
    plt.show()


def animate(t, im1, im2, sample_u, sample_v):
    """
    Data animation function animating an image over time.
    :param t: The current time step
    :param im1: The matplotlib image object for u (activator)
    :param im2: The matplotlib image object for v (inhibitor)
    :param sample_u: The data field (activator)
    :param sample_v: The data field (inhibitor)
    :return: The matplotlib image object updated with the current time step's
        image date
    """
    im1.set_array(np.transpose(sample_u[t]))
    im2.set_array(np.transpose(sample_v[t]))


def main():
    """
    Main method used to create the datasets.
    """

    # Determine the root path for this script and set up a path for the data
    root_path = os.path.abspath("")

    # Create dataset using the parameters from the configuration file
    simulator = Simulator(
        diffusion_coefficient_u=DIFFUSION_COEFFICIENT_U,
        diffusion_coefficient_v=DIFFUSION_COEFFICIENT_V,
        k = K,
        t_max=T_MAX,
        t_steps=T_STEPS,
        x_left=X_LEFT,
        x_right=X_RIGHT,
        x_steps=X_STEPS,
        y_bottom=Y_BOTTOM,
        y_top = Y_TOP,
        y_steps=Y_STEPS,
        train_data=TRAIN_DATA
    )

    # Create train, validation and test data
    generate_sample(simulator=simulator,
                    visualize=VISUALIZE_DATA,
                    save_data=SAVE_DATA,
                    root_path=root_path
                    )


if __name__ == "__main__":
    main()

    print("Done.")
