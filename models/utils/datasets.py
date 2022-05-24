"""
This file contains the organization of the custom datasets such that it can be
read efficiently in combination with the DataLoader from PyTorch to prevent that
data reading and preparing becomes the bottleneck.

This script was inspired by
https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
"""

from torch.utils import data
import numpy as np
import os
import glob


class DatasetBurger(data.Dataset):
    """
    The dataset class which can be used with PyTorch's DataLoader.
    """

    def __init__(self, root_path, dataset_type, dataset_name, mode):
        """
        Constructor class setting up the data loader
        :param root_path: The root path of the project
        :param dataset_type: The type of the dataset (e.g. "burgers")
        :param dataset_name: The name of the dataset (e.g. "d0.01_pi")
        :param mode: Any of "train", "val" or "test"
        """

        # Determine the path to the data and the file paths to all samples
        data_root_path = os.path.join(
            root_path, dataset_type, dataset_name, mode
        )
        self.data_paths = np.sort(
            glob.glob(os.path.join(data_root_path, "*.npy"))
        )

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        """
        Generates a sample batch in the form [batch_size, dim], where dim is
        the number of features, here (t, x).
        :param index: The index of the sample in the path array
        :return: One batch of data as np.array
        """
        return np.load(self.data_paths[index])


class DatasetDiffSorp(data.Dataset):
    """
    The dataset class which can be used with PyTorch's DataLoader.
    """

    def __init__(self, root_path, dataset_type, dataset_name, mode):
        """
        Constructor class setting up the data loader
        :param root_path: The root path of the project
        :param dataset_type: The type of the dataset (e.g. "burgers")
        :param dataset_name: The name of the dataset (e.g. "d0.01_pi")
        :param mode: Any of "train", "val" or "test"
        """

        # Determine the path to the data and the file paths to all samples
        data_root_path = os.path.join(
            root_path, dataset_type, dataset_name, mode
        )
        self.data_paths = np.sort(
            glob.glob(os.path.join(data_root_path, "*.npy"))
        )

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        """
        Generates a sample batch in the form [batch_size, dim], where dim is
        the number of features, here (t, x).
        :param index: The index of the sample in the path array
        :return: One batch of data as np.array
        """
        return np.load(self.data_paths[index])


class DatasetDiffReact(data.Dataset):
    """
    The dataset class which can be used with PyTorch's DataLoader.
    """

    def __init__(self, root_path, dataset_type, dataset_name, mode):
        """
        Constructor class setting up the data loader
        :param root_path: The root path of the project
        :param dataset_type: The type of the dataset (e.g. "burgers")
        :param dataset_name: The name of the dataset (e.g. "d0.01_pi")
        :param mode: Any of "train", "val" or "test"
        """

        # Determine the path to the data and the file paths to all samples
        data_root_path = os.path.join(
            root_path, dataset_type, dataset_name, mode
        )
        self.data_paths = np.sort(
            glob.glob(os.path.join(data_root_path, "*.npy"))
        )        

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        """
        Generates a sample batch in the form [batch_size, dim], where dim is
        the number of features, here (t, x).
        :param index: The index of the sample in the path array
        :return: One batch of data as np.array
        """
        return np.load(self.data_paths[index])
    
class DatasetAllenCahn(data.Dataset):
    """
    The dataset class which can be used with PyTorch's DataLoader.
    """

    def __init__(self, root_path, dataset_type, dataset_name, mode):
        """
        Constructor class setting up the data loader
        :param root_path: The root path of the project
        :param dataset_type: The type of the dataset (e.g. "burgers")
        :param dataset_name: The name of the dataset (e.g. "d0.01_pi")
        :param mode: Any of "train", "val" or "test"
        """

        # Determine the path to the data and the file paths to all samples
        data_root_path = os.path.join(
            root_path, dataset_type, dataset_name, mode
        )
        self.data_paths = np.sort(
            glob.glob(os.path.join(data_root_path, "*.npy"))
        )

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        """
        Generates a sample batch in the form [batch_size, dim], where dim is
        the number of features, here (t, x).
        :param index: The index of the sample in the path array
        :return: One batch of data as np.array
        """
        return np.load(self.data_paths[index])


class DatasetBurger2D(data.Dataset):
    """
    The dataset class which can be used with PyTorch's DataLoader.
    """

    def __init__(self, root_path, dataset_type, dataset_name, mode):
        """
        Constructor class setting up the data loader
        :param root_path: The root path of the project
        :param dataset_type: The type of the dataset (e.g. "burgers")
        :param dataset_name: The name of the dataset (e.g. "d0.01_pi")
        :param mode: Any of "train", "val" or "test"
        """

        # Determine the path to the data and the file paths to all samples
        data_root_path = os.path.join(
            root_path, dataset_type, dataset_name, mode
        )
        self.data_paths = np.sort(
            glob.glob(os.path.join(data_root_path, "*.npy"))
        )        

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        """
        Generates a sample batch in the form [batch_size, dim], where dim is
        the number of features, here (t, x).
        :param index: The index of the sample in the path array
        :return: One batch of data as np.array
        """
        return np.load(self.data_paths[index])
