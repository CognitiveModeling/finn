# -*- coding: utf-8 -*-
"""
Created on March 2021
@author: Timothy Praditia

This script contains the Initialize class that reads the configuration file
and construct an object with the corresponding configuration parameters
"""

import torch as th
import os
import shutil
import sys
sys.path.append("../models")
from utils.configuration import Configuration
import utils.helper_functions as helpers

class Initialize:
    
    def __init__(self, config):
        """
        Constructor
        
        """
        
        # Load the user configurations
        self.config = config
        
        # Append the model number to the name of the model
        model_number = self.config.model.number
        self.config.model.name = self.config.model.name + "_" + str(model_number).zfill(2)
    
        # Print some information to console
        print("Model name:", self.config.model.name)
    
        # Hide the GPU(s) in case the user specified to use the CPU in the config
        # file
        if self.config.general.device == "CPU":
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # Set device on GPU if specified in the configuration file, else CPU
        # device = helpers.determine_device()
        self.device = th.device(self.config.general.device)
        
        
        # # SET WORKING PATH
        # self.main_path = os.getcwd()
        
        # # MODEL NAME & SETTING
        # self.model_name = params.model_name
        
        # self.model_path = self.main_path + '\\' + self.model_name
        # self.check_dir(self.model_path)
        
        # self.log_path = self.main_path + '\\runs\\' + self.model_name
        # # Remove old log files to prevent unclear visualization in tensorboard
        # self.check_dir(self.log_path, remove=True)
        
        # self.save_model = params.save_model
        # self.continue_training = params.continue_training
        # self.device_name = params.device_name
        # self.device = self.determine_device()

        # # NETWORK HYPER-PARAMETERS
        # self.flux_layers = params.flux_layers
        # self.state_layers = params.state_layers
        # self.flux_nodes = params.flux_nodes
        # self.state_nodes = params.state_nodes
        # self.learning_rate = params.learning_rate
        # self.error_mult = params.error_mult
        # self.phys_mult = params.phys_mult
        # self.epochs = params.epochs
        # self.lbfgs_optim = params.lbfgs_optim
        
        # # SIMULATION-RELATED INPUTS
        # self.num_vars = params.num_vars