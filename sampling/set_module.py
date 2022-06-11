# -*- coding: utf-8 -*-
"""
Created on March 2021
@author: Timothy Praditia

This script contains the Set_Module class that constructs separate objects for
different core samples, based on the parameters read from the Excel input file
"""

import torch
import numpy as np
import pandas as pd


class Set_Module:
    
    def __init__(self, filename, params):
        """
        Constructor
        
        Inputs:
            filename    : the corresponding filename for the core sample
            params      : the configuration object containing the model settings

        """

        # Load parameters from the Excel file
        in_params = pd.read_excel(filename, sheet_name=1, index_col=0, header=None)
        
        # Determine the device on which the training takes place
        self.device = params.device
        
        self.config = params.config
        
        # Soil Parameters
        self.D = in_params[1]['D']
        self.por = in_params[1]['por']
        self.rho_s = in_params[1]['rho_s']
        
        
        # Simulation Domain
        self.X = in_params[1]['X']
        self.Nx = int(in_params[1]['Nx'])
        self.dx = self.X / (self.Nx+1)
        self.T = in_params[1]['T']
        self.Nt = int(in_params[1]['Nt'])
        self.r = in_params[1]['sample_radius']
        self.A = np.pi * self.r**2
        self.Q = in_params[1]['Q']
        self.solubility = in_params[1]['solubility']
        self.cauchy_mult = self.por * self.A / self.Q
        
        ## Effective diffusion coefficient for each variable
        self.D_eff = np.array([self.D / (self.dx**2),
                               self.D * self.por / (self.rho_s/1000) / (self.dx**2)])
        