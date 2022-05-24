#!/usr/bin/env python3

"""
Physics-derivatives learning layer. Implementation taken and modified from
https://github.com/vincent-leguen/PhyDNet
"""

import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint_adjoint, odeint
import torch.nn.functional as F

class ConvNet1D(nn.Module):
    def __init__(self, bc, state_c, hidden=16, sigmoid=False, device=torch.device('cpu')):
        super().__init__()
        kernel_size = 3
        padding = kernel_size // 2
        self.bc = bc
        self.state_c = state_c
        self.device = device
        self.net = nn.Sequential(
            nn.Conv1d(state_c, hidden, kernel_size=kernel_size, padding=0, bias=False),
            nn.BatchNorm1d(hidden, track_running_stats=False),
            nn.Tanh(),
            nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(hidden, track_running_stats=False),
            nn.Tanh(),
            nn.Conv1d(hidden, state_c, kernel_size=kernel_size, padding=padding),
        ).to(device=self.device)
        if sigmoid:
            self.net.add_module("Sigmoid", nn.Sigmoid())

    def forward(self, t, x):
        x = torch.cat((self.bc[:1,:,:1],x,self.bc[:1,:,1:]),dim=2)
        return self.net(x)

class ConvNet2D(nn.Module):
    def __init__(self, bc, state_c, hidden=16, sigmoid=False, device=torch.device('cpu')):
        super().__init__()
        kernel_size = 3
        padding = kernel_size // 2
        self.bc = bc
        self.state_c = state_c
        self.device = device
        self.net = nn.Sequential(
            nn.Conv2d(state_c, hidden, kernel_size=kernel_size, padding=0, bias=False),
            nn.BatchNorm2d(hidden, track_running_stats=False),
            nn.Tanh(),
            nn.Conv2d(hidden, hidden, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(hidden, track_running_stats=False),
            nn.Tanh(),
            nn.Conv2d(hidden, state_c, kernel_size=kernel_size, padding=padding),
        ).to(device=self.device)
        if sigmoid:
            self.net.add_module("Sigmoid", nn.Sigmoid())

    def forward(self, t, x):
        Nx = x.size(-2)
        x = torch.cat((self.bc[:1,:,:1].repeat(1,1,Nx).unsqueeze(-1),x,
                            self.bc[:1,:,1:2].repeat(1,1,Nx).unsqueeze(-1)),dim=3)
        Ny = x.size(-1)
        x = torch.cat((self.bc[:1,:,2:3].repeat(1,1,Ny).unsqueeze(-2),x,
                            self.bc[:1,:,3:4].repeat(1,1,Ny).unsqueeze(-2)),dim=2)
        return self.net(x)


class NeuralODE(nn.Module):
    def __init__(self, convnet):
        super().__init__()
        self.convnet = convnet

    def forward(self, t, u0):
        u = odeint(self.convnet, y0=u0, t=t)
        # u: T x batch_size x n_c (x h x w)
        dim_seq = u0.dim() + 1
        dims = [1, 0, 2] + list(range(dim_seq))[3:]
        return u.permute(*dims)   # batch_size x T x n_c (x h x w)