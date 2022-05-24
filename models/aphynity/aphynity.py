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
from collections import OrderedDict

class BurgersParamPDE(nn.Module):

    def __init__(self, dx, bc, device):
        super().__init__()

        self._dx     = dx
        self.bc      = bc
        self.device  = device
        
        self._laplacian = nn.Parameter(torch.tensor(
            [
                [ 1, -2,  1]
            ],
        ).float().to(device=self.device).view(1, 1, 3) / (self._dx * self._dx),
            requires_grad=False)
        
        self._adv_left = nn.Parameter(torch.tensor(
            [
                [-1,  1,  0]
            ],
        ).float().to(device=self.device).view(1, 1, 3) / (self._dx),
            requires_grad=False)
        
        self._adv_right = nn.Parameter(torch.tensor(
            [
                [ 0, -1,  1]
            ],
        ).float().to(device=self.device).view(1, 1, 3) / (self._dx),
            requires_grad=False)
        
        # Set the retardation function
        self.params_org = nn.ParameterDict({
            'd_org': nn.Parameter(torch.tensor(-2.))
        })

        self.params = OrderedDict()

    def forward(self, state):
        # state = [batch_size=1, num_channels, Nx, Ny]
        u = state
        
        self.params['d'] = torch.sigmoid(self.params_org['d_org']) * 1e-2
        
        U_ = torch.cat((self.bc[:1,:,:1],u,self.bc[:1,:,1:]),dim=2)
        Delta_u = F.conv1d(U_, self._laplacian)
        
        u_plus = torch.maximum(u,torch.zeros(1).to(device=self.device))
        u_min = torch.minimum(u,torch.zeros(1).to(device=self.device))
        
        Adv = u_plus*F.conv1d(U_, self._adv_left) \
            + u_min*F.conv1d(U_, self._adv_right) \
        
        (d,) = list(self.params.values())
        
        dUdt = d * Delta_u - Adv
        
        return dUdt

class DiffSorpParamPDE(nn.Module):

    def __init__(self, dx, bc, config, device):
        super().__init__()

        self._dx     = dx
        self.bc      = bc
        self.device  = device
        self.config = config
        
        self._laplacian = nn.Parameter(torch.tensor(
            [
                [ 1, -2,  1]
            ],
        ).float().to(device=self.device).view(1, 1, 3) / (self._dx * self._dx),
            requires_grad=False)
        
        # Set the retardation function
        if "linear" in self.config.data.name:
            self.retardation_function = self.retardation_linear
            self.params_org = nn.ParameterDict({
                'd_org': nn.Parameter(torch.tensor(-2.)),
                'por_org': nn.Parameter(torch.tensor(-2.)),
                'rho_org': nn.Parameter(torch.tensor(-2.)),
                'kd_org': nn.Parameter(torch.tensor(-2.))
            })
        elif "freundlich" in self.config.data.name:
            self.retardation_function = self.retardation_freundlich
            self.params_org = nn.ParameterDict({
                'd_org': nn.Parameter(torch.tensor(-2.)),
                'por_org': nn.Parameter(torch.tensor(-2.)),
                'rho_org': nn.Parameter(torch.tensor(-2.)),
                'kf_org': nn.Parameter(torch.tensor(-2.)),
                'nf_org': nn.Parameter(torch.tensor(-2.))
            })
        elif "langmuir" in self.config.data.name:
            self.retardation_function = self.retardation_langmuir
            self.params_org = nn.ParameterDict({
                'd_org': nn.Parameter(torch.tensor(-2.)),
                'por_org': nn.Parameter(torch.tensor(-2.)),
                'rho_org': nn.Parameter(torch.tensor(-2.)),
                'smax_org': nn.Parameter(torch.tensor(-2.)),
                'kl_org': nn.Parameter(torch.tensor(-2.))
            })

        self.params = OrderedDict()
        
    def retardation_linear(self, u):
        """
        Linear retardation factor function.
        :param u: The simulation field
        :return: The linearly computed retardation factor
        """
        return 1 + ((1 - self.por)/self.por)*self.rho*self.kd

    def retardation_freundlich(self, u):
        """
        Freundlich redardation factor function.
        :param u: The simulation field
        :return: The Freundlich-based retardation factor
        """
        return 1 + ((1 - self.por)/self.por)*self.rho*self.kf*self.nf*\
                    (u + 1e-6)**(self.nf - 1)

    def retardation_langmuir(self, u):
        """
        Langmuir retardation factor function.
        :param u: The simulation field
        :return: The Langmuir-based retardation factor
        """
        return 1 + ((1 - self.por)/self.por)*self.rho*\
                    ((self.smax*self.kl)/(u + self.kl)**2)

    def forward(self, state):
        # state = [batch_size=1, num_channels, Nx, Ny]
        C = state[:,:1]
        Ct = state[:,1:]
        
        self.params['d'] = torch.sigmoid(self.params_org['d_org']) * 1e-2
        self.params['por'] = torch.sigmoid(self.params_org['por_org'])
        self.params['rho'] = torch.sigmoid(self.params_org['rho_org']) * 1e4
        
        C_ = torch.cat((self.bc[:1,:1,:1],C,self.bc[:1,:1,1:]),dim=2)
        Delta_c = F.conv1d(C_, self._laplacian)
        
        if "linear" in self.config.data.name:
            self.params['kd'] = torch.sigmoid(self.params_org['kd_org']) * 1e-2
            (self.d, self.por, self.rho, self.kd) = list(self.params.values())
        elif "freundlich" in self.config.data.name:
            self.params['kf'] = torch.sigmoid(self.params_org['kf_org']) * 1e-2
            self.params['nf'] = torch.sigmoid(self.params_org['nf_org']) * 1e-2
            (self.d, self.por, self.rho, self.kf, self.nf) = list(self.params.values())
        elif "langmuir" in self.config.data.name:
            self.params['smax'] = torch.sigmoid(self.params_org['smax_org']) * 1e-2
            self.params['kl'] = torch.sigmoid(self.params_org['kl_org']) * 1e-2
            (self.d, self.por, self.rho, self.smax, self.kl) = list(self.params.values())
            
        retardation = self.retardation_function(C)
        
        dUdt = self.d / retardation * Delta_c
        dVdt = self.d * self.por / (self.rho/1000) * Delta_c
        
        return torch.cat([dUdt, dVdt], dim=1)


class DiffReactParamPDE(nn.Module):

    def __init__(self, dx, bc, device):
        super().__init__()

        self._dx     = dx
        self._dy     = dx
        self.bc      = bc
        self.device  = device
        
        self._laplacian = nn.Parameter(torch.tensor(
            [
                [ 0,  1,  0],
                [ 1, -4,  1],
                [ 0,  1,  0],
            ],
        ).float().to(device=self.device).view(1, 1, 3, 3) / (self._dx * self._dx),
            requires_grad=False)
        
        self.params_org = nn.ParameterDict({
            'a_org': nn.Parameter(torch.tensor(-2.)), 
            'b_org': nn.Parameter(torch.tensor(-2.)),
            'k_org': nn.Parameter(torch.tensor(-2.)),
        })

        self.params = OrderedDict()

    def forward(self, state):
        # state = [batch_size=1, num_channels, Nx, Ny]
        U = state[:,:1]
        V = state[:,1:]

        self.params['a'] = torch.sigmoid(self.params_org['a_org']) * 1e-2
        self.params['b'] = torch.sigmoid(self.params_org['b_org']) * 1e-2
        
        Nx = U.size(-2)
        U_ = torch.cat((self.bc[:1,:1,:1].repeat(1,1,Nx).unsqueeze(-1),U,
                           self.bc[:1,:1,1:2].repeat(1,1,Nx).unsqueeze(-1)),dim=3)
        V_ = torch.cat((self.bc[:1,1:,:1].repeat(1,1,Nx).unsqueeze(-1),V,
                           self.bc[:1,1:,1:2].repeat(1,1,Nx).unsqueeze(-1)),dim=3)
        Ny = U_.size(-1)
        U_ = torch.cat((self.bc[:1,:1,2:3].repeat(1,1,Ny).unsqueeze(-2),U_,
                           self.bc[:1,:1,3:4].repeat(1,1,Ny).unsqueeze(-2)),dim=2)
        V_ = torch.cat((self.bc[:1,1:,2:3].repeat(1,1,Ny).unsqueeze(-2),V_,
                           self.bc[:1,1:,3:4].repeat(1,1,Ny).unsqueeze(-2)),dim=2)
                
        Delta_u = F.conv2d(U_, self._laplacian)
        Delta_v = F.conv2d(V_, self._laplacian)

        self.params['k'] = torch.sigmoid(self.params_org['k_org']) * 1e-2
        (a, b, k) = list(self.params.values())
        dUdt = a * Delta_u + U - U.pow(3) - V - k
        dVdt = b * Delta_v + U - V
        
        # dUdt = a * Delta_u
        # dVdt = b * Delta_v
        
        return torch.cat([dUdt, dVdt], dim=1)
    

class AllenCahnParamPDE(nn.Module):

    def __init__(self, dx, bc, device):
        super().__init__()

        self._dx     = dx
        self.bc      = bc
        self.device  = device
        
        self._laplacian = nn.Parameter(torch.tensor(
            [
                [ 1, -2,  1]
            ],
        ).float().to(device=self.device).view(1, 1, 3) / (self._dx * self._dx),
            requires_grad=False)
        
        # Set the retardation function
        self.params_org = nn.ParameterDict({
            'd_org': nn.Parameter(torch.tensor(-2.))
        })

        self.params = OrderedDict()

    def forward(self, state):
        # state = [batch_size=1, num_channels, Nx, Ny]
        u = state
        
        self.params['d'] = torch.sigmoid(self.params_org['d_org']) * 1e-2
        
        U_ = F.pad(u, pad=(1,1), mode='circular')
        Delta_u = F.conv1d(U_, self._laplacian)
        
        (d,) = list(self.params.values())
        
        dUdt = d * Delta_u + 5*u - 5*u**3
        
        return dUdt
    

class Burgers2DParamPDE(nn.Module):

    def __init__(self, dx, bc, device):
        super().__init__()

        self._dx     = dx
        self._dy     = dx
        self.bc      = bc
        self.device  = device
        
        self._laplacian = nn.Parameter(torch.tensor(
            [
                [ 0,  1,  0],
                [ 1, -4,  1],
                [ 0,  1,  0],
            ],
        ).float().to(device=self.device).view(1, 1, 3, 3) / (self._dx * self._dx),
            requires_grad=False)
        
        self._adv_x_left = nn.Parameter(torch.tensor(
            [
                [ 0,  0,  0],
                [-1,  1,  0],
                [ 0,  0,  0],
            ],
        ).float().to(device=self.device).view(1, 1, 3, 3) / (self._dx),
            requires_grad=False)
        
        self._adv_x_right = nn.Parameter(torch.tensor(
            [
                [ 0,  0,  0],
                [ 0, -1,  1],
                [ 0,  0,  0],
            ],
        ).float().to(device=self.device).view(1, 1, 3, 3) / (self._dx),
            requires_grad=False)
        
        self._adv_y_bottom = nn.Parameter(torch.tensor(
            [
                [ 0, -1,  0],
                [ 0,  1,  0],
                [ 0,  0,  0],
            ],
        ).float().to(device=self.device).view(1, 1, 3, 3) / (self._dy),
            requires_grad=False)
        
        self._adv_y_top = nn.Parameter(torch.tensor(
            [
                [ 0,  0,  0],
                [ 0, -1,  0],
                [ 0,  1,  0],
            ],
        ).float().to(device=self.device).view(1, 1, 3, 3) / (self._dy),
            requires_grad=False)
        
        self.params_org = nn.ParameterDict({
            'd_org': nn.Parameter(torch.tensor(-2.))
        })

        self.params = OrderedDict()

    def forward(self, state):
        # state = [batch_size=1, num_channels, Nx, Ny]
        self.params['d'] = torch.sigmoid(self.params_org['d_org']) * 1e-2
        
        Nx = state.size(-2)
        state = torch.cat((self.bc[:1,:,:1].repeat(1,1,Nx).unsqueeze(-1),state,
                           self.bc[:1,:,1:2].repeat(1,1,Nx).unsqueeze(-1)),dim=3)
        Ny = state.size(-1)
        state = torch.cat((self.bc[:1,:,2:3].repeat(1,1,Ny).unsqueeze(-2),state,
                           self.bc[:1,:,3:4].repeat(1,1,Ny).unsqueeze(-2)),dim=2)
                
        Delta_u = F.conv2d(state, self._laplacian)
        
        u_plus = torch.maximum(state[:,:,1:-1,1:-1],torch.zeros(1).to(device=self.device))
        u_min = torch.minimum(state[:,:,1:-1,1:-1],torch.zeros(1).to(device=self.device))
        
        Adv = u_plus*F.conv2d(state, self._adv_x_left) \
            + u_min*F.conv2d(state, self._adv_x_right) \
            + u_plus*F.conv2d(state, self._adv_y_bottom) \
            + u_min*F.conv2d(state, self._adv_y_top) \

        (d, ) = list(self.params.values())
        
        dUdt = d * Delta_u - Adv
        # dUdt = d * Delta_u

        return dUdt
    

class ConvNet2DEstimator(nn.Module):
    def __init__(self, bc, state_c, hidden=16, sigmoid=False, device=torch.device('cpu')):
        super().__init__()
        kernel_size = 3
        padding = kernel_size // 2
        self.bc = bc
        self.state_c = state_c
        self.device = device
        self.net = nn.Sequential(
            nn.Conv2d(state_c, hidden, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(hidden, track_running_stats=False),
            nn.ReLU(True),
            nn.Conv2d(hidden, hidden, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(hidden, track_running_stats=False),
            nn.ReLU(True),
            nn.Conv2d(hidden, state_c, kernel_size=kernel_size, padding=padding),
        ).to(device=self.device)
        if sigmoid:
            self.net.add_module("Sigmoid", nn.Sigmoid())

    def forward(self, x):
        x = self.net(x)
        return x

    def get_derivatives(self, x):
        batch_size, T, nc, h, w = x.shape
        x = x.view(batch_size * T, nc, h, w).contiguous()
        x = self.forward(x)
        x = x.view(batch_size, T, self.state_c, h, w).contiguous()
        return x
    
    
class ConvNet1DEstimator(nn.Module):
    def __init__(self, bc, state_c, hidden=16, sigmoid=False, device=torch.device('cpu')):
        super().__init__()
        kernel_size = 3
        padding = kernel_size // 2
        self.bc = bc
        self.state_c = state_c
        self.device = device
        self.net = nn.Sequential(
            nn.Conv1d(state_c, hidden, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(hidden, track_running_stats=False),
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(hidden, track_running_stats=False),
            nn.ReLU(True),
            nn.Conv1d(hidden, state_c, kernel_size=kernel_size, padding=padding),
        ).to(device=self.device)
        if sigmoid:
            self.net.add_module("Sigmoid", nn.Sigmoid())

    def forward(self, x):
        x = self.net(x)
        return x * 1e-4

    def get_derivatives(self, x):
        batch_size, T, nc, h = x.shape
        x = x.view(batch_size * T, nc, h).contiguous()
        x = self.forward(x)
        x = x.view(batch_size, T, self.state_c, h).contiguous()
        return x


class DerivativeEstimator(nn.Module):
    def __init__(self, model_phy, model_aug):
        super().__init__()
        self.model_phy = model_phy
        self.model_aug = model_aug

    def forward(self, t, state):
        res_phy = self.model_phy(state)
        res_aug = self.model_aug(state)
        
        return res_phy + res_aug

class Forecaster(nn.Module):
    def __init__(self, model_phy, model_aug, method='rk4', options=None):
        super().__init__()

        self.model_phy = model_phy
        self.model_aug = model_aug

        self.derivative_estimator = DerivativeEstimator(self.model_phy, self.model_aug)
        self.method = method
        self.options = options
        self.int_ = odeint 
        
    def forward(self, y0, t):
        # y0 = y[:,0]
        res = self.int_(self.derivative_estimator, y0=y0, t=t, method=self.method, options=self.options)
        # res: T x batch_size x n_c (x h x w)
        dim_seq = y0.dim() + 1
        dims = [1, 0, 2] + list(range(dim_seq))[3:]
        return res.permute(*dims)   # batch_size x T x n_c (x h x w)
    
    def get_pde_params(self):
        return self.model_phy.params