# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 08:56:57 2021

@author: timot
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.distributions import Normal
from torchdiffeq import odeint
import pickle
import os

torch.set_default_dtype(torch.float32)

class Sampler:
    def __init__(self, model, model_temp, step_size, noise_tol, cfg, data,
                 update_freq=10, save_freq=50, continue_sampling=False):
        
        self.model = model
        self.model_temp = model_temp
        
        trained_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        
        self.param_size = trained_params
        
        self.prior = torch.distributions.Normal(trained_params,
                                                torch.ones_like(trained_params)*0.05)
        
        self.step_size = step_size
        
        self.noise_tol = noise_tol
        
        self.noise = torch.distributions.Normal(0, torch.ones_like(trained_params))
        
        self.update_freq = update_freq
        
        self.save_freq = save_freq
        
        self.continue_sampling = continue_sampling
        
        self.cfg = cfg
        
        self.data = data
        
        self.inp_func = torch.linspace(0,2,501).unsqueeze(-1).to(device=self.cfg.device)
        
        self.save_path = os.path.abspath("") + "/" + self.cfg.config.sampling.name + ".pickle"
        
    def log_prior(self, flat_params):
        # log prior
        log_prior = self.prior.log_prob(flat_params).mean()
        return log_prior
    
    def model_out(self, model, u, t):
        output = model(t=t, u=u).to(device=self.cfg.device)
        
        # Calculate predicted breakthrough curve
        cauchy_mult = self.cfg.cauchy_mult * self.cfg.D_eff[0] * self.cfg.dx
        pred = ((output[:,-2,0] - output[:,-1,0]) * cauchy_mult).squeeze()
        
        return pred
    
    def func_nn(self, model):
        ret = (1/model.func_nn(self.inp_func)).squeeze()
        
        return ret
    
    def loss(self, model, output, ret, kl_weight=0.01):
        """Method for calculating the negative expected lower bound
        Args:
            y0 (torch.Tensor): initial condition of the differential
                equation.
            t (torch.Tensor): timesteps where we want to perform the
                solution with the ode solver.
            y (torch.Tensor): true trajectory(target) of the differential
                equation.
            samples (int): for how many samples to perform the monte carlo
                approximation of the ELBO
        Returns:
            torch.Tensor: Scalar of the ELBO loss
        """
        
        log_prior = self.log_prior(torch.nn.utils.parameters_to_vector(model.parameters()))
        log_like = Normal(1e3*self.data, self.noise_tol).log_prob(
                1e3*output).mean()
        
        log_prior_phys = Normal(torch.zeros(len(ret)-1).to(device=self.cfg.device), 1e-5).log_prob(
            torch.relu(ret[1:]-ret[:-1])).mean()

        loss = -kl_weight * (log_prior + log_prior_phys) - log_like
        
        return loss


class Metropolis_Hastings(Sampler):
    def __init__(self, model, model_temp, step_size, noise_tol, cfg, data,
                 update_freq=10, save_freq=50, continue_sampling=False):
        
        super().__init__(model, model_temp, step_size, noise_tol, cfg, data,
                 update_freq, save_freq, continue_sampling)
    
    def acceptance(self, u, t):
        
        self.param_new = torch.normal(self.param_old, self.step_size)
        
        torch.nn.utils.vector_to_parameters(self.param_new, self.model_temp.parameters())
        self.output_new = self.model_out(self.model_temp, u, t)
        self.func_new = self.func_nn(self.model_temp)
        self.post_new = -self.loss(self.model_temp, self.output_new, self.func_new)
        
        acc = torch.min(torch.tensor([1.]).to(device=self.cfg.device),
                        torch.exp(self.post_new-self.post_old))
        
        print(self.post_new.item(), self.post_old.item())
        
        return acc
    
    def sample(self, iters, u, t):
        
        if self.continue_sampling:
            pickle_file = open(self.save_path, "rb")
            self = pickle.load(pickle_file)
            pickle_file.close()
            self.save_path = os.path.abspath("") + "/" + self.cfg.config.sampling.name + ".pickle"
        else:
            self.param_sample = []
            self.output_sample = []
            self.func_sample = []
            self.post_sample = []
            
            self.param_old = torch.nn.utils.parameters_to_vector(self.model.parameters())
            self.output_old = self.model_out(self.model, u, t)
            self.func_old = self.func_nn(self.model)
            self.post_old = -self.loss(self.model, self.output_old, self.func_old)
        
            with torch.no_grad():
                self.param_sample.append(self.param_old)
                self.output_sample.append(self.output_old)
                self.func_sample.append(self.func_old)
                self.post_sample.append(self.post_old.item())
        
            self.step = 0
            self.acc_rate = 0
    
        
        while self.step < iters - 1:
            
            q = self.acceptance(u, t)
            
            if q >= torch.rand(1).to(device=self.cfg.device):
                ind = 1
                self.param_old = torch.nn.utils.parameters_to_vector(self.model_temp.parameters())
                torch.nn.utils.vector_to_parameters(self.param_old, self.model.parameters())
                self.output_old = self.output_new
                self.func_old = self.func_new
                self.post_old = self.post_new
                self.acc_rate += 1
            else:
                ind = 0
                        
            self.step += 1
            print(self.step, q.item(), ind)
            
            if (self.step+1)%self.update_freq == 0:
                with torch.no_grad():
                    self.param_sample.append(self.param_old)
                    self.output_sample.append(self.output_old)
                    self.func_sample.append(self.func_old)
                    self.post_sample.append(self.post_old.item())
            
            if (self.step+1)%self.save_freq == 0:
                pickle_file = open(self.save_path, "wb")
                pickle.dump(self, pickle_file)
                pickle_file.close()
            
        acc_rate = self.acc_rate/self.step
        print(acc_rate)
        

class MALA(Sampler):
    def __init__(self, model, model_temp, step_size, noise_tol, cfg, data,
                 update_freq=10, save_freq=50, continue_sampling=False):
        
        super().__init__(model, model_temp, step_size, noise_tol, cfg, data,
                 update_freq, save_freq, continue_sampling)
    
    def acceptance(self, u, t):
        
        self.param_new = self.param_old - self.step_size*self.grad_old + np.sqrt(2*self.step_size) * self.noise.rsample()
        
        torch.nn.utils.vector_to_parameters(self.param_new, self.model_temp.parameters())
        self.output_new = self.model_out(self.model_temp, u, t)
        self.func_new = self.func_nn(self.model_temp)
        self.post_new = -self.loss(self.model_temp, self.output_new, self.func_new)
        grad = torch.autograd.grad(-self.post_new, self.model_temp.parameters())
        self.grad_new = torch.nn.utils.parameters_to_vector(grad)
        
        q_num = self.post_new - torch.sum((self.param_old - self.param_new +
                    self.step_size*self.grad_new)**2)/(4*self.step_size)
        q_den = self.post_old - torch.sum((self.param_new - self.param_old +
                        self.step_size*self.grad_old)**2)/(4*self.step_size)
        
        acc = torch.min(torch.tensor([1.]).to(device=self.cfg.device),
                        torch.exp(q_num-q_den))
        
        print(q_num.item(), q_den.item())
        
        return acc
    
    def sample(self, iters, u, t):
        
        # inp = torch.linspace(0,2,501).unsqueeze(-1).to(device=self.cfg.device)
        
        if self.continue_sampling:
            pickle_file = open(self.save_path, "rb")
            self = pickle.load(pickle_file)
            pickle_file.close()
            self.save_path = os.path.abspath("") + "/" + self.cfg.config.sampling.name + ".pickle"
        else:
            self.param_sample = []
            self.output_sample = []
            self.func_sample = []
            self.post_sample = []
            
            self.param_old = torch.nn.utils.parameters_to_vector(self.model.parameters())
            self.output_old = self.model_out(self.model, u, t)
            self.func_old = self.func_nn(self.model)
            self.post_old = -self.loss(self.model, self.output_old, self.func_old)
            grad = torch.autograd.grad(-self.post_old, self.model.parameters())
            self.grad_old = torch.nn.utils.parameters_to_vector(grad)
        
            with torch.no_grad():
                self.param_sample.append(self.param_old)
                self.output_sample.append(self.output_old)
                self.func_sample.append(self.func_old)
                self.post_sample.append(self.post_old.item())
        
            self.step = 0
            self.acc_rate = 0
    
        
        while self.step < iters - 1:
            
            q = self.acceptance(u, t)
            
            if q >= torch.rand(1).to(device=self.cfg.device):
                ind = 1
                self.param_old = torch.nn.utils.parameters_to_vector(self.model_temp.parameters())
                torch.nn.utils.vector_to_parameters(self.param_old, self.model.parameters())
                self.output_old = self.output_new
                self.func_old = self.func_new
                self.post_old = self.post_new
                self.acc_rate += 1
            else:
                ind = 0
                        
            self.step += 1
            print(self.step, q.item(), ind)
            
            if (self.step+1)%self.update_freq == 0:
                with torch.no_grad():
                    self.param_sample.append(self.param_old)
                    self.output_sample.append(self.output_old)
                    self.func_sample.append(self.func_old)
                    self.post_sample.append(self.post_old.item())
            
            if (self.step+1)%self.save_freq == 0:
                pickle_file = open(self.save_path, "wb")
                pickle.dump(self, pickle_file)
                pickle_file.close()
            
        acc_rate = self.acc_rate/self.step
        print(acc_rate)
        

class Barker(Sampler):
    def __init__(self, model, model_temp, step_size, noise_tol, cfg, data,
                 update_freq=10, save_freq=50, continue_sampling=False):
        
        super().__init__(model, model_temp, step_size, noise_tol, cfg, data,
                 update_freq, save_freq, continue_sampling)
    
    def acceptance(self, u, t):
        
        z = torch.normal(torch.zeros_like(self.param_old), self.step_size)
        prob_z = 1/(1 + torch.exp(-z * self.grad_old))
        prob_acc = torch.rand(len(prob_z))
        b_pos = (prob_z >= prob_acc) * 1
        b_neg = (prob_z < prob_acc) * -1
        
        self.param_new = self.param_old + b_pos*z + b_neg*z
        
        torch.nn.utils.vector_to_parameters(self.param_new, self.model_temp.parameters())
        self.output_new = self.model_out(self.model_temp, u, t)
        self.func_new = self.func_nn(self.model_temp)
        self.post_new = -self.loss(self.model_temp, self.output_new, self.func_new)
        grad = torch.autograd.grad(self.post_new, self.model_temp.parameters())
        self.grad_new = torch.nn.utils.parameters_to_vector(grad)
        
        q_num = self.post_new + torch.sum(torch.log(1+torch.exp((
            self.param_old-self.param_new)*self.grad_old)))
        q_den = self.post_old + torch.sum(torch.log(1+torch.exp((
            self.param_new-self.param_old)*self.grad_new)))
        
        acc = torch.min(torch.tensor([1.]).to(device=self.cfg.device),
                        torch.exp(q_num-q_den))
        
        print(q_num.item(), q_den.item())
        
        return acc
    
    def sample(self, iters, u, t):
        
        # inp = torch.linspace(0,2,501).unsqueeze(-1).to(device=self.cfg.device)
        
        if self.continue_sampling:
            pickle_file = open(self.save_path, "rb")
            self = pickle.load(pickle_file)
            pickle_file.close()
            self.save_path = os.path.abspath("") + "/" + self.cfg.config.sampling.name + ".pickle"
        else:
            self.param_sample = []
            self.output_sample = []
            self.func_sample = []
            self.post_sample = []
            
            self.param_old = torch.nn.utils.parameters_to_vector(self.model.parameters())
            self.output_old = self.model_out(self.model, u, t)
            self.func_old = self.func_nn(self.model)
            self.post_old = -self.loss(self.model, self.output_old, self.func_old)
            grad = torch.autograd.grad(self.post_old, self.model.parameters())
            self.grad_old = torch.nn.utils.parameters_to_vector(grad)
        
            with torch.no_grad():
                self.param_sample.append(self.param_old)
                self.output_sample.append(self.output_old)
                self.func_sample.append(self.func_old)
                self.post_sample.append(self.post_old.item())
        
            self.step = 0
            self.acc_rate = 0
    
        
        while self.step < iters - 1:
            
            q = self.acceptance(u, t)
            
            if q >= torch.rand(1).to(device=self.cfg.device):
                ind = 1
                self.param_old = torch.nn.utils.parameters_to_vector(self.model_temp.parameters())
                torch.nn.utils.vector_to_parameters(self.param_old, self.model.parameters())
                self.output_old = self.output_new
                self.func_old = self.func_new
                self.post_old = self.post_new
                self.acc_rate += 1
            else:
                ind = 0
                        
            self.step += 1
            print(self.step, q.item(), ind)
            
            if (self.step+1)%self.update_freq == 0:
                with torch.no_grad():
                    self.param_sample.append(self.param_old)
                    self.output_sample.append(self.output_old)
                    self.func_sample.append(self.func_old)
                    self.post_sample.append(self.post_old.item())
            
            if (self.step+1)%self.save_freq == 0:
                pickle_file = open(self.save_path, "wb")
                pickle.dump(self, pickle_file)
                pickle_file.close()
            
        acc_rate = self.acc_rate/self.step
        print(acc_rate)