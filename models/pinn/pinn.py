#! env/bin/python3

"""
PyTorch implementation of the physics informed neural network for continuous
PDE determination. Inspired from the original tensorflow model from
https://github.com/maziarraissi/PINNs
"""

import numpy as np
import torch as th
import torch.nn as nn


class PINN(nn.Module):
    """
    Physics informed neural network class
    """
    
    def __init__(self):
        """
        Constructor method to initialize a PINN instance.
        """
        super(PINN, self).__init__()

    def init_approximation_net(self, layer_sizes, device):
        """
        This method sets up the neural network for solving e.g. u(t, x).
        :param layer_sizes: A list of neurons per layer
        :param device: The device on which tensor operations are performed
        """
        layers = list()

        for layer_idx in range(len(layer_sizes) - 1):
            layer = nn.Linear(
                in_features=layer_sizes[layer_idx],
                out_features=layer_sizes[layer_idx + 1],
                bias=True
            ).to(device=device)
            layers.append(layer)

            if layer_idx < len(layer_sizes) - 2:
                layers.append(nn.Tanh())

        return nn.Sequential(*nn.ModuleList(layers))

    def forward(self):
        pass

    def net_f(self):
        pass

    def dv1_dv2(self, v1, v2):
        """
        This method computes the derivative of v1 with respect to v2 using
        PyTorch's autograd from
        https://pytorch.org/docs/stable/generated/torch.autograd.grad.html
        :param v1: The variable that has to be derived
        :param v2: Variable with respect to which the function is derived to
        :return: The derivative of v1 w.r.t. v2
        """
        return th.autograd.grad(outputs=v1,
                                inputs=v2, 
                                grad_outputs=th.ones_like(v1),
                                retain_graph=True,
                                create_graph=True)[0]


class PINN_Burger(PINN):
    """
    PINN class for learning and predicting the Burger's equation.
    """

    def __init__(self, layer_sizes, device):
        """
        Constructor method to initialize a PINN instance.
        :param layer_sizes: List with numbers of neurons per layer
        :param device: The device (GPU/CPU) for the calculations
        """
        super(PINN_Burger, self).__init__()

        # Initialize the neural network for approximating u
        self.net_u = self.init_approximation_net(layer_sizes=layer_sizes,
                                                 device=device)

    def forward(self, t, x):
        """
        Computes the forward pass of the PINN network given an input.
        :param t: The current time step
        :param x: The input to the model
        :return: The output of the model
        """
        t.requires_grad_(True)
        x.requires_grad_(True)
        u = self.net_u(th.cat((t, x), dim=1))
        f = self.net_f(u, t, x)
        return u, f

    def net_f(self, u, t, x):
        """
        Forward pass of the neural net that solves f based on automatic
        differentiation.
        :param u: The solution of net_u for u(t, x)
        :param t: Current time step
        :param x: Model input
        :return: The solution of net_f
        """
        u_t = self.dv1_dv2(v1=u, v2=t)
        u_x = self.dv1_dv2(v1=u, v2=x)
        u_xx = self.dv1_dv2(v1=u_x, v2=x)
        return u_t + u*u_x - (0.01/np.pi)*u_xx


class PINN_DiffSorp(PINN):
    """
    PINN class for learning and predicting the diffusion sorption equations.
    """

    def __init__(self, layer_sizes, device, config):
        """
        Constructor method to initialize a PINN instance.
        :param layer_sizes: List with numbers of neurons per layer
        :param device: The device (GPU/CPU) for the calculations
        :param config: The configuration file
        """
        super(PINN_DiffSorp, self).__init__()

        # Initialize the neural network for approximating u
        self.net_uv = self.init_approximation_net(layer_sizes=layer_sizes,
                                                 device=device)
        
        self.D = config.data.diffusion_sorption.D
        self.por = config.data.diffusion_sorption.porosity
        self.rho_s = config.data.diffusion_sorption.rho_s
        self.k_f = config.data.diffusion_sorption.k_f_nominator/self.rho_s
        self.n_f = config.data.diffusion_sorption.n_f
        self.s_max = config.data.diffusion_sorption.s_max
        self.kl = config.data.diffusion_sorption.kl
        self.kd = config.data.diffusion_sorption.kd
        self.solubility = config.data.diffusion_sorption.solubility

        # Set the retardation function
        if "linear" in config.data.name:
            self.retardation_function = self.retardation_linear
        elif "freundlich" in config.data.name:
            self.retardation_function = self.retardation_freundlich
        elif "langmuir" in config.data.name:
            self.retardation_function = self.retardation_langmuir
            
    def init_approximation_net(self, layer_sizes, device):
        """
        This method sets up the neural network for solving e.g. u(t, x).
        :param layer_sizes: A list of neurons per layer
        :param device: The device on which tensor operations are performed
        """
        layers = list()

        for layer_idx in range(len(layer_sizes) - 1):
            layer = nn.Linear(
                in_features=layer_sizes[layer_idx],
                out_features=layer_sizes[layer_idx + 1],
                bias=True
            ).to(device=device)
            layers.append(layer)

            if layer_idx < len(layer_sizes) - 2:
                layers.append(nn.Tanh())
            elif layer_idx == len(layer_sizes) -2:
                layers.append(nn.Sigmoid())

        return nn.Sequential(*nn.ModuleList(layers))

    def forward(self, t, x):
        """
        Computes the forward pass of the PINN network given an input.
        :param t: The current time step
        :param x: The input to the model
        :return: The output of the model
        """
        t.requires_grad_(True)
        x.requires_grad_(True)
        uv = self.net_uv(th.cat((t/1000, x), dim=1))
        u = uv[...,0:1]
        v = uv[...,1:]
        f = self.net_f(u, t, x)
        g = self.net_g(u, v, t, x)
        return u, v, f, g

    def net_f(self, u, t, x):
        """
        Forward pass of the neural net that solves f based on automatic
        differentiation.
        :param u: The solution of net_u for u(t, x)
        :param t: Current time step
        :param x: Model input
        :return: The solution of net_f
        """
        u_t = self.dv1_dv2(v1=u, v2=t)
        u_x = self.dv1_dv2(v1=u, v2=x)
        u_xx = self.dv1_dv2(v1=u_x, v2=x)
        
        return u_t - (self.D/self.retardation_function(u=u))*u_xx
    
    def net_g(self, u, v, t, x):
        """
        Forward pass of the neural net that solves f based on automatic
        differentiation.
        :param u: The solution of net_u for u(t, x)
        :param v: The solution of net_u for v(t, x)
        :param t: Current time step
        :param x: Model input
        :return: The solution of net_f
        """
        v_t = self.dv1_dv2(v1=v, v2=t)
        u_x = self.dv1_dv2(v1=u, v2=x)
        u_xx = self.dv1_dv2(v1=u_x, v2=x)
        return v_t - (self.D*self.por/(self.rho_s/1000))*u_xx

    def retardation_linear(self, u):
        """
        Linear retardation factor function.
        :param u: The simulation field
        :return: The linearly computed retardation factor
        """
        return 1 + ((1 - self.por)/self.por)*self.rho_s*self.kd

    def retardation_freundlich(self, u):
        """
        Freundlich redardation factor function.
        :param u: The simulation field
        :return: The Freundlich-based retardation factor
        """
        return 1 + ((1 - self.por)/self.por)*self.rho_s*self.k_f*self.n_f*\
                    (u + 1e-6)**(self.n_f - 1)

    def retardation_langmuir(self, u):
        """
        Langmuir retardation factor function.
        :param u: The simulation field
        :return: The Langmuir-based retardation factor
        """
        return 1 + ((1 - self.por)/self.por)*self.rho_s*\
                    ((self.s_max*self.kl)/(u + self.kl)**2)


class PINN_DiffReact(PINN):
    """
    PINN class for learning and predicting the diffusion reaction 
    Fitzhugh-Nagumo equations.
    """

    def __init__(self, layer_sizes, device, config):
        """
        Constructor method to initialize a PINN instance.
        :param layer_sizes: List with numbers of neurons per layer
        :param device: The device (GPU/CPU) for the calculations
        :param config: The configuration file
        """
        super(PINN_DiffReact, self).__init__()

        self.k = config.data.diffusion_reaction.k
        self.D_u = config.data.diffusion_reaction.D_u
        self.D_v = config.data.diffusion_reaction.D_v

        # Initialize the neural networks for approximating u and v
        self.net_uv = self.init_approximation_net(layer_sizes=layer_sizes,
                                                  device=device)

    def forward(self, t, x, y):
        """
        Computes the forward pass of the PINN network given an input.
        :param t: The current time step
        :param x: The input to the model
        :return: The output of the model
        """

        t.requires_grad_(True)
        x.requires_grad_(True)
        y.requires_grad_(True)

        uv = self.net_uv(th.cat((t, x, y), dim=1))
        u = uv[...,0:1]
        v = uv[...,1:]
        
        f = self.net_f(u, v, t, x, y)
        g = self.net_g(u, v, t, x, y)
        #f = th.zeros_like(u)
        #g = th.zeros_like(v)

        return u, v, f, g

    def net_f(self, u, v, t, x, y):
        """
        Forward pass of the neural net that solves f based on automatic
        differentiation.
        :param u: The solution of net_u for u(t, x, y)
        :param v: The solution of net_v for v(t, x, y)
        :param t: Current time step
        :param x: Model input x-position
        :param y: Model input y-position
        :return: The solution of net_f
        """
        u_t = self.dv1_dv2(u, t)
        u_x = self.dv1_dv2(u, x)
        u_y = self.dv1_dv2(u, y)
        u_xx = self.dv1_dv2(u_x, x)
        u_yy = self.dv1_dv2(u_y, y)
        return u - u**3 - self.k - v + self.D_u*(u_xx + u_yy) - u_t

    def net_g(self, u, v, t, x, y):
        """
        Forward pass of the neural net that solves g based on automatic
        differentiation.
        :param u: The solution of net_u for u(t, x, y)
        :param v: The solution of net_v for v(t, x, y)
        :param t: Current time step
        :param x: Model input x-position
        :param y: Model input y-position
        :return: The solution of net_f
        """
        v_t = self.dv1_dv2(v, t)
        v_x = self.dv1_dv2(v, x)
        v_y = self.dv1_dv2(v, y)
        v_xx = self.dv1_dv2(v_x, x)
        v_yy = self.dv1_dv2(v_y, y)
        return u - v + self.D_v*(v_xx + v_yy) - v_t
    
class PINN_AllenCahn(PINN):
    """
    PINN class for learning and predicting the Burger's equation.
    """

    def __init__(self, layer_sizes, device):
        """
        Constructor method to initialize a PINN instance.
        :param layer_sizes: List with numbers of neurons per layer
        :param device: The device (GPU/CPU) for the calculations
        """
        super(PINN_AllenCahn, self).__init__()

        # Initialize the neural network for approximating u
        self.net_u = self.init_approximation_net(layer_sizes=layer_sizes,
                                                 device=device)

    def forward(self, t, x):
        """
        Computes the forward pass of the PINN network given an input.
        :param t: The current time step
        :param x: The input to the model
        :return: The output of the model
        """
        t.requires_grad_(True)
        x.requires_grad_(True)
        u = self.net_u(th.cat((t, x), dim=1))
        f = self.net_f(u, t, x)
        return u, f

    def net_f(self, u, t, x):
        """
        Forward pass of the neural net that solves f based on automatic
        differentiation.
        :param u: The solution of net_u for u(t, x)
        :param t: Current time step
        :param x: Model input
        :return: The solution of net_f
        """
        u_t = self.dv1_dv2(v1=u, v2=t)
        u_x = self.dv1_dv2(v1=u, v2=x)
        u_xx = self.dv1_dv2(v1=u_x, v2=x)
        return u_t - 0.0001*u_xx + 5*u**3 -5*u


class PINN_Burger2D(PINN):
    """
    PINN class for learning and predicting the Burger's equation.
    """

    def __init__(self, layer_sizes, device):
        """
        Constructor method to initialize a PINN instance.
        :param layer_sizes: List with numbers of neurons per layer
        :param device: The device (GPU/CPU) for the calculations
        """
        super(PINN_Burger2D, self).__init__()

        # Initialize the neural network for approximating u
        self.net_u = self.init_approximation_net(layer_sizes=layer_sizes,
                                                 device=device)

    def forward(self, t, x, y):
        """
        Computes the forward pass of the PINN network given an input.
        :param t: The current time step
        :param x: The input to the model
        :return: The output of the model
        """
        t.requires_grad_(True)
        x.requires_grad_(True)
        y.requires_grad_(True)
        u = self.net_u(th.cat((t, x, y), dim=1))
        f = self.net_f(u, t, x, y)
        return u, f

    def net_f(self, u, t, x, y):
        """
        Forward pass of the neural net that solves f based on automatic
        differentiation.
        :param u: The solution of net_u for u(t, x)
        :param t: Current time step
        :param x: Model input
        :return: The solution of net_f
        """
        u_t = self.dv1_dv2(v1=u, v2=t)
        u_x = self.dv1_dv2(v1=u, v2=x)
        u_xx = self.dv1_dv2(v1=u_x, v2=x)
        u_y = self.dv1_dv2(v1=u, v2=y)
        u_yy = self.dv1_dv2(v1=u_y, v2=y)
        return u_t + u*u_x + u*u_y - (0.01/np.pi)*(u_xx + u_yy)
