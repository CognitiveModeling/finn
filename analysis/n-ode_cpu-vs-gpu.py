# -*- coding: utf-8 -*-
"""
This script contrasts the runtime of training a simple feed forward neural
network under incorporation of the Neural ODE method from
https://github.com/rtqichen/torchdiffeq
"""

import torch.nn as nn
import torch as th
from torchdiffeq import odeint
import time
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, t, y):
        return self.net(y)


device_cpu = th.device("cpu")
device_cuda = th.device("cuda")

t_cpu = th.linspace(0,1,1001).to(device=device_cpu)
t_cuda = th.linspace(0,1,1001).to(device=device_cuda)

model_cpu = Model().to(device=device_cpu)
model_cuda = Model().to(device=device_cuda)

runtime_cpu = th.empty(240)
runtime_cuda = th.empty(240)
num_batches = th.empty(240)

with th.no_grad():
    for i in range(10,250):
        num_batches[i-10] = i**2
        inp_cpu = th.zeros(i**2,1).to(device=device_cpu)
        inp_cuda = th.zeros(i**2,1).to(device=device_cuda)
        start_time_cpu = time.time()
        out_cpu = odeint(model_cpu,inp_cpu,t_cpu)
        runtime_cpu[i-10] = time.time() - start_time_cpu
        start_time_cuda = time.time()
        out_cuda = odeint(model_cuda,inp_cuda,t_cuda)
        runtime_cuda[i-10] = time.time() - start_time_cuda
        
        print(i-10, runtime_cpu[i-10], runtime_cuda[i-10])
    
plt.figure()
plt.plot(num_batches, runtime_cpu, label="CPU")
plt.plot(num_batches, runtime_cuda, label="GPU")
plt.xlabel("Number of batches")
plt.ylabel("Runtime [s]")
plt.title("Neural ODE runtime comparison")
plt.legend()
plt.tight_layout()
plt.show()
