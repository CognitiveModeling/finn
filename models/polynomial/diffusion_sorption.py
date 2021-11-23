# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 09:15:03 2021
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Data path
root_path = os.path.abspath("../../data")
data_type = "diffusion_sorption"
data_name = "data_train"
data_name_ext = "data_ext"
data_path = os.path.join(root_path, data_type, data_name)

# Polynomial configuration
max_degree = 3

# Load data and reshape
t = np.load(os.path.join(data_path, "t_series.npy"))
x = np.load(os.path.join(data_path, "x_series.npy"))
c = np.load(os.path.join(data_path, "sample_c.npy"))
ct = np.load(os.path.join(data_path, "sample_ct.npy"))

t_mesh, x_mesh = np.meshgrid(t,x)
t_mesh = t_mesh.flatten()
x_mesh = x_mesh.flatten()

inp_mesh = np.stack((t_mesh,x_mesh),-1)
c_flat = c.transpose().flatten()
ct_flat = ct.transpose().flatten()
out_mesh = np.stack((c_flat,ct_flat),-1)

# Fit polynomial to data
model = make_pipeline(PolynomialFeatures(max_degree), Lasso(alpha=1e-3))
model.fit(inp_mesh,out_mesh)

# Predict training data and plot
pred = model.predict(inp_mesh)[...,0]
pred = pred.reshape((len(x),len(t))).transpose()

fig, ax = plt.subplots(2, 2, figsize=(4.5,4))
h = ax[0,0].imshow(pred.transpose(), interpolation='nearest',
              extent=[t.min(), t.max(),
                      x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax[0,0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(h, cax=cax)
h.set_clim(c.min(), c.max())
ax[0,0].set_title("Training")
ax[0,0].set_ylabel("$x$")
ax[0,0].set_xlabel("$t$")
ax[0,0].axes.set_xticks([0, 2500])
ax[0,0].axes.set_yticks([0.0, 0.5, 1.0])
ax[0,0].set_xticklabels([0, 2500])
ax[0,0].set_yticklabels([0.0, 0.5, 1.0])

h = ax[1,0].plot(x, c[-1], "ro-", markersize=2, linewidth = 0.5)
h = ax[1,0].plot(x, pred[-1])
ax[1,0].set_title("t = 2500")
ax[1,0].set_ylabel("$u$")
ax[1,0].set_xlabel("$x$")
ax[1,0].axes.set_xticks([0, 0.5, 1])
ax[1,0].axes.set_yticks([0, 0.5, 1])
ax[1,0].set_xticklabels([0, 0.5, 1])
ax[1,0].set_yticklabels([0, 0.5, 1])

print("MSE train = ", np.mean((c-pred)**2))

# Extrapolation

# Load data and reshape
data_path = os.path.join(root_path, data_type, data_name_ext)
t = np.load(os.path.join(data_path, "t_series.npy"))
x = np.load(os.path.join(data_path, "x_series.npy"))
c = np.load(os.path.join(data_path, "sample_c.npy"))
ct = np.load(os.path.join(data_path, "sample_ct.npy"))

t_mesh, x_mesh = np.meshgrid(t,x)
t_mesh = t_mesh.flatten()
x_mesh = x_mesh.flatten()

inp_mesh = np.stack((t_mesh,x_mesh),-1)
c_flat = c.transpose().flatten()
ct_flat = ct.transpose().flatten()
out_mesh = np.stack((c_flat,ct_flat),-1)

# Predict extrapolation and plot
pred = model.predict(inp_mesh)[...,0]
pred = pred.reshape((len(x),len(t))).transpose()

h = ax[0,1].imshow(pred.transpose(), interpolation='nearest',
              extent=[t.min(), t.max(),
                      x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax[0,1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(h, cax=cax)
h.set_clim(c.min(), c.max())
ax[0,1].set_title("Extrapolation")
ax[0,1].set_ylabel("$x$")
ax[0,1].set_xlabel("$t$")
ax[0,1].axes.set_xticks([0, 10000])
ax[0,1].axes.set_yticks([0.0, 0.5, 1.0])
ax[0,1].set_xticklabels([0, 10000])
ax[0,1].set_yticklabels([0.0, 0.5, 1.0])

h = ax[1,1].plot(x, c[-1], "ro-", markersize=2, linewidth = 0.5, label="Data")
h = ax[1,1].plot(x, pred[-1],label="Prediction")
ax[1,1].set_title("t = 10000")
ax[1,1].set_ylabel("$u$")
ax[1,1].set_xlabel("$x$")
ax[1,1].axes.set_xticks([0, 0.5, 1])
ax[1,1].axes.set_yticks([0, 10, 20])
ax[1,1].set_xticklabels([0, 0.5, 1])
ax[1,1].set_yticklabels([0, 10, 20])

fig.legend(loc=8, ncol=2)
plt.suptitle("Diffusion sorption, polynomial (d=3)")
plt.tight_layout(rect=[0,0.05,1,0.95])  # [left, bottom, right, top]
plt.savefig('poly_diff_sorp.pdf')

print("MSE ext = ", np.mean((c-pred)**2))