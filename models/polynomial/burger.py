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
data_type = "burger"
data_name = "data_train"
data_name_ext = "data_ext"
data_path = os.path.join(root_path, data_type, data_name)

# Polynomial configuration
max_degree = 7

# Load data and reshape
t = np.load(os.path.join(data_path, "t_series.npy"))
x = np.load(os.path.join(data_path, "x_series.npy"))
u = np.load(os.path.join(data_path, "sample.npy"))

t_mesh, x_mesh = np.meshgrid(t,x)
t_mesh = t_mesh.flatten()
x_mesh = x_mesh.flatten()

inp_mesh = np.stack((t_mesh,x_mesh), -1)
out_mesh = u.transpose().flatten()
out_mesh = np.expand_dims(out_mesh,-1)

# Fit polynomial to data
model = make_pipeline(PolynomialFeatures(max_degree), Lasso(alpha=1e-3))
model.fit(inp_mesh,out_mesh)

# Predict training data and plot
pred = model.predict(inp_mesh)
pred = pred.reshape((len(x),len(t))).transpose()

fig, ax = plt.subplots(2, 2, figsize=(4.5,4))
h = ax[0,0].imshow(pred.transpose(), interpolation='nearest',
              extent=[t.min(), t.max(),
                      x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax[0,0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(h, cax=cax)
h.set_clim(u.min(), u.max())
ax[0,0].set_title("Training")
ax[0,0].set_ylabel("$x$")
ax[0,0].set_xlabel("$t$")
ax[0,0].axes.set_xticks([0, 0.5, 1.0])
ax[0,0].axes.set_yticks([-1, 0, 1])

h = ax[1,0].plot(x, u[-1], "ro-", markersize=2, linewidth = 0.5)
h = ax[1,0].plot(x, pred[-1])
ax[1,0].set_title("t = 1")
ax[1,0].set_ylabel("$u$")
ax[1,0].set_xlabel("$x$")
ax[1,0].axes.set_xticks([-1, 0, 1])
ax[1,0].axes.set_yticks([-0.5, 0, 0.5])

print("MSE train = ", np.mean((u-pred)**2))


# # Extrapolation

# Load data and reshape
data_path = os.path.join(root_path, data_type, data_name_ext)
t = np.load(os.path.join(data_path, "t_series.npy"))
x = np.load(os.path.join(data_path, "x_series.npy"))
u = np.load(os.path.join(data_path, "sample.npy"))

t_mesh, x_mesh = np.meshgrid(t,x)
t_mesh = t_mesh.flatten()
x_mesh = x_mesh.flatten()

inp_mesh = np.stack((t_mesh,x_mesh),-1)
out_mesh = u.transpose().flatten()
out_mesh = np.expand_dims(out_mesh,-1)

# Predict extrapolation and plot

pred = model.predict(inp_mesh)
pred = pred.reshape((len(x),len(t))).transpose()

# fig, ax = plt.subplots(2, 1, figsize=(3,5))
h = ax[0,1].imshow(pred.transpose(), interpolation='nearest',
              extent=[t.min(), t.max(),
                      x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax[0,1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(h, cax=cax)
h.set_clim(u.min(), u.max())
ax[0,1].set_title("Extrapolation")
ax[0,1].set_ylabel("$x$")
ax[0,1].set_xlabel("$t$")
ax[0,1].axes.set_xticks([0, 1.0, 2.0])
ax[0,1].axes.set_yticks([-1, 0, 1])

h = ax[1,1].plot(x, u[-1], "ro-", markersize=2, linewidth = 0.5, label="Data")
h = ax[1,1].plot(x, pred[-1],label="Prediction")
ax[1,1].set_title("t = 2")
ax[1,1].set_ylabel("$u$")
ax[1,1].set_xlabel("$x$")
ax[1,1].axes.set_xticks([-1, 0, 1])
ax[1,1].axes.set_yticks([-0.5, 0, 0.5])

fig.legend(loc=8, ncol=2)
fig.suptitle(f"Burgers', polynomial (d=5)")
plt.tight_layout(rect=[0,0.05,1,0.95])  # [left, bottom, right, top]
plt.savefig('poly_burger.pdf')

print("MSE ext = ", np.mean((u-pred)**2))