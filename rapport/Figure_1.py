#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Imports 
get_ipython().run_line_magic('matplotlib', 'qt')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from matplotlib.colors import hsv_to_rgb
import tifffile as tiff
import math
from sklearn.cluster import OPTICS
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np


# In[2]:


data = pd.read_csv(r'C:\Users\LOCCO\Project_Curie\pour_lola\zone3_cc.csv',sep=',')
print(data.columns)
mask = tiff.imread(r"C:\Users\LOCCO\Project_Curie\pour_lola\zone3_mask1.tif")    
mask = np.array(mask.transpose()) #Fiji écrit en y,x

#Import data
frame = data['frame'].values
X = data[['x [nm]', 'y [nm]', 'z [nm]']].values
rho = data['rho'].values
delta = data['delta'].values
N_photons = data['intensity'].values
sigma = data[['sigmax [nm]', 'sigmay [nm]', 'sigmaz [nm]']].values

ix = np.clip((X[:, 0] / (120/5)).astype(int), 0, mask.shape[0] - 1)
iy = np.clip((X[:, 1] / (120/5)).astype(int), 0, mask.shape[1] - 1)

mask_vect = (mask[ix, iy] > 0) 

X_masked = X[mask_vect]
sigma_masked_z31   = sigma[mask_vect]
rho_masked     = rho[mask_vect]
delta_masked   = delta[mask_vect]
frame_masked   = frame[mask_vect]


# In[3]:


data = pd.read_csv(r'C:\Users\LOCCO\Project_Curie\acquisiton_souris\image_Pos0_driftcorrected.csv',sep=',')
print(data.columns)
mask = tiff.imread(r"C:\Users\LOCCO\Project_Curie\acquisiton_souris\mask_bis.tif")    
mask = np.array(mask.transpose()) #Fiji écrit en y,x

#Import data
frame = data['frame'].values
X = data[['x [nm]', 'y [nm]', 'z [nm]']].values
rho = data['rho'].values
delta = data['delta'].values
N_photons = data['intensity'].values
sigma = data[['sigmax [nm]', 'sigmay [nm]', 'sigmaz [nm]']].values

ix = np.clip((X[:, 0] / (120/5)).astype(int), 0, mask.shape[0] - 1)
iy = np.clip((X[:, 1] / (120/5)).astype(int), 0, mask.shape[1] - 1)

mask_vect = (mask[ix, iy] > 0) 

X_masked = X[mask_vect]
sigma_masked_z0   = sigma[mask_vect]
rho_masked     = rho[mask_vect]
delta_masked   = delta[mask_vect]
frame_masked   = frame[mask_vect]


# In[4]:


data = pd.read_csv(r'C:\Users\LOCCO\Project_Curie\pour_lola\zone1_cc.csv',sep=',')
print(data.columns)
mask = tiff.imread(r"C:\Users\LOCCO\Project_Curie\pour_lola\zone1_mask.tif")    
mask = np.array(mask.transpose()) #Fiji écrit en y,x

#Import data
frame = data['frame'].values
X = data[['x [nm]', 'y [nm]', 'z [nm]']].values
rho = data['rho'].values
delta = data['delta'].values
N_photons = data['intensity'].values
sigma = data[['sigmax [nm]', 'sigmay [nm]', 'sigmaz [nm]']].values

ix = np.clip((X[:, 0] / (120/5)).astype(int), 0, mask.shape[0] - 1)
iy = np.clip((X[:, 1] / (120/5)).astype(int), 0, mask.shape[1] - 1)

mask_vect = (mask[ix, iy] > 0) 

X_masked = X[mask_vect]
sigma_masked_z1   = sigma[mask_vect]
rho_masked     = rho[mask_vect]
delta_masked   = delta[mask_vect]
frame_masked   = frame[mask_vect]


# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 3 sous-graphiques côte à côte (1 ligne, 3 colonnes)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

datasets = [sigma_masked_z1, sigma_masked_z31, sigma_masked_z0]
labels = ["Noyau 0", "Noyau 1", "Noyau 2"]

# --- Sigma_x ---
for data, label in zip(datasets, labels):
    values = data[:, 0]
    x = np.linspace(0, values.max(), 500)
    kde = gaussian_kde(values)
    ax1.plot(x, kde(x), label=label)
ax1.axvline(x=240, color="red", linestyle="--", label="Seuil = 240")
ax1.set_title(r"Estimation de densité de $\sigma_x$")
ax1.set_xlabel(r"$\sigma_x$ (nm)")
ax1.set_ylabel("Densité")
ax1.legend()

# --- Sigma_y ---
for data, label in zip(datasets, labels):
    values = data[:, 1]
    x = np.linspace(0, values.max(), 500)
    kde = gaussian_kde(values)
    ax2.plot(x, kde(x), label=label)
ax2.axvline(x=240, color="red", linestyle="--", label="Seuil = 240")
ax2.set_title(r"Estimation de densité de $\sigma_y$")
ax2.set_xlabel(r"$\sigma_y$ (nm)")
ax2.set_ylabel("Densité")
ax2.legend()

# --- Sigma_z ---
for data, label in zip(datasets, labels):
    values = data[:, 2]
    x = np.linspace(0, values.max(), 500)
    kde = gaussian_kde(values)
    ax3.plot(x, kde(x), label=label)
ax3.axvline(x=720, color="red", linestyle="--", label="Seuil = 720")
ax3.set_title(r"Estimation de densité de $\sigma_z$ ")
ax3.set_xlabel(r"$\sigma_z$ (nm)")
ax3.set_ylabel("Densité")
ax3.legend()

#fig.suptitle("Détermination des seuils de coupure pour filtrage\nde points aberrants dans trois jeux de données")
plt.tight_layout()
plt.show()
fig.savefig("Incertitudes.pdf",bbox_inches="tight")


# In[8]:


import sys
get_ipython().system('{sys.executable} -m pip install seaborn')

