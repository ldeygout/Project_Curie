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
import matplotlib
import os
import imageio
import re


# In[2]:


#Import data treated
data_treated = pd.read_csv(r'C:\Users\LOCCO\Project_Curie\labmeeting\datatreatedA2Z0CCD70150.csv',sep=',')
frame_treated = data_treated['frame'].values
X_treated = (data_treated[['x [nm]', 'y [nm]', 'z [nm]']].values).T
rho_treated = data_treated['rho'].values
delta_treated = data_treated['delta'].values


# In[ ]:


os.makedirs(r'C:\Users\LOCCO\Project_Curie\labmeeting\temp_evol_A2Z1CCD70150', exist_ok=True)


# In[ ]:


def make_gif(folder, output_name):
    # Récupère tous les fichiers PNG triés
    file_list = sorted(
        [f for f in os.listdir(folder) if f.endswith(".png")],
        key=lambda f: int(re.search(r'\d+', f).group())
    )

    # Charge les images
    images = [imageio.imread(os.path.join(folder, f)) for f in file_list]

    # Sauvegarde le GIF
    imageio.mimsave(output_name, images, duration=0.1)


# In[ ]:


bins = np.arange(frame_treated.min(), frame_treated.max() + 1000,1000)

for i in range(len(bins)):
    mask_temp = (frame_treated < bins[i]+1000)
    x_temp, y_temp, z_temp = X_treated[0,mask_temp],X_treated[1,mask_temp],X_treated[2,mask_temp]
    rho_temp = rho_treated[mask_temp]
    x_mean, y_mean , z_mean = np.mean(x_temp), np.mean(y_temp), np.mean(z_temp)

    fig = plt.figure(figsize=(14, 6))
    ax_scatter = fig.add_axes([0.05, 0.1, 0.7, 0.85])  # [left, bottom, width, height]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=180)
    sc = ax_scatter.scatter(x_temp, y_temp,
                            c=rho_temp, cmap=matplotlib.colormaps['hsv'],
                            norm=norm, s=0.01)
    ax_scatter.set_xlim(X_treated[0,:].min(), X_treated[0,:].max())
    ax_scatter.set_ylim(X_treated[1,:].min(), X_treated[1,:].max())
    ax_scatter.set_aspect('equal')
    ax_scatter.set_title(f't = {int(bins[i]):04d}')
    fig.savefig(
        os.path.join(r"C:\Users\LOCCO\Project_Curie\labmeeting\temp_evol_A2Z0CCD70150", f"frame_{int(bins[i]):04d}.png"),
        dpi=100
    )
    plt.close(fig)


make_gif(r"C:\Users\LOCCO\Project_Curie\labmeeting\temp_evol_A2Z0CCD70150", r"C:\Users\LOCCO\Project_Curie\labmeeting\temp_evol_A2Z0CCD70150\animation_temporelle.gif")


# In[ ]:


os.makedirs(r"C:\Users\LOCCO\Project_Curie\labmeeting\temp_evol_A2Z1CCD70150\temp_evol_density_x", exist_ok=True)

# Épaisseur de la tranche en y et z 

x_center = 19000
x_thickness = 100  # ex: garde |y| < 2.5


# Bins en x fixes pour toutes les frames (cohérence du film)
y_bins = np.linspace(X_treated[:,1].min(), X_treated[:,1].max(), 500)

for i in range(len(bins)):
    mask_temp = (frame_treated < bins[i] + 1000)
    x_t, y_t, z_t =  X_treated[0,mask_temp],X_treated[1,mask_temp],X_treated[2,mask_temp]

    # Masque pour la tranche centrale en y et z


    mask_slice = (np.abs(x_t - x_center) <x_thickness/2) 
    y_slice = y_t[mask_slice]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_slice, bins=y_bins)
    ax.set_xlim(9500, 13000)
    ax.set_ylim(0, 100)  # ajuste si tu veux un ylim fixe pour toutes les frames
    ax.set_xlabel("x")
    ax.set_ylabel("Densité (comptes)")
    ax.set_title(f"Densité vs x | t = {int(bins[i]):04d}")

    fig.savefig(
        os.path.join(r"C:\Users\LOCCO\Project_Curie\labmeeting\temp_evol_A2Z1CCD70150\temp_evol_density_x", f"frame_{int(bins[i]):04d}.png"),
        dpi=100
    )
    plt.close(fig)

make_gif(r"C:\Users\LOCCO\Project_Curie\labmeeting\temp_evol_A2Z1CCD70150\temp_evol_density_x",
         r"C:\Users\LOCCO\Project_Curie\labmeeting\temp_evol_A2Z1CCD70150\temp_evol_density_x\density_x.gif")


# In[ ]:


y_center = 8000
y_thickness = 500  # ex: garde |y| < 2.5

bins = np.arange(frame_treated.min(), frame_treated.max() + 10000,10000)
# Bins en x fixes pour toutes les frames (cohérence du film)
x_bins = np.linspace(X_treated[:,1].min(), X_treated[:,1].max(), 500)

for i in range(len(bins)):
    mask_temp = (frame_treated < bins[i] + 1000)
    x_t, y_t, z_t =  X_treated[0,mask_temp],X_treated[1,mask_temp],X_treated[2,mask_temp]

    # Masque pour la tranche centrale en y et z


    mask_slice = (np.abs(y_t - y_center) <y_thickness/2) 
    x_slice = x_t[mask_slice]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_slice, bins=y_bins)
    ax.set_xlim(9500, 13000)
    ax.set_ylim(0, 100)  # ajuste si tu veux un ylim fixe pour toutes les frames
    ax.set_xlabel("x")
    ax.set_ylabel("Densité (comptes)")
    ax.set_title(f"Densité vs x | t = {int(bins[i]):04d}")


# In[ ]:


from scipy.stats import gaussian_kde

y_center = 8000
y_thickness = 500  # ex: garde |y| < 2.5

bins = np.arange(frame_treated.min(), frame_treated.max() + 10000, 10000)
# Bins en x fixes pour toutes les frames (cohérence du film)
x_bins = np.linspace(X_treated[:,1].min(), X_treated[:,1].max(), 500)

for i in range(len(bins)):
    mask_temp = (frame_treated < bins[i] + 1000)
    x_t, y_t, z_t = X_treated[0, mask_temp], X_treated[1, mask_temp], X_treated[2, mask_temp]

    # Masque pour la tranche centrale en y et z
    mask_slice = (np.abs(y_t - y_center) < y_thickness / 2)
    x_slice = x_t[mask_slice]

    fig, ax = plt.subplots(figsize=(8, 5))

    if len(x_slice) > 1:  # gaussian_kde nécessite au moins 2 points
        kde = gaussian_kde(x_slice)
        x_eval = np.linspace(11000, 15000, 500)
        ax.plot(x_eval, kde(x_eval))

    ax.set_xlim(11000, 15000)
    ax.set_ylim(0, 0.01)  # ajuste l'échelle, la KDE est normalisée en aire (pas en comptes)
    ax.set_xlabel("x")
    ax.set_ylabel("Densité (KDE)")
    ax.set_title(f"Densité vs x | t = {int(bins[i]):04d}")


# In[7]:


from scipy.stats import gaussian_kde

y_center = 8000
y_thickness = 500  # ex: garde |y| < 2.5

bins = np.arange(frame_treated.min(), frame_treated.max() + 5000, 5000)
x_bins = np.linspace(X_treated[:,1].min(), X_treated[:,1].max(), 500)

fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.viridis(np.linspace(0, 1, len(bins)))
x_eval = np.linspace(11000, 16000, 500)

for i in range(len(bins)):
    mask_temp = (frame_treated < bins[i] + 1000)
    x_t, y_t, z_t = X_treated[0, mask_temp], X_treated[1, mask_temp], X_treated[2, mask_temp]

    mask_slice = (np.abs(y_t - y_center) < y_thickness / 2)
    x_slice = x_t[mask_slice]

    if len(x_slice) > 1:
        kde = gaussian_kde(x_slice)
        ax.plot(x_eval, kde(x_eval), color=colors[i], label=f"t = {int(bins[i])}")

ax.set_xlim(11000, 16000)
ax.set_xlabel("x")
ax.set_ylabel("Densité (KDE)")
ax.set_title("Évolution de la densité vs x")

# Colorbar pour représenter le temps plutôt qu'une légende encombrée
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=bins.min(), vmax=bins.max()))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("t (frame)")

plt.show()


# In[11]:


y_center = 8000
y_thickness = 500  # ex: garde |y| < 2.5

bins = np.arange(frame_treated.min(), frame_treated.max() + 5000, 5000)
x_bins = np.linspace(11800, 15300, 100)  # bins fixes pour toutes les frames

fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.viridis(np.linspace(0, 1, len(bins)))

for i in range(len(bins)):
    mask_temp = (frame_treated < bins[i] + 1000)
    x_t, y_t, z_t = X_treated[0, mask_temp], X_treated[1, mask_temp], X_treated[2, mask_temp]

    mask_slice = (np.abs(y_t - y_center) < y_thickness / 2)
    x_slice = x_t[mask_slice]

    if len(x_slice) > 0:
        ax.hist(x_slice, bins=x_bins, histtype='step', color=colors[i], lw=1.5)

ax.set_xlim(11800, 15300)
ax.set_xlabel("x")
ax.set_ylabel("Comptes")
#ax.set_title("Évolution temporelle de la distribution de localisations dans la tranche ")

sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=bins.min(), vmax=bins.max()))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("t (frame)")

plt.show()

