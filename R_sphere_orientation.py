#!/usr/bin/env python
# coding: utf-8

# In[70]:


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
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
import numpy as np
from tqdm import tqdm
import matplotlib.colors as mcolors
import matplotlib as mpl
import joblib
from sklearn.cluster import cluster_optics_xi
from scipy.signal import savgol_filter


# In[71]:


#Import data treated
data_treated = pd.read_csv(r'C:\Users\LOCCO\Project_Curie\labmeeting\datatreatedA2Z0CCD70150.csv',sep=',')
frame_treated = data_treated['frame'].values
X_treated = (data_treated[['x [nm]', 'y [nm]', 'z [nm]']].values)
rho_treated = data_treated['rho'].values
delta_treated = data_treated['delta'].values


# In[7]:


float(0) in rho_treated


# In[9]:


os.makedirs(r'C:\Users\LOCCO\Project_Curie\labmeeting\orientation_evol_A2Z0CCD70150', exist_ok=True)


# In[72]:


n_pts = len(X_treated)
r = 50
chunk_size = 500

n_evenements_50 = np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\labmeeting\orientation_evol_A2Z0CCD70150\n_evenements_50.npy',
    mode='w+', dtype=np.float32, shape=(n_pts,)
)
mean_orientation_50 = np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\labmeeting\orientation_evol_A2Z0CCD70150\mean_orientation_50.npy',
    mode='w+', dtype=np.float32, shape=(n_pts,)
)

std_orientation_50 = np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\labmeeting\orientation_evol_A2Z0CCD70150\std_orientation_50.npy',
    mode='w+', dtype=np.float32, shape=(n_pts,)
)


for chunk_start in tqdm(range(0, n_pts, chunk_size), desc="Chunks"):
    chunk_end = min(chunk_start + chunk_size, n_pts)
    X_chunk   = X_treated[chunk_start:chunk_end]

    # dist shape: (chunk_size, n_pts)
    dist_chunk = cdist(X_chunk, X_treated).astype(np.float32)
    mask_chunk = dist_chunk <= r  # (chunk_size, n_pts)

    n_evenements_50[chunk_start:chunk_end] = mask_chunk.sum(axis=1)

    # Calcul mean/std par ligne avec masque
    for i, mask_row in enumerate(mask_chunk):
        vals = rho_treated[mask_row]  # voisins de X_chunk[i] dans X_treated
        if len(vals) > 1:
            mean_orientation_50[chunk_start + i] = vals.mean()
            std_orientation_50[chunk_start + i]  = vals.std()
        else:
            mean_orientation_50[chunk_start + i] = np.nan
            std_orientation_50[chunk_start + i]  = np.nan

n_evenements_50.flush()
mean_orientation_50.flush()
std_orientation_50.flush()


# In[ ]:


r = 50
np.save(r'C:\Users\LOCCO\Project_Curie\labmeeting\orientation_evol_A2Z0CCD70150\r.npy', r)


# In[73]:


clust = joblib.load(r'C:\Users\LOCCO\Project_Curie\labmeeting\optics_A2Z0CCD70150\minpts25\minpts25.pkl')
labels, _ = cluster_optics_xi(
        reachability=clust.reachability_,
        predecessor=clust.predecessor_,
        ordering=clust.ordering_,
        min_samples=25,
        xi=0.05
    )
ordering = clust.ordering_  # indices des points dans l'ordre de propagation OPTICS
reachability = clust.reachability_[clust.ordering_]
space = np.arange(len(clust.ordering_))


# In[59]:


n_evenements_60 = np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\labmeeting\orientation_evol_A2Z0CCD70150\n_evenements_60.npy',
    mode='r'
)

mean_orientation_60 =  np.lib.format.open_memmap(
   r'C:\Users\LOCCO\Project_Curie\labmeeting\orientation_evol_A2Z0CCD70150\mean_orientation_60.npy',
    mode='r'
)

std_orientation_60 = np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\labmeeting\orientation_evol_A2Z0CCD70150\std_orientation_60.npy',
    mode='r'
)

n_evenements_100 = np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\labmeeting\orientation_evol_A2Z0CCD70150\n_evenements_100.npy',
    mode='r'
)

mean_orientation_100 =  np.lib.format.open_memmap(
   r'C:\Users\LOCCO\Project_Curie\labmeeting\orientation_evol_A2Z0CCD70150\mean_orientation_100.npy',
    mode='r'
)

std_orientation_100 = np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\labmeeting\orientation_evol_A2Z0CCD70150\std_orientation_100.npy',
    mode='r'
)


# In[61]:


mean_orientation = mean_orientation_60[ordering]
std_orientation = std_orientation_60[ordering]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(18, 6))

ax1.plot(reachability, color='black')
ax1.set_ylabel("Distance d'accéssibilité")


ax2.plot(mean_orientation, color='red')
ax2.set_ylabel(f"Orientation (°)")

ax3.plot(std_orientation, color ='blue')
ax3.set_ylabel(f"Ecart-type (°)")

plt.title(f'Orientation (moyenne, écart type) des voisins compris dans une sphère de r=60 nm')
plt.tight_layout()
plt.show()


# In[36]:


x = np.linspace(0, 180, 500)

std_orientation_clean = std_orientation[~np.isnan(std_orientation)]
kde = gaussian_kde(std_orientation_clean)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, kde(x))
ax.set_xlabel("Points")
ax.set_ylabel("KDE de std d'orientation")
ax.set_xlim(0, std_orientation_clean.max())
ax.set_ylim(0, kde(x).max())
ax.set_title(f"KDE std variation ")
ax.legend()
plt.show()


# In[38]:


x = np.linspace(0, 180, 500)

mean_orientation_clean = mean_orientation[~np.isnan(std_orientation)]
kde = gaussian_kde(mean_orientation_clean)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, kde(x))
ax.set_xlabel("Points")
ax.set_ylabel("KDE de mean d'orientation")
ax.set_xlim(0, mean_orientation_clean.max())
ax.set_ylim(0, kde(x).max())
ax.set_title(f"KDE mean variation ")
ax.legend()
plt.show()


# In[39]:


fig, ax = plt.subplots(figsize=(8, 5))
n_bins = np.linspace(0, 180, 150)  # bins fixes pour toutes les courbes

counts, edges = np.histogram(std_orientation_clean, bins=n_bins, density=False)
counts_norm = counts / counts.max()
centers = 0.5 * (edges[:-1] + edges[1:])

ax.step(centers, counts_norm, where='mid', color='red', lw=2,
        label=f"r={r}")
plt.show()


# In[62]:


fig, ax = plt.subplots(figsize=(8, 5))
n_bins = np.linspace(0, 180, 150)  # bins fixes pour toutes les courbes
'''std_orientation_clean = std_orientation_100[~np.isnan(std_orientation_100)]
counts, edges = np.histogram(std_orientation_clean, bins=n_bins, density=False)
counts_norm = counts / counts.max()
centers = 0.5 * (edges[:-1] + edges[1:])

ax.step(centers, counts_norm, where='mid', color='red', lw=2,
        label=f"r=100")'''

std_orientation_clean = std_orientation_60[~np.isnan(std_orientation_60)]
counts, edges = np.histogram(std_orientation_clean, bins=n_bins, density=False)
counts_norm = counts / counts.max()
centers = 0.5 * (edges[:-1] + edges[1:])
ax.step(centers, counts_norm, where='mid', color='red', lw=2,
        label=f"Ecart-type (°)")

mean_orientation_clean = mean_orientation_60[~np.isnan(mean_orientation_60)]
counts, edges = np.histogram(mean_orientation_clean, bins=n_bins, density=False)
counts_norm = counts / counts.max()
centers = 0.5 * (edges[:-1] + edges[1:])

ax.step(centers, counts_norm, where='mid', color='green', lw=2,
        label=f"Moyenne (°)")
ax.set_title(f"Histogrammes de l'orientation (moyenne, écart type) des voisins compris dans une sphère de r=60 nm")
ax.legend()
plt.show()


# In[58]:


fig, ax = plt.subplots(figsize=(8, 5))
n_bins = np.linspace(0, 180, 150)  # bins fixes pour toutes les courbes
mean_orientation_clean = mean_orientation_100[~np.isnan(mean_orientation_100)]
counts, edges = np.histogram(mean_orientation_clean, bins=n_bins, density=False)
counts_norm = counts / counts.max()
centers = 0.5 * (edges[:-1] + edges[1:])

ax.step(centers, counts_norm, where='mid', color='red', lw=2,
        label=f"r=100")

mean_orientation_clean = mean_orientation_60[~np.isnan(mean_orientation_60)]
counts, edges = np.histogram(mean_orientation_clean, bins=n_bins, density=False)
counts_norm = counts / counts.max()
centers = 0.5 * (edges[:-1] + edges[1:])

ax.step(centers, counts_norm, where='mid', color='green', lw=2,
        label=f"r=60")
ax.set_title(f"histogramme mean orientation")
ax.legend()
plt.show()


# In[48]:


fig = plt.figure(figsize=(14, 6))

# Main scatter
ax_scatter = fig.add_axes([0.05, 0.1, 0.7, 0.85])  # [left, bottom, width, height]
norm = matplotlib.colors.Normalize(vmin=0, vmax=50)
sc = ax_scatter.scatter(X_treated[:,0], X_treated[:,1],
                        c=std_orientation_60, cmap=matplotlib.colormaps['inferno'],
                        norm=norm, s=0.01)
ax_scatter.set_aspect('equal')
ax_scatter.set_title('Localizations colored by ρ')
ax_cbar = fig.add_axes([0.78, 0.1, 0.02, 0.85])
fig.colorbar(sc, cax=ax_cbar, label='std orientation')
plt.show()


# In[74]:


mean_orientation_50_simili = np.zeros(len(n_evenements_50))
std_orientation_50_simili = np.zeros(len(n_evenements_50))

for i in range(len(n_evenements_60)):
    n = int(n_evenements_60[i])
    if n > 1:
        orientations = np.random.uniform(0, 180, size=n).round(13)
        mean_orientation_50_simili[i] = orientations.mean()
        std_orientation_50_simili[i] = orientations.std()
    else:
        mean_orientation_50_simili[i] = np.nan
        std_orientation_50_simili[i] = np.nan


# In[75]:


np.save(r'C:\Users\LOCCO\Project_Curie\labmeeting\orientation_evol_A2Z0CCD70150\mean_orientation_50_simili.npy', mean_orientation_50_simili)
np.save(r'C:\Users\LOCCO\Project_Curie\labmeeting\orientation_evol_A2Z0CCD70150\std_orientation_50_simili.npy', std_orientation_50_simili)


# In[77]:


fig, ax = plt.subplots(figsize=(8, 5))
n_bins = np.linspace(0, 180, 150)  # bins fixes pour toutes les courbes


std_orientation_clean = std_orientation_50_simili[~np.isnan(std_orientation_50_simili)]
counts, edges = np.histogram(std_orientation_clean, bins=n_bins, density=False)
counts_norm = counts / counts.max()
centers = 0.5 * (edges[:-1] + edges[1:])
ax.step(centers, counts_norm, where='mid', color='gray', lw=1,
        label=f"Ecart-type simulé uniforme (°)")

mean_orientation_clean = mean_orientation_50_simili[~np.isnan(mean_orientation_50_simili)]
counts, edges = np.histogram(mean_orientation_clean, bins=n_bins, density=False)
counts_norm = counts / counts.max()
centers = 0.5 * (edges[:-1] + edges[1:])
ax.step(centers, counts_norm, where='mid', color='black', lw=1,
        label=f"Moyenne simulé uniforme (°)")

std_orientation_clean = std_orientation_50[~np.isnan(std_orientation_50)]
counts, edges = np.histogram(std_orientation_clean, bins=n_bins, density=False)
counts_norm = counts / counts.max()
centers = 0.5 * (edges[:-1] + edges[1:])
ax.step(centers, counts_norm, where='mid', color='blue', lw=2,
        label=f"Ecart-type data d'origine(°)")

mean_orientation_clean = mean_orientation_50[~np.isnan(mean_orientation_50)]
counts, edges = np.histogram(mean_orientation_clean, bins=n_bins, density=False)
counts_norm = counts / counts.max()
centers = 0.5 * (edges[:-1] + edges[1:])
ax.step(centers, counts_norm, where='mid', color='red', lw=2,
        label=f"Moyenne data d'origine(°)")

ax.set_title(f"Histogrammes de l'orientation (moyenne, écart type) des voisins compris dans une sphère de r=50 nm")
ax.legend()
plt.show()

