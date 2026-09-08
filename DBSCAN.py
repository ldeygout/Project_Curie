#!/usr/bin/env python
# coding: utf-8

# In[20]:


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
import matplotlib.gridspec as gridspecs
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


# In[21]:


#Import data treated
data_treated = pd.read_csv(r'C:\Users\LOCCO\Project_Curie\labmeeting\datatreatedA2Z0CCD70150.csv',sep=',')
frame_treated = data_treated['frame'].values
X_treated = (data_treated[['x [nm]', 'y [nm]', 'z [nm]']].values).T
rho_treated = data_treated['rho'].values
delta_treated = data_treated['delta'].values


# In[6]:


len(frame_treated)


# In[22]:


def plot_k_distance_graph(data_loc, n_test, minpts):
    data_loc = data_loc.T
    colors = plt.cm.tab10(np.linspace(0, 1, n_test+1))
    plt.figure(figsize=(10, 6))
    d = np.array([])
    means = [] 
    for ki in range(n_test+1):
        color = colors[ki+1 % len(colors)]
        neigh = NearestNeighbors(n_neighbors=50*ki+10)
        neigh.fit(data_loc)
        distances, _ = neigh.kneighbors(data_loc)
        distances = np.sort(distances[:, -1])
        d = np.append(d, distances)

        mean_val = distances[:60000].mean()
        means.append(mean_val)
        #plt.plot(distances, marker='o', markersize=3, color=color, label=f'Voisin n°={50*ki+10}, plateau à {mean_val:.2f} nm, tournant à {mean_val*3:.0f} nm, tournant à {mean_val*2:.0f} nm')
        plt.plot(distances, marker='o', markersize=3, color=color, label=f'Voisin n°={50*ki+10}, plateau à {mean_val:.2f} nm, tournant à {mean_val*2:.0f} nm')
        plt.hlines(mean_val, xmin=0, xmax=len(data_loc), color='gray', linestyle='--', lw=1.2)
        threshold_1 = mean_val * 3
        threshold_2 = mean_val * 2
        #idx_cross = np.argmax(distances >= threshold_1)  # premier indice où distances >= threshold
        #if distances[idx_cross] >= threshold_1:  # vérifie qu'on a bien trouvé un croisement
            #plt.scatter(idx_cross, distances[idx_cross], marker='x', color='red', s=200, linewidths=3, zorder=5)
        idx_cross = np.argmax(distances >= threshold_2)  # premier indice où distances >= threshold
        if distances[idx_cross] >= threshold_2:  # vérifie qu'on a bien trouvé un croisement
            plt.scatter(idx_cross, distances[idx_cross], marker='x', color='red', s=200, linewidths=3, zorder=5)



    plt.xlabel('Points ordonnés par distance croissante')
    plt.ylabel('Distance (nm)')
    #plt.title('K-distance Graph')
    plt.grid(True)
    plt.legend()
    plt.show()
    return d


# In[23]:


plot = plot_k_distance_graph(X_treated, 0, 10)


# In[20]:


plt.close("all")


# In[8]:


eps = 190 #nm
k = 10
dbscan_model = DBSCAN(eps=eps, min_samples=k)
labels = dbscan_model.fit_predict(X_treated.T)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f'Number of clusters found: {n_clusters}')
print(f'Number of noise points: {n_noise}')
n_points_cluster1 = np.sum(labels == 0)
print(f'Nombre de points dans le cluster 1: {n_points_cluster1}')


# In[9]:


from matplotlib.colors import ListedColormap, BoundaryNorm
plt.close('all')
plt.rcParams['figure.figsize'] = [12,12]
mask = labels != -1

cluster_labels = np.unique(labels)
cluster_labels = cluster_labels[cluster_labels != -1]  # optionnel
n_clusters = len(cluster_labels)

cmap = plt.cm.get_cmap('hsv', n_clusters)

norm = BoundaryNorm(
    boundaries=np.arange(n_clusters + 1) - 0.5,
    ncolors=n_clusters
)

sc = plt.scatter(
    X_treated[0,mask], X_treated[1,mask],
    c=labels[mask],
    cmap=cmap,
    norm=norm,
    s=0.01
)

# bruit en gris
plt.scatter(
    X_treated[0,~mask], X_treated[1,~mask],
    c='lightgray',
    s=0.01,
    label='Noise'
)
plt.axis('equal')
plt.title(f'Clusters identified by DBSCAN: MinPts = {k}, eps = {eps} nm. n_clusters = {n_clusters}, noise_pts = {n_noise}')
cbar = plt.colorbar(sc, ticks=cluster_labels)
cbar.set_label("Cluster")
cbar.set_ticklabels([f"{k}" for k in cluster_labels])

plt.show()


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

mask_noise = labels == -1
#ax.scatter(X_treated[mask_noise, 0], X_treated[mask_noise, 1], X_treated[mask_noise, 2],
#           c='lightgray', s=2, label='noise')

sc = ax.scatter(X_treated[~mask_noise, 0], X_treated[~mask_noise, 1], X_treated[~mask_noise, 2],
                c=labels[~mask_noise], cmap='tab20', s=5)

ax.set_title(f'DBSCAN: {n_clusters} clusters, {n_noise} noise points (eps={eps})')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.tight_layout()
plt.show()

