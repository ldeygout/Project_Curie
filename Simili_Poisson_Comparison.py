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
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
import numpy as np
from tqdm import tqdm
import matplotlib.colors as mcolors
import matplotlib as mpl
import joblib
from sklearn.cluster import cluster_optics_xi
from scipy.signal import savgol_filter


# In[2]:


#Compared data set
#Import data
data_treated = pd.read_csv(r'C:\Users\LOCCO\Project_Curie\labmeeting\datatreatedA2Z0CCD70150.csv',sep=',')
frame_treated = data_treated['frame'].values
X_treated = (data_treated[['x [nm]', 'y [nm]', 'z [nm]']].values)
rho_treated = data_treated['rho'].values
delta_treated = data_treated['delta'].values

std = pd.read_csv(r'C:\Users\LOCCO\Project_Curie\labmeeting\stdA2Z0CCD70150.csv',sep=',')
stdx= std['stdx'].values
stdy = std['stdy'].values
stdz = std['stdz'].values
stdrho = std['stdrho'].values


# In[3]:


#Caracteristiques
n_pts = len(frame_treated)
mean_stdx = np.mean(stdx)
mean_stdy = np.mean(stdy)
mean_stdz = np.mean(stdz)
mean_stdrho = np.mean(stdrho)
mean_x = np.mean(X_treated[:,0])
mean_y = np.mean(X_treated[:,1])
mean_z = np.mean(X_treated[:,2])

#Bornes
min_x, max_x = X_treated[:,0].min(), X_treated[:,0].max()
min_y, max_y = X_treated[:,1].min(), X_treated[:,1].max()
min_z, max_z = X_treated[:,2].min(), X_treated[:,2].max()


# In[4]:


#Without Noise
rng = np.random.default_rng(0) #Reproductible random data generator

simili_X = np.array([
    rng.poisson(mean_x, n_pts),
    rng.poisson(mean_y, n_pts),
    rng.poisson(mean_z, n_pts)]).T

simili_rho = rng.uniform(0,180, n_pts )


# In[5]:


os.makedirs(r'C:\Users\LOCCO\Project_Curie\labmeeting\simili_poiss_A2Z0CCD70150', exist_ok=True)
os.makedirs(r'C:\Users\LOCCO\Project_Curie\labmeeting\simili_poiss_A2Z0CCD70150\minpts25', exist_ok=True)


# In[6]:


clust = OPTICS(min_samples=25, max_eps = np.inf, cluster_method='xi', xi=0.01)
clust.fit(simili_X)

joblib.dump(clust, r'C:\Users\LOCCO\Project_Curie\labmeeting\simili_poiss_A2Z0CCD70150\minpts25\minpts25.pkl')


# In[ ]:


clust = joblib.load(r'C:\Users\LOCCO\Project_Curie\labmeeting\simili_poiss_A2Z0CCD70150\minpts25\minpts25.pkl')


# In[7]:


labels, _ = cluster_optics_xi(
        reachability=clust.reachability_,
        predecessor=clust.predecessor_,
        ordering=clust.ordering_,
        min_samples=25,
        xi=0.005
    )
reachability = clust.reachability_[clust.ordering_]
space = np.arange(len(clust.ordering_))

plt.figure(figsize=(12, 5))
plt.plot(space, reachability, '+', 
                color="black", alpha=0.7)
plt.ylabel('Eps distance')
plt.xlabel('Arranged point index')
plt.title('Reachability Plot MinPts = 25')

plt.show()


# In[8]:


for xi in [0.01]:
    labels, _ = cluster_optics_xi(
        reachability=clust.reachability_,
        predecessor=clust.predecessor_,
        ordering=clust.ordering_,
        min_samples=25,
        xi=xi
    )
    mask_reach = clust.reachability_ <= 200
    labels_filtered = labels.copy()
    labels_filtered[~mask_reach] = -1
    labels_ordered = labels_filtered[clust.ordering_]
    n_clusters = len(np.unique(labels)) - 1
    print(f"xi={xi} → {n_clusters} clusters, {(labels==-1).sum()} noise points")

    colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(labels))))

    color_map = {label: 'grey' if label == -1 else plt.cm.hsv((label * 37) % n_clusters / n_clusters)
             for label in np.unique(labels)}
    plt.figure(figsize=(12, 5))
    for label in np.unique(labels_ordered):
        mask = (labels_ordered == label)
        lbl = f"Bruit" if label == -1 else f"Cluster {label}"
        plt.plot(space[mask], reachability[mask], '+', 
                color=color_map[label], label=lbl, alpha=0.7)


    plt.ylabel('Eps distance')
    plt.xlabel('Arranged point index')
    plt.title(f"Colored reachability plot, (xi={xi}) — {n_clusters} clusters")
    #plt.legend()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 8))

    for label, color in zip(np.unique(labels), colors):
        mask = labels == label
        if label == -1:
            ax.scatter(simili_X[mask, 0], simili_X[mask, 1], c='grey', s=0.1, alpha=0.3)
        else:
            ax.scatter(simili_X[mask, 0], simili_X[mask, 1], color=color_map[label], s=0.1)

    ax.set_aspect('equal')
    ax.set_title(f"OPTICS clusters (xi={xi}) — {n_clusters} clusters")
    plt.show()

