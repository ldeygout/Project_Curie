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


# In[3]:


#Import data treated
data_treated = pd.read_csv(r'C:\Users\LOCCO\Project_Curie\labmeeting\datatreatedA2Z0CCD70150.csv',sep=',')
frame_treated = data_treated['frame'].values
X_treated = (data_treated[['x [nm]', 'y [nm]', 'z [nm]']].values)
rho_treated = data_treated['rho'].values
delta_treated = data_treated['delta'].values


# In[4]:


os.makedirs(r'C:\Users\LOCCO\Project_Curie\labmeeting\optics_A2Z0CCD70150\minpts25', exist_ok=True)


# In[77]:


clust = OPTICS(min_samples=25, max_eps = np.inf, cluster_method='xi', xi=0.01)
clust.fit(X_treated)

joblib.dump(clust, r'C:\Users\LOCCO\Project_Curie\labmeeting\optics_A2Z0CCD70150\minpts25\minpts25.pkl')


# In[2]:


clust = joblib.load(r'C:\Users\LOCCO\Project_Curie\labmeeting\optics_A2Z0CCD70150\minpts25\minpts25.pkl')


# In[3]:


labels, _ = cluster_optics_xi(
        reachability=clust.reachability_,
        predecessor=clust.predecessor_,
        ordering=clust.ordering_,
        min_samples=25,
        xi=0.05
    )
reachability = clust.reachability_[clust.ordering_]
space = np.arange(len(clust.ordering_))

plt.figure(figsize=(12, 5))
plt.plot(space, reachability, '+', 
                color="black", alpha=0.7)
plt.ylabel('Rayons eps')
plt.xlabel('Points ordonnées')
plt.title('Diagramme de Reachability, MinPts = 25')

plt.show()


# In[18]:


reachability_noinfnonan = reachability[(~np.isnan(reachability))&(~np.isinf(reachability))]
space_noinfnonan = np.arange(len(reachability_noinfnonan))

plt.figure(figsize=(12, 5))
plt.plot(space, reachability_noinfnonan, '+', 
                color="black", alpha=0.7)
plt.ylabel('Eps distance')
plt.xlabel('Arranged point index')
plt.title('Reachability Plot MinPts = 25')
plt.show()


# In[27]:


window_size = 750
poly_order = 3
y_smooth = savgol_filter(reachability_noinfnonan, window_size, poly_order)

plt.figure(figsize=(12, 5))
plt.plot(space, reachability, '+', 
                color="black", alpha=0.7)
plt.plot(space_noinfnonan, y_smooth, 
                color="red", alpha=0.7, label='Smoothed Signal')

plt.ylabel('Eps distance')
plt.xlabel('Arranged point index')
plt.title('Reachability Plot MinPts = 25')
plt.show()


# In[ ]:


for xi in [0.3, 0.005]:
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
            ax.scatter(X_treated[mask, 0], X_treated[mask, 1], c='grey', s=0.1, alpha=0.3)
        else:
            ax.scatter(X_treated[mask, 0], X_treated[mask, 1], color=color_map[label], s=0.1)

    ax.set_aspect('equal')
    ax.set_title(f"OPTICS clusters (xi={xi}) — {n_clusters} clusters")
    plt.show()


# In[80]:


os.makedirs(r'C:\Users\LOCCO\Project_Curie\labmeeting\optics_A2Z0CCD70150\minpts25\xi005', exist_ok=True)


# In[81]:


labels, _ = cluster_optics_xi(
    reachability=clust.reachability_,
    predecessor=clust.predecessor_,
    ordering=clust.ordering_,
    min_samples=25,
    xi=0.005
)
labels_filtered = labels.copy()
mask_reach = (clust.reachability_ <= 200)
labels_filtered[~mask_reach] = -1
unique_labels = np.unique(labels_filtered)
unique_labels = unique_labels[unique_labels != -1]  # exclure le bruit

for label in unique_labels:
    mask = labels_filtered == label
    rho_cluster = rho_treated[mask]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='polar')
    rho_wrapped = np.concatenate([rho_cluster, rho_cluster + 180]) % 360
    theta = np.deg2rad(rho_wrapped)
    n, bins_rho, patches = ax.hist(theta, bins=36, range=(0, 2*np.pi))
    ax.set_title(f'Cluster {label} — {mask.sum()} points')
    plt.tight_layout()
    plt.savefig(
        os.path.join(r"C:\Users\LOCCO\Project_Curie\labmeeting\optics_A2Z0CCD70150\minpts25\xi005", f"cluster_{label:04d}.png"),
        dpi=100
    )
    plt.close(fig)


# In[25]:


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


# In[82]:


make_gif(r"C:\Users\LOCCO\Project_Curie\labmeeting\optics_A2Z0CCD70150\minpts25\xi005",r"C:\Users\LOCCO\Project_Curie\labmeeting\optics_A2Z0CCD70150\minpts25\xi005\animation_rho.gif")


# In[5]:


#
os.makedirs(r'C:\Users\LOCCO\Project_Curie\labmeeting\shuff_optics_A2Z0CCD70150\minpts25', exist_ok=True)

rng = np.random.default_rng(0)
n_shuffles = 5


for i in range(n_shuffles):
    shuffled_X = X_treated.copy()
    for col in range(shuffled_X.shape[1]):
        rng.shuffle(shuffled_X[:, col])  # independent shuffle per column

    clust_shuf = OPTICS(min_samples=25, max_eps = np.inf, cluster_method='xi', xi=0.01)
    clust_shuf.fit(shuffled_X)
    joblib.dump(clust_shuf, rf'C:\Users\LOCCO\Project_Curie\labmeeting\shuff_optics_A2Z0CCD70150\minpts25\shuff{i}.pkl')


# In[8]:


for i in range(n_shuffles):
    clust = joblib.load(rf'C:\Users\LOCCO\Project_Curie\labmeeting\shuff_optics_A2Z0CCD70150\minpts25\shuff{i}.pkl')
    labels, _ = cluster_optics_xi(
            reachability=clust.reachability_,
            predecessor=clust.predecessor_,
            ordering=clust.ordering_,
            min_samples=25,
            xi=0.05
        )
    reachability = clust.reachability_[clust.ordering_]
    space = np.arange(len(clust.ordering_))

    plt.figure(figsize=(12, 5))
    plt.plot(space, reachability, '+', 
                    color="black", alpha=0.7)
    plt.ylabel('Eps distance')
    plt.xlabel('Arranged point index')
    plt.title(f'Reachability Plot shuffled {i} MinPts = 25')

    plt.show()



# In[18]:


clust = joblib.load(r'C:\Users\LOCCO\Project_Curie\labmeeting\simili_A2Z0CCD70150\minpts25\minpts25.pkl')
labels, _ = cluster_optics_xi(
        reachability=clust.reachability_,
        predecessor=clust.predecessor_,
        ordering=clust.ordering_,
        min_samples=25,
        xi=0.05
    )
reachability = clust.reachability_[clust.ordering_]
space = np.arange(len(clust.ordering_))

plt.figure(figsize=(12, 5))
plt.plot(space, reachability, '+', 
                color="black", alpha=0.7)
plt.ylabel('Rayons eps')
plt.xlabel('Points ordonnées')
plt.title('Diagramme de Reachability pour distribution Gaussienne, MinPts = 25')

plt.show()

