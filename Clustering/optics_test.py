#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 23:37:28 2026

@author: Lola
"""
	
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
import sklearn.datasets._samples_generator as sg
from tqdm import tqdm
import time
from matplotlib.lines import Line2D


#%% 1D, No noise, Gaussien, same size for all 3 clusters

rng = np.random.default_rng(0) #Reproductible random data generator

 #Commençons par les trois clusters ont la même taille
n1 = 500
n2 = 500
n3 = 500

# cluster 1 (serré)
c1 = rng.normal(loc=0.0, scale=0.2, size=n1)
# cluster 2 (medium)
c2 = rng.normal(loc=5.0, scale=0.3, size=n2)
# cluster 3 (large)
c3 = rng.normal(loc=10.0, scale=0.3, size=n3)

X = np.concatenate([c1, c2, c3]).reshape(-1, 1)  #Optics attend un array 2D
Y_GT = np.concatenate([np.zeros(n1), np.ones(n2), np.full(n3, 2)]) #Étiquettes

clust = OPTICS(min_samples=10)
clust.fit_predict(X)
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]
space = np.arange(len(X))

# Noise boundary = 98th percentile des points non-infinis
finite_reach  = reachability[np.isfinite(reachability)]
noise_threshold = np.percentile(finite_reach, 98)

# Couleurs par cluster (-1 = bruit)
colors = {-1: 'gray', 0: 'steelblue', 1: 'darkorange', 2: 'seagreen'}
label_names = {-1: 'Bruit', 0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3'}

fig, ax = plt.subplots(figsize=(10, 5))

# Scatter coloré par cluster
for lbl, color in colors.items():
    mask = labels == lbl
    if mask.any():
        ax.scatter(space[mask], reachability[mask],
                   marker='+', s=20, c=color,
                   label=label_names[lbl], zorder=2)

# Ligne de seuil bruit
ax.axhline(y=noise_threshold, color='red', linestyle='--',
           linewidth=1.2, label=f'Seuil bruit ({noise_threshold:.2f})', zorder=3)

ax.set_xlabel('Arranged point index')
ax.set_ylabel('Eps distance')
ax.legend(loc='upper left')
plt.title(f"Reachability plot pour 3 clusters de taile {n1}, {n2} et {n3}")
plt.tight_layout()
plt.show()

#%%
from matplotlib.lines import Line2D

xi = 0.5
maxeps = np.inf
rng = np.random.default_rng(42) #Rendre le run de cellule reproductible
n1, n2, n3 = rng.integers(3, 501), rng.integers(3, 501), rng.integers(3, 501)

c1 = rng.normal(loc=0.0, scale=0.2, size=n1)
c2 = rng.normal(loc=5.0, scale=0.3, size=n2)
c3 = rng.normal(loc=10.0, scale=0.3, size=n3)

X    = np.concatenate([c1, c2, c3]).reshape(-1, 1)
Y_GT = np.concatenate([np.zeros(n1), np.ones(n2), np.full(n3, 2)])

clust = OPTICS(min_samples=10, xi = xi, max_eps = maxeps)
clust.fit_predict(X)

reachability = clust.reachability_[clust.ordering_]
labels       = clust.labels_[clust.ordering_]
space        = np.arange(len(X))

finite_reach    = reachability[np.isfinite(reachability)]
noise_threshold = np.percentile(finite_reach, 98)

colors = {-1: 'gray', 0: 'steelblue', 1: 'darkorange', 2: 'seagreen'}

def cluster_meta(lbl):
    if lbl == -1:
        mask = clust.labels_ == -1
        return f"n={mask.sum()}"
    mask = clust.labels_ == lbl
    vals = X[mask, 0]
    return f"n={len(vals)}, µ={vals.mean():.2f}, σ={vals.std():.2f}"

label_names = {
    -1: f"Bruit — {cluster_meta(-1)}",
     0: f"Cluster 1 — {cluster_meta(0)}",
     1: f"Cluster 2 — {cluster_meta(1)}",
     2: f"Cluster 3 — {cluster_meta(2)}",
}

def gt_meta(lbl):
    mask = Y_GT == lbl
    vals = X[mask, 0]
    return f"n={len(vals)}, µ={vals.mean():.2f}, σ={vals.std():.2f}"

# --- Figure (doit être créée AVANT tout appel à ax) ---
fig, ax = plt.subplots(figsize=(10, 5))

for lbl, color in colors.items():
    mask = labels == lbl
    if mask.any():
        ax.scatter(space[mask], reachability[mask],
                   marker='+', s=20, c=color,
                   label=label_names[lbl], zorder=2)

ax.axhline(y=noise_threshold, color='red', linestyle='--',
           linewidth=1.2, label=f'Seuil bruit ({noise_threshold:.2f})', zorder=3)

ax.set_xlabel('Arranged point index')
ax.set_ylabel('Eps distance')
plt.title(f"Reachability plot xi = {xi} et max_eps = {maxeps}")

# Légende 1 — OPTICS
leg1 = ax.legend(loc='upper left', title='OPTICS labels')

# Légende 2 — Ground truth
gt_handles = [
    Line2D([0], [0], marker='+', color='w', markerfacecolor='steelblue',
           markeredgecolor='steelblue', markersize=8,
           label=f"GT Cluster 1 — {gt_meta(0)}"),
    Line2D([0], [0], marker='+', color='w', markerfacecolor='darkorange',
           markeredgecolor='darkorange', markersize=8,
           label=f"GT Cluster 2 — {gt_meta(1)}"),
    Line2D([0], [0], marker='+', color='w', markerfacecolor='seagreen',
           markeredgecolor='seagreen', markersize=8,
           label=f"GT Cluster 3 — {gt_meta(2)}"),
]
leg2 = ax.legend(handles=gt_handles, loc='upper right', title='Ground truth - Clusters 1D Gaussiens')
ax.add_artist(leg1)  # Remettre leg1 après le second .legend()

plt.tight_layout()
plt.show()
        
#%% 1D, No noise, Gaussien, different sizes
rng = np.random.default_rng(0) #Reproductible random data generator
n1, n2, n3 = rng.integers(3, 501), rng.integers(3, 501), rng.integers(3, 501)

X = np.concatenate([c1, c2, c3]).reshape(-1, 1)  #Optics attend un array 2D
Y_GT = np.concatenate([np.zeros(n1), np.ones(n2), np.full(n3, 2)]) #Étiquettes

clust = OPTICS(min_samples=10)
clust.fit_predict(X)
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]
space = np.arange(len(X))

plt.plot(space,reachability,'+' )
plt.ylabel('Eps distance')
plt.xlabel('Arranged point index')


# Étapes avec progression manuelle
steps = [ ("Génération des données", lambda: None), ("Concaténation", lambda: None), ("fit OPTICS", lambda: clust.fit(X)), ("Extraction reachability", lambda: None),]

with tqdm(total=len(steps), desc="Pipeline OPTICS") as pbar:
    for name, fn in steps:
        pbar.set_description(name)
        fn()
        pbar.update(1)
#%%
idx = np.arange(len(X))
rng.shuffle(idx)

X = X[idx]
y_true = y_true[idx]



# Noise boundary = 98th percentile des points non-infinis
finite_reach  = reachability[np.isfinite(reachability)]
noise_threshold = np.percentile(finite_reach, 98)

# Couleurs par cluster (-1 = bruit)
colors = {-1: 'gray', 0: 'steelblue', 1: 'darkorange', 2: 'seagreen'}
label_names = {-1: 'Bruit', 0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3'}

fig, ax = plt.subplots(figsize=(10, 5))

# Scatter coloré par cluster
for lbl, color in colors.items():
    mask = labels == lbl
    if mask.any():
        ax.scatter(space[mask], reachability[mask],
                   marker='+', s=20, c=color,
                   label=label_names[lbl], zorder=2)

# Ligne de seuil bruit
ax.axhline(y=noise_threshold, color='red', linestyle='--',
           linewidth=1.2, label=f'Seuil bruit ({noise_threshold:.2f})', zorder=3)

ax.set_xlabel('Arranged point index')
ax.set_ylabel('Eps distance')
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()



