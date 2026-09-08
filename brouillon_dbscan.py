#!/usr/bin/env python
# coding: utf-8

# # Post-processing avec DBScan 
# 
# Data utilisé : pre-processing effectué avec algorithme de Louise (CSV) + drift correction avec ImageJ (cross-correlation)
# 

# In[1]:


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


# In[1]:


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


# In[2]:


# Recuperation of data
data = pd.read_csv('image_Pos0_driftcorrected.csv',sep=',')
print(data.columns)

mask = tiff.imread("Mask.tif")    
mask = np.array(mask.transpose()) #Fiji écrit en y,x


# In[3]:


X = data[['x [nm]', 'y [nm]', 'z [nm]']].values
sigma = data[['sigmax [nm]', 'sigmay [nm]', 'sigmaz [nm]']].values
frame = data[['frame']].values
intensity = data['intensity'].values
rho =  data['rho'].values
delta = data['delta'].values

print(X.shape)
print(mask.shape)

'''#Filtrons les points avec une incertitude trop grande, cf data 
mask_sigma = ((sigma[:,0] <= 240) & (sigma[:,1] <= 240) & (sigma[:,2] <= 720))

X       = X[mask_sigma]
sigma   = sigma[mask_sigma]
rho     = rho[mask_sigma]
delta   = delta[mask_sigma]
frame   = frame[mask_sigma]'''


# In[4]:


sigma_norm = sigma/np.sqrt(intensity[:, None])
center = sigma_norm.mean(axis=0)


# In[10]:


mask_vect = (mask[(X[:, 0] / (120/5)).astype(int),(X[:, 1] / (120/5)).astype(int)] > 0)
X_masked = X[mask_vect]
sigma_masked   = sigma[mask_vect]
rho_masked     = rho[mask_vect]
delta_masked   = delta[mask_vect]
frame_masked   = frame[mask_vect]


# In[24]:


#Filtrons les points hors de la cellule
X_masked = []
for i in range(X.shape[0]):
    #Taille de pixel vaut 120 nm
    x , y = int(X[i,0]/25), int(X[i,1]/25)
    #Fiji écrit en y,x
    if mask[x,y] > 0:
        X_masked.append(X[i,:])

X_masked = np.array(X_masked)


# In[24]:


plt.imshow(mask.T)
plt.gca().invert_yaxis()


# In[ ]:


x_pix = (X[:,0] / 120).astype(int)
y_pix = (X[:,1] / 120).astype(int)

#Fiji écrit en y,x
mask_cell = ((mask[y_pix, x_pix] > 0))

X_masked = X[mask_cell]


# In[9]:


print(mask.shape)
print(max(X[:,0]), max(X[:,1]))

int(max(X[:,1]))/120


# In[15]:


#To do once, the kernel dies fast
plt.close('all')
plt.rcParams['figure.figsize'] = [12,12]
hues = rho_masked / 180.0
hsv_colors = np.stack((hues, np.ones_like(hues), np.ones_like(hues)), axis=1)
rgb_colors = hsv_to_rgb(hsv_colors)
plt.scatter(X_masked[:,0], X_masked[:,1], c=rgb_colors, s=0.01)
plt.axis('equal')



# In[ ]:





# In[26]:


#To do once, the kernel dies fast
plt.close('all')
plt.rcParams['figure.figsize'] = [12,12]
'''hues = rho / 180.0
hsv_colors = np.stack((hues, np.ones_like(hues), np.ones_like(hues)), axis=1)
rgb_colors = hsv_to_rgb(hsv_colors)'''
plt.scatter(X_masked[:,0], X_masked[:,1])
plt.axis('equal')


# In[ ]:


plt.close('all')
#ROI à déterminer visualisation 
x_0 = 10464
x_1 = 17688
y_0 = 6384
y_1 = 14064
mask_roi = ((X[:,0] >= x_0) & (X[:,0] <= x_1) &(X[:,1] >= y_0) & (X[:,1] <= y_1))

X       = X[mask_roi]
sigma   = sigma[mask_roi]
rho     = rho[mask_roi]
delta   = delta[mask_roi]
frame   = frame[mask_roi]


# In[ ]:


#Visuel filtré et ROI
plt.close('all')
plt.rcParams['figure.figsize'] = [12,12]
hues = rho / 180.0
hsv_colors = np.stack((hues, np.ones_like(hues), np.ones_like(hues)), axis=1)
rgb_colors = hsv_to_rgb(hsv_colors)
plt.scatter(X_f_ROI[:,0], X_f_ROI[:,1], c=rgb_colors, s=0.01)
plt.axis('equal')



# In[ ]:


#Posoitions des points en fonction du frame
plt.close('all')
vals = frame
sc = plt.scatter(X[:,0], X[:,1], c=vals, cmap='coolwarm', s=0.01)
cb = plt.colorbar(sc)
plt.axis('equal')




# In[11]:


#From https://www.geeksforgeeks.org/machine-learning/implementing-dbscan-algorithm-using-sklearn/

def plot_k_distance_graph(data_loc, n_test, minpts):
    colors = plt.cm.tab10(np.linspace(0, 1, 2*minpts))
    plt.figure(figsize=(10, 6))
    d = np.array([])
    for ki in range(minpts,minpts+n_test):
        color = colors[ki % len(colors)]
        neigh = NearestNeighbors(n_neighbors=ki)
        neigh.fit(data_loc)
        distances, _ = neigh.kneighbors(data_loc)
        distances = np.sort(distances[:, ki-1])
        d = np.append(d, distances)
        plt.plot(distances, marker='o', markersize=3, color=color, label=f'k={ki}')
    plt.xlabel('Points sorted by distance')
    plt.ylabel('k-th nearest neighbor distance (nm)')
    plt.title('K-distance Graph')
    plt.grid(True)
    plt.legend()
    plt.show()
    return d



# In[27]:


# k = MinPts
k= 3
plot = plot_k_distance_graph(X_masked, 3, k)


# In[ ]:


abcisse = np.arange(0, X.shape[0],1)
dy_dx = np.gradient(plot, abcisse)

plt.figure(figsize=(10, 6))
plt.plot(abcisse,plot, marker='o', markersize=3, label=f'k={k}')
plt.plot(abcisse, dy_dx, '-', label=f'Dérivée de la courbe k-distance')
plt.xlabel('Points sorted by distance')
plt.ylabel('k-th nearest neighbor distance (nm)')
plt.title('K-distance Graph')
plt.grid(True)
plt.legend()
plt.show()


# In[30]:


#Definissisons... à l'oeil
plt.close('all')
eps = 28 #nm
k = 5
dbscan_model = DBSCAN(eps=eps, min_samples=k)
labels = dbscan_model.fit_predict(X_masked)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f'Number of clusters found: {n_clusters}')
print(f'Number of noise points: {n_noise}')


# In[ ]:


eps_values = np.arange(10, 51, 1)  # eps de 10 à 50 nm
n_clusters_list = np.zeros(len(eps_values), dtype=int)
n_noise_list = np.zeros(len(eps_values), dtype=int)

for eps in eps_values:
    dbscan_model = DBSCAN(eps=eps, min_samples=k)
    labels = dbscan_model.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    n_clusters_list[eps_values == eps] = n_clusters
    n_noise_list[eps_values == eps] = n_noise




# In[41]:


plt.close('all')
plt.figure(figsize=(10, 6))
plt.plot(eps_values, n_clusters_list, '-o', label='Nombre de clusters')

plt.plot(eps_values[np.argmax(n_clusters_list)], np.max(n_clusters_list), marker='x', markersize=10, color='red', label=f'Max clusters {np.max(n_clusters_list)} pour eps={eps_values[np.argmax(n_clusters_list)]} nm')
plt.xlabel('eps (nm)')
plt.ylabel('Nombre')
plt.title('Évolution du clustering DBSCAN en fonction de eps')
plt.legend()
plt.grid(True)

plt.show()
plt.figure(figsize=(10, 6))
plt.plot(eps_values, n_noise_list, '-o', label='Nombre de points bruit')
plt.plot(eps_values[np.argmax(n_clusters_list)], n_noise_list[np.argmax(n_clusters_list)], marker='x', markersize=10, color='red', label=f'Max clusters {np.max(n_clusters_list)} pour eps={eps_values[np.argmax(n_clusters_list)]} nm')

plt.xlabel('eps (nm)')
plt.ylabel('Nombre')
plt.title('Évolution du bruit en DBSCAN en fonction de eps')
plt.legend()
plt.grid(True)

plt.show()


# In[ ]:


k_list = np.arange(3, 9)  # MinPts de 3 à 10
eps = 28 #nm
n_clusters_list = np.zeros(len(k_list), dtype=int)
n_noise_list = np.zeros(len(k_list), dtype=int)

for k in k_list:
    dbscan_model = DBSCAN(eps=eps, min_samples=k)
    labels = dbscan_model.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    n_clusters_list[k_list == k] = n_clusters
    n_noise_list[k_list == k] = n_noise


# In[ ]:


plt.close('all')
plt.figure(figsize=(10, 6))
plt.plot(k_list, n_clusters_list, '-o', label='Nombre de clusters')
plt.plot(k_list[np.argmax(n_clusters_list)], np.max(n_clusters_list), marker='x', markersize=10, color='red', label=f'Max clusters {np.max(n_clusters_list)} pour eps={eps_values[np.argmax(n_clusters_list)]} nm')
plt.xlabel('eps (nm)')
plt.ylabel('Nombre')
plt.title('Évolution du clustering DBSCAN en fonction de eps')
plt.legend()
plt.grid(True)

plt.show()
plt.figure(figsize=(10, 6))
plt.plot(k_list, n_noise_list, '-o', label='Nombre de points bruit')
plt.plot(k_list[np.argmax(n_clusters_list)], n_noise_list[np.argmax(n_clusters_list)], marker='x', markersize=10, color='red', label=f'Max clusters {np.max(n_clusters_list)} pour eps={eps_values[np.argmax(n_clusters_list)]} nm')

plt.xlabel('eps (nm)')
plt.ylabel('Nombre')
plt.title('Évolution du bruit en DBSCAN en fonction de eps')
plt.legend()
plt.grid(True)

plt.show()


# In[ ]:


plt.close('all')
plt.figure(figsize=(10, 10))

# Normal histogram
bins = 500
H, xedges, yedges = np.histogram2d(X[:, 0], X[:, 1], bins=bins)

plt.imshow(H.T, origin='lower', cmap='inferno', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='equal')
plt.colorbar(label='Point density')
plt.title(f'Point density n_bins ={ bins}')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

plt.figure(figsize=(10, 10))

# Cluster histogram

cluster_labels = np.unique(labels)
cluster_labels = cluster_labels[cluster_labels != -1]  # remove noise
centroids = np.array([X[labels == lab].mean(axis=0) for lab in cluster_labels])
H, xedges, yedges = np.histogram2d(centroids[:, 0],centroids[:, 1],bins=bins)

plt.imshow(H.T,origin='lower',cmap='inferno',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],aspect='equal')
plt.colorbar(label='Cluster density')
plt.title(f'Spatial density of clusters (centroid-based) n_bins ={ bins}')
plt.xlabel('X')
plt.ylabel('Y')

plt.show()


# In[ ]:


from skimage.measure import marching_cubes
import pyvista as pv


# In[ ]:


#DATA 3D ... bins 3D? 
bins = 60
cluster_labels = np.unique(labels)
cluster_labels = cluster_labels[cluster_labels != -1]

centroids = np.array([
    X[labels == lab].mean(axis=0)  # X is (N, 3)
    for lab in cluster_labels
])
bins = 50  # number of voxels per dimension

H, edges = np.histogramdd(
    centroids,
    bins=bins
)
dx = edges[0][1] - edges[0][0]
dy = edges[1][1] - edges[1][0]
dz = edges[2][1] - edges[2][0]

cluster_density = H / (dx * dy * dz)  # clusters / nm³

plt.close('all')


verts, faces, _, _ = marching_cubes(
    cluster_density,
    level=cluster_density.mean()
)

mesh = pv.PolyData(verts, faces)
plotter = pv.Plotter()
plotter.add_mesh(mesh, cmap='inferno')
plotter.show()


# In[25]:


#REPRESENTATION 2D DE LA DENSITE (PTS ET CLUSTER) AVEC NBRE DE BINS QUI VARIE

bin_list = [50, 100, 150, 200, 300, 400]
plt.close('all')
plt.figure(figsize=(12, 8))

xmin, xmax = X[:, 0].min(), X[:, 0].max()
ymin, ymax = X[:, 1].min(), X[:, 1].max()
for i, bins in enumerate(bin_list):
    H, _, _ = np.histogram2d(
        centroids[:,0], centroids[:,1],
        bins=bins,
        range=[[xmin, xmax], [ymin, ymax]]
    )

    plt.subplot(2, 3, i+1)
    plt.imshow(
        H.T,
        origin='lower',
        cmap='inferno',
        extent=[xmin, xmax, ymin, ymax],
        aspect='equal'
    )
    plt.title(f'bins = {bins}')
    plt.axis('off')

plt.suptitle('Cluster density vs bin size')
plt.tight_layout()
plt.show()


# Debut de OPTCS? 

'''from scipy.stats import pearsonr

def density_map(bins):
    H, _, _ = np.histogram2d(
        centroids[:,0], centroids[:,1],
        bins=bins,
        range=[[xmin, xmax], [ymin, ymax]]
    )
    return H / H.sum()   # normalize

correlations = []

for b1, b2 in zip(bin_list[:-1], bin_list[1:]):
    D1 = density_map(b1)
    D2 = density_map(b2)

    # resize smaller to larger if needed
    D1_flat = D1.flatten()
    D2_flat = D2[:D1.shape[0], :D1.shape[1]].flatten()

    corr, _ = pearsonr(D1_flat, D2_flat)
    correlations.append(corr)

plt.figure()
plt.plot(bin_list[1:], correlations, '-o')
plt.xlabel('Bins')
plt.ylabel('Correlation with previous scale')
plt.title('Stability of cluster density vs bin size')
plt.grid(True)
plt.show()'''



# In[ ]:


print(n_clusters_list)
print(np.argmax(n_clusters_list))
print(np.max(n_clusters_list))


# In[31]:


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
    X_masked[mask,0], X_masked[mask,1],
    c=labels[mask],
    cmap=cmap,
    norm=norm,
    s=0.01
)

# bruit en gris
plt.scatter(
    X_masked[~mask,0], X_masked[~mask,1],
    c='lightgray',
    s=0.01,
    label='Noise'
)
plt.axis('equal')
plt.title(f'Clusters identified by DBSCAN: MinPts = {k}, eps = {eps} nm. n_clusters = {n_clusters}, noise_pts = {n_noise}')

'''cbar = plt.colorbar(sc, ticks=cluster_labels)
cbar.set_label("Cluster")
cbar.set_ticklabels([f"{k}" for k in cluster_labels])'''

plt.show()


# In[ ]:


#INUTILE Pour peu de clusters, on peut les visualiser séparément
plt.close('all')

cluster_labels = np.unique(labels[labels != -1])
n_clusters = len(cluster_labels)

ncols = 4
nrows = math.ceil(n_clusters / ncols)


xmin, xmax = X[:, 0].min(), X[:, 0].max()
ymin, ymax = X[:, 1].min(), X[:, 1].max()



fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(4*ncols, 4*nrows),
    constrained_layout=True,
    sharex=True,
    sharey=True
)

axes = axes.flatten()

for ax, label in zip(axes, cluster_labels):
    pts = X[labels == label]
    ax.scatter(pts[:, 0], pts[:, 1], s=0.05)
    ax.set_title(f'Cluster {label}\n(n={len(pts)})')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


    ax.set_box_aspect(1)


for ax in axes[len(cluster_labels):]:
    ax.axis('off')




for ax in axes[len(cluster_labels):]:
    ax.axis('off')


fig.suptitle(
    f'Clusters identified by DBSCAN: MinPts = {k}, eps = {eps} nm',
    fontsize=16
)

plt.show()


# In[ ]:


# INUTILE
trunc = 150000
d=5

abcisse = np.arange(trunc, X_f_ROI.shape[0],1)
fit = np.polyfit(abcisse, plot[trunc:], d)     
dfit = np.polyder(fit)           
racines= np.roots(dfit)
poly = np.poly1d(fit)
y_fit = poly(abcisse)

# Turning point d'intérêt est celui le plus à droite
coude = np.array([int(np.max(racines)), plot[int(np.max(racines))]])

#Check expérimental 
plt.figure(figsize=(10, 6))
plt.plot(abcisse,plot[trunc:], marker='o', markersize=3, label=f'k={k}')
plt.plot(abcisse, y_fit, '-', label=f'Fit degré {d}')
plt.plot(coude[0], coude[1], marker='x', markersize=10, color='red', label='Coude')
plt.xlabel('Points sorted by distance')
plt.ylabel('k-th nearest neighbor distance (nm)')
plt.title('K-distance Graph')
plt.grid(True)
plt.legend()
plt.show()

#Check total
plt.figure(figsize=(10, 6))
plt.plot(plot, marker='o', markersize=3, label=f'k={k}')
plt.plot(coude[0], coude[1], marker='x', markersize=10, color='red', label='Coude')
plt.xlabel('Points sorted by distance')
plt.ylabel('k-th nearest neighbor distance (nm)')
plt.title('K-distance Graph')
plt.grid(True)
plt.legend()
plt.show()


# In[ ]:


plt.close('all')


# In[ ]:




