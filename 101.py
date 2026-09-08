#!/usr/bin/env python
# coding: utf-8

# # Code densité 101

# Imports

# In[2]:


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


# In[3]:


from scipy import stats
x = np.array([60,110,160,210])
y = np.array([379,476,549,610])
plt.scatter(x, y)
plt.show()
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print(f'Pente (slope): {slope:.4f}')
print(f'Ordonnée à l\'origine (intercept): {intercept:.4f}')
print(f'Coefficient de corrélation (r): {r_value:.4f}')
print(f'R²: {r_value**2:.4f}')
print(f'p-value: {p_value:.4e}')


# In[ ]:


for i,rid in enumerate(r):
    plt.figure()
    plt.hist(densitesm[:,i], bins="auto")  # arguments are passed to np.histogram
    plt.title(f"Histogramme de densité pour un rayon de {rid} nm")
    plt.xlabel("Densité en nb_pts dans la sphère de rayon r")
    plt.ylabel("Nobre de points ayant cette densité")
    plt.show()


# In[77]:


for i,rid in enumerate(r):
    counts, bin_edges = np.histogram(densitesm[:,i], bins=300)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mask = counts > 0
    plt.figure(figsize=(8, 5))
    plt.bar(bin_centers, counts, width=(bin_edges[1]-bin_edges[0]), alpha=0.6, label="Histogramme")
    plt.xlabel("Densité en nb_pts dans la sphère de rayon r")
    plt.ylabel("Nombre de points")
    plt.title(f"Histogramme de densité (r = {rid} nm) — fit exponentiel")
    plt.legend()
    plt.tight_layout()
    plt.show()


# In[12]:


from sklearn.neighbors import KDTree
from scipy import sparse
import numpy as np

def morans_i_3d(X, values, k=10):
    """Compute Global Moran's I using k nearest neighbors in 3D."""
    n = len(values)

    # Build KNN graph
    tree = KDTree(X)
    dist, idx = tree.query(X, k=k+1)  # +1 because point queries itself
    idx = idx[:, 1:]  # remove self

    # Build row-standardized weight matrix (sparse)
    rows = np.repeat(np.arange(n), k)
    cols = idx.flatten()
    data = np.ones(n * k) / k  # row-standardized
    W = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    # Moran's I formula
    z = values - values.mean()
    numerator = n * z @ W @ z
    denominator = (z @ z) * W.sum()
    I = numerator / denominator

    # Spatial lag
    spatial_lag = W @ values

    return I, spatial_lag

#I, spatial_lag = morans_i_3d(X_treated, rho_treated, k=10)
#print(f"Moran's I: {I:.4f}")
# I > 0 → spatial clustering of density
# I ≈ 0 → random
# I < 0 → dispersed


# In[13]:


def morans_i_permutation_test(X, values, k=10, n_permutations=999):
    I_obs, _ = morans_i_3d(X, values, k=k)

    I_sim = []
    for _ in range(n_permutations):
        shuffled = np.random.permutation(values)
        I_perm, _ = morans_i_3d(X, shuffled, k=k)
        I_sim.append(I_perm)

    I_sim = np.array(I_sim)
    p_value = (np.sum(I_sim >= I_obs) + 1) / (n_permutations + 1)
    z_score = (I_obs - I_sim.mean()) / I_sim.std()

    print(f"Moran's I  : {I_obs:.4f}")
    print(f"E[I]       : {I_sim.mean():.4f}")
    print(f"p-value    : {p_value:.4f}")
    print(f"z-score    : {z_score:.4f}")

    return p_value, z_score

morans_i_permutation_test(X_treated, rho_treated, n_permutations=999)


# In[ ]:


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


clust = OPTICS(min_samples=10, max_eps = np.inf)
clust.fit(X_treated)
reachability = clust.reachability_[clust.ordering_]  

labels = clust.labels_[clust.ordering_]
space = np.arange(len(X_treated))



# In[ ]:


plt.figure()
plt.plot(space,reachability,'+' )
plt.ylabel('Eps distance')
plt.xlabel('Arranged point index')
plt.title('Reachability Plot for OPTICS Clustering')


# In[ ]:


#  BIMODAL histogram = clusters appart from noise

from scipy.stats import gaussian_kde

a,b = np.shape(densitesm)
for i in range(b): 
    tree = KDTree(X_treated)
    # count neighbors within 150nm for each point (excluding self)
    density = tree.query_radius(X_treated, r=150, count_only=True) - 1
    x = np.linspace(0, density.max(), 500)
    kde = gaussian_kde(density)
    plt.plot(x, kde(x))
    plt.xlabel("nb neighbors at 150nm")
    plt.title("KDE of density")
    plt.show()


# In[78]:


from scipy.stats import gaussian_kde

a,b = np.shape(densitesm)
for i in range(b):
    density = densite_m[:, i]
    rid = r[i]

    x = np.linspace(0, density.max(), 500)
    kde = gaussian_kde(density, bw_method=0.1)

    plt.figure()
    plt.plot(x, kde(x))
    plt.xlabel(f"Nb voisins à {rid} nm")
    plt.title(f"KDE de densité — r={rid} nm")
    plt.show()


# In[ ]:


plt.figure()
plt.hist(densite_m, bins=150)  
plt.title(f"Histogramme de densité pour un rayon de {r} nm")
plt.xlabel("Densité en nb_pts dans la sphère de rayon r")
plt.ylabel("Nobre de points ayant cette densité")
plt.show()

plt.figure()
plt.hist(orientation_m, bins='auto')  
plt.title(f"Histogramme d'orientation par regroupements sphériques de rayon {r} nm")
plt.xlabel("rho (°) moyen")
plt.ylabel("Nobre de points ayant ce rho")
plt.show()


# In[ ]:


X_treated = X_treated.T
n_pts = len(X_treated)
r = np.arange(r_min ,r_max, r_step)
x_range = np.linspace(0, 180, 200)
chunk_size = 500

average_rho = np.abs(np.mean(np.cos(np.radians(rho_treated))))

rng = np.random.default_rng(0) 
rho_stat_test = rng.uniform(0,180,n_pts)


# Écriture directe sur disque — jamais chargé entièrement en RAM
#kde_m = np.lib.format.open_memmap(
#    r'C:\Users\LOCCO\Project_Curie\test_or\kdem.npy',
#    mode='w+', dtype=np.float32, shape=(n_pts, len(r), len(x_range))
#)

densite_m = np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\test_or_2\densitem.npy',
    mode='w+', dtype=np.float32, shape=(n_pts, len(r))
)

meancos = np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\test_or_2\meancos.npy',
    mode='w+', dtype=np.float32, shape=(n_pts, len(r))
)

meancos_stat = np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\test_or_2\meancos_stat.npy',
    mode='w+', dtype=np.float32, shape=(n_pts, len(r))
)

for chunk_start in tqdm(range(0, n_pts, chunk_size), desc="Chunks"):
    chunk_end  = min(chunk_start + chunk_size, n_pts)
    X_chunk    = X_treated[chunk_start:chunk_end]
    dist_chunk = cdist(X_chunk, X_treated).astype(np.float32)

    for i, rid in enumerate(r):
        mask_chunk = dist_chunk <= rid
        densite_m[chunk_start:chunk_end, i] = mask_chunk.sum(axis=1)

        for local_id in range(chunk_end - chunk_start):
            rho_in = rho_treated[mask_chunk[local_id]]
            rho_in_stat = rho_stat_test[mask_chunk[local_id]]

            if len(rho_in) < 2:
                meancos[chunk_start+local_id, i] = 0
                meancos_stat[chunk_start+local_id, i]=0
                continue

            if np.std(rho_in) < 1e-10:
                # Tous les voisins ont le même rho → pic de Dirac à cette valeur
                idx = np.argmin(np.abs(x_range - rho_in[0]))
                #kde_m[chunk_start + local_id, i, idx] = 1.0
                meancos[chunk_start+local_id, i] = np.abs(np.mean(np.cos(np.radians(rho_in))))  
                meancos_stat[chunk_start+local_id, i] = np.abs(np.mean(np.cos(np.radians(rho_in_stat))))  
                continue

            meancos[chunk_start+local_id, i] = np.abs(np.mean(np.cos(np.radians(rho_in))))  
            meancos_stat[chunk_start+local_id, i] = np.abs(np.mean(np.cos(np.radians(rho_in_stat))))  
            #kde = gaussian_kde(rho_in, bw_method=0.1)
            #kde_m[chunk_start + local_id, i, :] = kde(x_range)


# Flush sur disque
#kde_m.flush()
densite_m.flush()
meancos.flush()
meancos_stat.flush()

# Sauvegarder les metadata séparément
np.save(r'C:\Users\LOCCO\Project_Curie\test_or_2\xrange.npy', x_range)
np.save(r'C:\Users\LOCCO\Project_Curie\test_or_2\r.npy', r)
print("Done.")


# In[69]:


import matplotlib.colors as mcolors
import matplotlib as mpl
data = pd.read_csv(r'C:\Users\LOCCO\Project_Curie\pour_lola\rayons.csv',sep=',')
densite_m = data['densite_145_nm'].values


vmin = 50
vmax = 400 
hsv_cmap = mpl.colormaps['inferno']
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

mask = (densite_m > 53)
X_treated_masked = X_treated[mask]
densite_m_masked = densite_m[mask]
# 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
sc = ax.scatter(X_treated_masked[:,0], X_treated_masked[:,1], X_treated_masked[:,2],
                c=densite_m_masked, cmap=hsv_cmap, norm=norm, s=0.1)
ax.axis('equal')
cb = plt.colorbar(sc)
cb.set_label('Densité (nb de pts)')


plt.show()


# In[23]:


data = pd.read_csv(r'C:\Users\LOCCO\Project_Curie\pour_lola\rayons.csv',sep=',')
densite_m = data['densite_145_nm'].values
hsv_cmap = mpl.colormaps['hsv']

norm = mpl.colors.Normalize(vmin=0, vmax=180)
mask = (densite_m > 350)
X_treated_masked = X_treated[mask]
rho_treated_masked = rho_treated[mask]

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
sc = ax.scatter(X_treated_masked[:,0], X_treated_masked[:,1], X_treated_masked[:,2],
                c=rho_treated_masked, cmap=hsv_cmap, norm=norm, s=0.8)
ax.axis('equal')
cb = plt.colorbar(sc)
cb.set_label('Angle rho (°)')


# In[93]:


counts, bin_edges = np.histogram(rho_treated, bins="auto")
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
mask = counts > 0

plt.figure(figsize=(8, 5))
plt.bar(bin_centers, counts, width=(bin_edges[1]-bin_edges[0]), alpha=0.6, label="Histogramme")
plt.xlabel("Rho (°)")
plt.ylabel("Nombre de points ayant cette orientation")
plt.title("Orientation (rho)")
plt.legend()
plt.tight_layout()
plt.show()


# In[95]:


from scipy.stats import gaussian_kde

# KDE sur les données brutes
kde = gaussian_kde(rho_treated, bw_method=0.1)  # ajuste bw_method si trop lisse/rugueux

x_range = np.linspace(0, 180, 1000)
kde_values = kde(x_range)

plt.figure()
plt.plot(x_range, kde_values)
plt.title("KDE de rho dans le noyau")
plt.show()


# In[97]:


np.abs(np.cos(np.mean(np.radians(rho_treated))))


# In[6]:


#Import data treated
data_treated = pd.read_csv(r'C:\Users\LOCCO\Project_Curie\pour_lola\zone1_cc_treated_withdelta.csv',sep=',')
frame_treated = data_treated['frame'].values
X_treated = data_treated[['x [nm]', 'y [nm]', 'z [nm]']].values
rho_treated = data_treated['rho'].values
delta_treated = data_treated['delta'].values


# In[3]:


X_treated.shape


# In[29]:


n_pts = len(X_treated)
rng = np.random.default_rng(0) 
rho_stat_test = rng.uniform(0,180,n_pts)


# In[34]:


os.makedirs(r'C:\Users\LOCCO\Project_Curie\test_or_2', exist_ok=True)


# In[35]:


from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
import numpy as np
from tqdm import tqdm

#X_treated = X_treated.T
n_pts = len(X_treated)
r = np.arange(40, 200, 10)
x_range = np.linspace(0, 180, 200)
chunk_size = 500

average_rho = np.abs(np.mean(np.cos(np.radians(rho_treated))))

rng = np.random.default_rng(0) 
rho_stat_test = rng.uniform(0,180,n_pts)


# Écriture directe sur disque — jamais chargé entièrement en RAM
#kde_m = np.lib.format.open_memmap(
#    r'C:\Users\LOCCO\Project_Curie\test_or\kdem.npy',
#    mode='w+', dtype=np.float32, shape=(n_pts, len(r), len(x_range))
#)

densite_m = np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\test_or_2\densitem.npy',
    mode='w+', dtype=np.float32, shape=(n_pts, len(r))
)

meancos = np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\test_or_2\meancos.npy',
    mode='w+', dtype=np.float32, shape=(n_pts, len(r))
)

meancos_stat = np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\test_or_2\meancos_stat.npy',
    mode='w+', dtype=np.float32, shape=(n_pts, len(r))
)

for chunk_start in tqdm(range(0, n_pts, chunk_size), desc="Chunks"):
    chunk_end  = min(chunk_start + chunk_size, n_pts)
    X_chunk    = X_treated[chunk_start:chunk_end]
    dist_chunk = cdist(X_chunk, X_treated).astype(np.float32)

    for i, rid in enumerate(r):
        mask_chunk = dist_chunk <= rid
        densite_m[chunk_start:chunk_end, i] = mask_chunk.sum(axis=1)

        for local_id in range(chunk_end - chunk_start):
            rho_in = rho_treated[mask_chunk[local_id]]
            rho_in_stat = rho_stat_test[mask_chunk[local_id]]

            if len(rho_in) < 2:
                meancos[chunk_start+local_id, i] = 0
                meancos_stat[chunk_start+local_id, i]=0
                continue

            if np.std(rho_in) < 1e-10:
                # Tous les voisins ont le même rho → pic de Dirac à cette valeur
                idx = np.argmin(np.abs(x_range - rho_in[0]))
                #kde_m[chunk_start + local_id, i, idx] = 1.0
                meancos[chunk_start+local_id, i] = np.abs(np.mean(np.cos(np.radians(rho_in))))  
                meancos_stat[chunk_start+local_id, i] = np.abs(np.mean(np.cos(np.radians(rho_in_stat))))  
                continue

            meancos[chunk_start+local_id, i] = np.abs(np.mean(np.cos(np.radians(rho_in))))  
            meancos_stat[chunk_start+local_id, i] = np.abs(np.mean(np.cos(np.radians(rho_in_stat))))  
            #kde = gaussian_kde(rho_in, bw_method=0.1)
            #kde_m[chunk_start + local_id, i, :] = kde(x_range)


# Flush sur disque
#kde_m.flush()
densite_m.flush()
meancos.flush()
meancos_stat.flush()

# Sauvegarder les metadata séparément
np.save(r'C:\Users\LOCCO\Project_Curie\test_or_2\xrange.npy', x_range)
np.save(r'C:\Users\LOCCO\Project_Curie\test_or_2\r.npy', r)
print("Done.")


# In[36]:


from sklearn.cluster import OPTICS


optics = OPTICS(min_samples=5, metric='euclidean')
optics.fit(X_treated)

ordering = optics.ordering_  # indices des points dans l'ordre de propagation OPTICS
np.save(r'C:\Users\LOCCO\Project_Curie\test_or_2\optics_ordering.npy', ordering)




# In[8]:


ordering =  np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\test_or_2\optics_ordering.npy',
    mode='r')

r =  np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\test_or_2\r.npy',
    mode='r')

meancos = np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\test_or_2\meancos.npy',
    mode='r'
)

meancos_stat =  np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\test_or_2\meancos_stat.npy',
    mode='r'
)

densitem = np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\test_or_2\densitem.npy',
    mode='r'
)
for i in range(len(r)):
    meancos_ordered = meancos[ordering, i]
    meancos_stat_ordered = meancos_stat[ordering, i]
    densite_ordered = densitem[ordering,i]/ np.max(densitem[:,i])
    # et tu traces

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(18, 6))
    '''
    ax1.plot(optics.reachability_[ordering], color='gray')
    ax1.set_ylabel("OPTICS reachability distance")
    '''

    ax1.plot(densite_ordered, color='red')
    ax1.set_ylabel(f"Number of neigbours in sphere")

    ax2.plot(meancos_ordered, color ='blue')
    ax2.set_ylabel(f"Mean polarisation in sphere ")

    ax3.plot(meancos_stat_ordered, color = 'green')
    ax3.set_ylabel("Mean polarisation of randomly distributed angles in sphere ")
    plt.title(f'r={r[i]}, densite_max={np.max(densitem[:,i])}')
    plt.tight_layout()
    plt.show()


# In[4]:


r =  np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\test_or_2\r.npy',
    mode='r')

ordering =  np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\test_or_2\optics_ordering.npy',
    mode='r')

meancos = np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\test_or_2\meancos.npy',
    mode='r'
)

meancos_stat =  np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\test_or_2\meancos_stat.npy',
    mode='r'
)

densitem = np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\test_or_2\densitem.npy',
    mode='r'
)
for i in range(len(r)):
    meancos_ordered = meancos[ordering, i]
    meancos_stat_ordered = meancos_stat[ordering, i]
    densite_ordered = densitem[ordering,i]/ np.max(densitem[:,i])
    # et tu traces
    kde_cosmean = gaussian_kde(meancos[:,i], bw_method=0.1) 
    x_range = np.linspace(0, 1, 500)
    kde_values_cosmean = kde_cosmean(x_range)
    kde_values_cosmean = kde_values_cosmean / kde_values_cosmean.max() #norm

    kde_cosmean_stat = gaussian_kde(meancos_stat[:,i], bw_method=0.1) 
    x_range = np.linspace(0, 1, 500)
    kde_values_cosmean_stat = kde_cosmean_stat(x_range)
    kde_values_cosmean_stat = kde_values_cosmean_stat / kde_values_cosmean_stat.max() #norm

    kde_densite = gaussian_kde(densitem[:,i], bw_method=0.3)  # ajuste bw_method si trop lisse/rugueux
    x_ = np.linspace(0, np.max(densitem[:,i]), 500)
    kde_values_densite= kde_densite(x_)
    kde_values_densite = kde_values_densite /kde_values_densite.max()


    fig = plt.figure(figsize=(18, 12))

    # signal ordonné — toute la largeur en haut
    ax_top = fig.add_subplot(2, 1, 1)
    ax_top.plot(meancos_ordered, color='steelblue', label='meancos = abs(mean(cos(rho_dans_sphere)))')
    ax_top.plot(densite_ordered, color='red', alpha=0.6, label='densité')
    ax_top.set_xlabel("ordre OPTICS")
    ax_top.set_title(f'r={r[i]}, densite_max={np.max(densitem[:, i]):.0f}')
    ax_top.legend()

    # KDE cosmean — bas gauche
    ax_kde1 = fig.add_subplot(2, 3, 4)
    ax_kde1.plot(x_range, kde_values_cosmean, color='steelblue')
    ax_kde1.set_title('KDE meancos normalisé')
    ax_kde1.set_xlabel('cosmean')

    # KDE cosmean_stat — bas milieu
    ax_kde2 = fig.add_subplot(2, 3, 5)
    ax_kde2.plot(x_range, kde_values_cosmean_stat, color='steelblue')
    ax_kde2.set_title('KDE meancos pour angles randomnisés normalisé')
    ax_kde2.set_xlabel('cosmean')

    # KDE densité — bas droite
    ax_kde3 = fig.add_subplot(2, 3, 6)
    ax_kde3.plot(x_range, kde_values_densite, color='red')
    ax_kde3.set_title('KDE densité normalisée')
    ax_kde3.set_xlabel('densité')

    ax_kde1.set_ylim(0, 1)
    ax_kde2.set_ylim(0, 1)
    ax_kde3.set_ylim(0, 1)
    fig.savefig(
        os.path.join(r'C:\Users\LOCCO\Project_Curie\test_or_2', f"eval{r[i]}.png"),
        dpi=100,
        bbox_inches='tight'
    )

    plt.tight_layout()
    plt.show()


# In[6]:


def morans_i_permutation_test(X, values, k=10, n_permutations=999):
    I_obs, _ = morans_i_3d(X, values, k=k)

    I_sim = []
    for _ in range(n_permutations):
        shuffled = np.random.permutation(values)
        I_perm, _ = morans_i_3d(X, shuffled, k=k)
        I_sim.append(I_perm)

    I_sim = np.array(I_sim)
    p_value = (np.sum(I_sim >= I_obs) + 1) / (n_permutations + 1)
    z_score = (I_obs - I_sim.mean()) / I_sim.std()

    print(f"Moran's I  : {I_obs:.4f}")
    print(f"E[I]       : {I_sim.mean():.4f}")
    print(f"p-value    : {p_value:.4f}")
    print(f"z-score    : {z_score:.4f}")

    return p_value, z_score

#morans_i_permutation_test(X_treated_masked, rho_treated_masked, n_permutations=999)


# In[7]:


X_treated.shape


# In[8]:


counts, bin_edges = np.histogram(rho_treated_masked, bins=150)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
mask = counts > 0

plt.figure(figsize=(8, 5))
plt.bar(bin_centers, counts, width=(bin_edges[1]-bin_edges[0]), alpha=0.6, label="Histogramme")
plt.xlabel("Rho (°)")
plt.ylabel("Nombre de points ayant cette orientation")
plt.title("Orientation (rho) des points dans les zones denses (densité > 350 por r = 145 nm)")
plt.legend()
plt.tight_layout()
plt.show()


# In[9]:


from scipy.stats import gaussian_kde

# KDE sur les données brutes
kde = gaussian_kde(rho_treated_masked, bw_method=0.1)  # ajuste bw_method si trop lisse/rugueux

x_range = np.linspace(rho_treated_masked.min(), rho_treated_masked.max(), 1000)
kde_values = kde(x_range)

plt.figure()
plt.plot(x_range, kde_values)
plt.xlabel("nb neighbors at 150nm")
plt.title("KDE de rho pour des points en zones denses")
plt.show()


# In[ ]:


clust = OPTICS(min_samples=10, max_eps = np.inf, cluster_method='xi', xi=0.05)
clust.fit(X_treated_masked)
reachability = clust.reachability_[clust.ordering_]  

labels = clust.labels_[clust.ordering_]
space = np.arange(len(X_treated_masked))

plt.figure()
plt.plot(space,reachability,'+' )
plt.ylabel('Eps distance')
plt.xlabel('Arranged point index')
plt.title('Reachability Plot for OPTICS Clustering')


# In[ ]:


clust = OPTICS(min_samples=50, max_eps = 70, cluster_method='xi', xi=0.03)
clust.fit(X_treated_masked)
reachability = clust.reachability_[clust.ordering_]  

labels = clust.labels_[clust.ordering_]
space = np.arange(len(X_treated_masked))

colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(labels))))
color_map = {label: colors[i] for i, label in enumerate(np.unique(labels))}

plt.figure(figsize=(12, 5))
for label in np.unique(labels):
    mask = labels == label
    lbl = f"Bruit" if label == -1 else f"Cluster {label}"
    plt.plot(space[mask], reachability[mask], '+', 
             color=color_map[label], label=lbl, alpha=0.7)

plt.axhline(y=100, color='red', linestyle='--', label='Seuil eps')  # si option 1
plt.ylabel('Eps distance')
plt.xlabel('Arranged point index')
plt.title('Reachability Plot — Clusters colorés')
plt.legend()
plt.show()


# In[ ]:


n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

for i in range(n_clusters):
    mask_clust = (clust.labels_ == i)
    X_treated_clust = X_treated_masked[mask_clust]
    rho_treated_clust = rho_treated_masked[mask_clust]
    kde = gaussian_kde(rho_treated_clust, bw_method=0.2)  # ajuste bw_method si trop lisse/rugueux

    x_range = np.linspace(0,180, 1000)
    kde_values = kde(x_range)
    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(x_range, kde_values)
    ax1.set_title(f"KDE de rho — Cluster {i}")

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_position([0.5, 0.05, 0.45, 0.85])  # [left, bottom, width, height]
    hsv_cmap = mpl.colormaps['hsv']
    norm = mpl.colors.Normalize(vmin=0, vmax=180)
    sc = ax2.scatter(X_treated_clust[:,0], X_treated_clust[:,1], X_treated_clust[:,2],
                    c=rho_treated_clust, cmap=hsv_cmap, norm=norm, s=10)

    ax2.set_box_aspect([1, 1, 1]) 
    cb = fig.colorbar(sc, ax=ax2, shrink=0.6)
    cb.set_label('Angle rho (°)')
    ax2.set_title(f"Disposition en localisation — Cluster {i}")

    plt.tight_layout()
    plt.show()


    morans_i_permutation_test(X_treated_clust, rho_treated_clust, n_permutations=999)

    '''
    plt.figure()
    plt.plot(x_range, kde_values)
    plt.xlabel("Orientatio rho(°)")
    plt.title(f"KDE de rho pour le cluster {i} (OPTICS)")
    plt.show()

    # 3D plot
    hsv_cmap = mpl.colormaps['hsv']

    norm = mpl.colors.Normalize(vmin=0, vmax=180)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(X_treated_clust[:,0], X_treated_clust[:,1], X_treated_clust[:,2],
                    c=rho_treated_clust, cmap=hsv_cmap, norm=norm, s=1)
    ax.axis('equal')
    cb = plt.colorbar(sc)
    cb.set_label('Angle rho (°)')'''




# In[ ]:


hsv_cmap = mpl.colormaps['hsv']
norm = mpl.colors.Normalize(vmin=0, vmax=n_clusters)
mask_bruit = (clust.labels_ >= 0)
X_treated_masked_clust = X_treated_masked[mask_bruit]
labels_clust = clust.labels_[mask_bruit]
# 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
sc = ax.scatter(X_treated_masked_clust[:,0], X_treated_masked_clust[:,1], X_treated_masked_clust[:,2],
                c=labels_clust, cmap=hsv_cmap, norm=norm, s=0.8)
ax.axis('equal')
cb = plt.colorbar(sc)
cb.set_label('Cluster id')


# In[ ]:


clust.labels_


# In[ ]:


x,y,z = X_treated[0,:], X_treated[1,:], X_treated[2,:]

fig,axes = plt.subplots(1,3, figsize=(15,5))

for ax, (a,b, label) in zip(axes, [(x,y,'X-Y'), (x,z,'X-Z'), (y,z,'Y-Z')]):
    k = gaussian_kde(np.vstack([a,b]))
    xi,yi = np.mgrid[np.min(a):np.max(a):1000j ,
                     np.min(b):np.max(b):1000j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    ax.pcolormesh(xi,yi,zi.reshape(xi.shape), cmap = 'inferno')
    ax.set_title(f'Densité par Kernel Density Estimation plan {label}')
plt.show()


# In[ ]:


x_min_int, x_max_int = int(11000), int(18000)
y_min_int, y_max_int =int(4000), int(12000)
z_min_int, z_max_int = math.floor(np.min(z)-500),int(900)


# In[ ]:


th_lat = 5
th_axial = 13
plt.close("all")
import os
os.makedirs("Densite_film", exist_ok=True)

z_grid = np.arange(z_min_int, z_max_int + 10*th_axial, 50)
x_grid = np.arange(x_min_int, x_max_int + 10*th_lat, 50)
y_grid = np.arange(y_min_int, y_max_int + 10*th_lat, 50)

ext_x_min, ext_x_max = x_grid.min() , x_grid.max() 
ext_y_min, ext_y_max = y_grid.min() , y_grid.max()

cmap = plt.cm.coolwarm.copy()
cmap.set_under("white")


for idx, zc in enumerate(tqdm(z_grid, desc="Processing slices")):

    fig , ax = plt.subplots(figsize=(6,6))
    # Sélection des points dans la tranche axiale
    mask_slice = (X_treated[:,2] >= zc ) & (X_treated[:,2] <= zc + 10*th_axial)
    x_slice = X_treated[:,0][mask_slice]
    y_slice = X_treated[:,1][mask_slice]
    #print(x_slice,y_slice)

    # Histogramme 2D = densité
    densite, _, _ = np.histogram2d(
        x_slice, y_slice,
        bins=[x_grid, y_grid]
    )
    #print(max(densite.flat))
    # Création figure + axe
    fig, ax = plt.subplots(figsize=(6,6))

    im = ax.imshow(
        densite.T,
        origin="lower",
        vmin=0.1, vmax=60,
        cmap="coolwarm",
        extent=[ext_x_min, ext_x_max, ext_y_min, ext_y_max],
        aspect="auto"
    )

    fig.colorbar(im, ax=ax, label="densité")
    ax.set_title(f"Slice z = {zc}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.savefig(
        os.path.join("Densite_film", f"frame_{idx:04d}.png"),
        dpi=100,
        bbox_inches='tight'
    )

    plt.close(fig)
plt.close("all")


# In[ ]:


np.max(densite)


# In[ ]:


import os
import imageio

def make_gif(folder, output_name):
    # Récupère tous les fichiers PNG triés
    file_list = sorted([
        f for f in os.listdir(folder)
        if f.endswith(".png")
    ])

    # Charge les images
    images = [imageio.imread(os.path.join(folder, f)) for f in file_list]

    # Sauvegarde le GIF
    imageio.mimsave(output_name, images, duration=0.1)

# Appel correct
make_gif("Densite_film", "animation_densite.gif")


# In[ ]:


frame_treated = data_treated['frame'].values
bins = np.arange(frame_treated.min(), frame_treated.max() + 1000,1000)

for i in range(len(bins)):
    mask_temp = (frame_treated >= bins[i]) & (frame_treated < bins[i]+1000)
    x_temp, y_temp, z_temp = X_treated[mask_temp,0],X_treated[mask_temp,1],X_treated[mask_temp,2]


    '''
    os.makedirs(f"Densite_film.temp{int(bins[i]):04d}", exist_ok=True)
    for idx, zc in enumerate(tqdm(z_grid, desc="Processing slices")):

        fig , ax = plt.subplots(figsize=(6,6))
        # Sélection des points dans la tranche axiale
        mask_slice = (z_temp >= zc ) & (z_temp <= zc + 10*th_axial)
        x_slice = x_temp[mask_slice]
        y_slice = y_temp[mask_slice]
        #print(x_slice,y_slice)

        # Histogramme 2D = densité
        densite, _, _ = np.histogram2d(
            x_slice, y_slice,
            bins=[x_grid, y_grid]
        )
        #print(max(densite.flat))
        # Création figure + axe
        fig, ax = plt.subplots(figsize=(6,6))

        im = ax.imshow(
            densite.T,
            origin="lower",
            vmin=0, vmax=10,
            cmap="coolwarm",
            extent=[ext_x_min, ext_x_max, ext_y_min, ext_y_max],
            aspect="auto"
        )

        fig.colorbar(im, ax=ax, label="densité")
        ax.set_title(f"Slice z = {zc}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        fig.savefig(
            os.path.join(f"Densite_film.temp{int(bins[i]):04d}", f"frame_{idx:04d}.png"),
            dpi=100,
            bbox_inches='tight'
        )

        plt.close(fig)
    '''
plt.close("all")



# In[ ]:


np.max(frame_treated)   


# In[ ]:




