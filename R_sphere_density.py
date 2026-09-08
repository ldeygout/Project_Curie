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


# In[2]:


#Import data treated
data_treated = pd.read_csv(r'C:\Users\LOCCO\Project_Curie\labmeeting\datatreatedA2Z0CCD70150.csv',sep=',')
frame_treated = data_treated['frame'].values
X_treated = (data_treated[['x [nm]', 'y [nm]', 'z [nm]']].values).T
rho_treated = data_treated['rho'].values
delta_treated = data_treated['delta'].values


# In[3]:


r_min = 40
r_max = 200
r_step = 5


# In[4]:


os.makedirs(r'C:\Users\LOCCO\Project_Curie\labmeeting\r_evol_A2Z0CCD70150', exist_ok=True)


# In[5]:


X_treated = X_treated.T
n_pts = len(X_treated)
r = np.arange(r_min ,r_max, r_step)
x_range = np.linspace(0, 180, 200)
chunk_size = 500

densite_m = np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\labmeeting\r_evol_A2Z0CCD70150\densitem.npy',
    mode='w+', dtype=np.float32, shape=(n_pts, len(r))
)

for chunk_start in tqdm(range(0, n_pts, chunk_size), desc="Chunks"):
    chunk_end  = min(chunk_start + chunk_size, n_pts)
    X_chunk    = X_treated[chunk_start:chunk_end]
    dist_chunk = cdist(X_chunk, X_treated).astype(np.float32)

    for i, rid in enumerate(r):
        mask_chunk = dist_chunk <= rid
        densite_m[chunk_start:chunk_end, i] = mask_chunk.sum(axis=1)

# Flush sur disque
densite_m.flush()

# Sauvegarder les metadata séparément
np.save(r'C:\Users\LOCCO\Project_Curie\labmeeting\r_evol_A2Z0CCD70150\xrange.npy', x_range)
np.save(r'C:\Users\LOCCO\Project_Curie\labmeeting\r_evol_A2Z0CCD70150\r.npy', r)
print("Done.")


# In[6]:


densite_m.shape


# In[12]:


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


x_lim = densite_m.max()

#Find ylim
y_lim = 0
for i in range(len(r)):
    density_i = densite_m[:, i].astype(np.float32)
    x = np.linspace(0, density_i.max(), 500)
    kde = gaussian_kde(density_i)
    y_lim = max(y_lim, kde(x).max())



# In[18]:


#Plot
for i, rid in enumerate(r):
    fig, ax = plt.subplots(figsize=(8, 5))
    density_i = densite_m[:, i].astype(np.float32)
    x = np.linspace(0,x_lim, 500)
    kde = gaussian_kde(density_i)
    ax.plot(x, kde(x))
    ax.set_xlabel("nb neighbors")
    ax.set_ylabel("KDE")
    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, kde(x).max())
    ax.set_title(f"KDE of counts for r={rid:.0f} ")
    ax.legend()
    fig.savefig(
        os.path.join(r"C:\Users\LOCCO\Project_Curie\labmeeting\r_evol_A2Z0CCD70150", f"kde_r_{int(rid):04d}.png"),
        dpi=100
    )
    plt.close(fig)


# In[19]:


make_gif(r"C:\Users\LOCCO\Project_Curie\labmeeting\r_evol_A2Z0CCD70150", r"C:\Users\LOCCO\Project_Curie\labmeeting\r_evol_A2Z0CCD70150\animation_r.gif")


# In[ ]:


densite_m = np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\labmeeting\r_evol_A2Z0CCD70150\densitem.npy',
    mode='r'
)


r  =  np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\labmeeting\r_evol_A2Z0CCD70150\r.npy',
    mode='r'
)

X_treated = (data_treated[['x [nm]', 'y [nm]', 'z [nm]']].values)

vmin = 50
vmax = 500 
hsv_cmap = mpl.colormaps['inferno']
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

mask = (densite_m[:, -1] > 50)
X_treated_masked = X_treated[mask]
densite_m_masked = densite_m[:, -1][mask]
# 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
sc = ax.scatter(X_treated_masked[:,0], X_treated_masked[:,1], X_treated_masked[:,2],
                c=densite_m_masked, cmap=hsv_cmap, norm=norm, s=0.1)
ax.axis('equal')
cb = plt.colorbar(sc)
cb.set_label('Densité (nb de pts)')


plt.show()


# In[10]:


#Plot
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
x_lim = densite_m.max()
densite_m = np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\labmeeting\r_evol_A2Z0CCD70150\densitem.npy',
    mode='r'
)

r  =  np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\labmeeting\r_evol_A2Z0CCD70150\r.npy',
    mode='r'
)
fig, ax = plt.subplots(figsize=(8, 5))
density_i = densite_m[:, -1].astype(np.float32)
x = np.linspace(0,x_lim, 500)
kde = gaussian_kde(density_i)
ax.plot(x, kde(x))

gmm = GaussianMixture(n_components=3, random_state=0)
gmm.fit(density_i.reshape(-1, 1))

means = gmm.means_.flatten()
stds = np.sqrt(gmm.covariances_.flatten())
weights = gmm.weights_.flatten()
order = np.argsort(means)
means, stds, weights = means[order], stds[order], weights[order]

colors = ['tab:red', 'tab:green', 'tab:blue']
total = np.zeros_like(x)

for k in range(3):
    gauss_k = weights[k] * norm.pdf(x, means[k], stds[k])
    total += gauss_k
    ax.plot(x, gauss_k, '--', color=colors[k],
            label=f"G{k+1}: μ={means[k]:.1f}, σ={stds[k]:.1f}, w={weights[k]:.2f}")

ax.plot(x, total, color='orange', lw=1.5, label="Somme GMM")

ax.set_xlabel("nb neighbors")
ax.set_ylabel("KDE")
ax.set_xlim(0, x_lim)
ax.set_ylim(0, kde(x).max())
ax.set_title(f"KDE of counts for r={r[-1]:.0f} ")
ax.legend()

plt.show()


# In[11]:


bics = []
n_range = range(1, 7)
for n in n_range:
    gmm_test = GaussianMixture(n_components=n, n_init=5, random_state=0)
    gmm_test.fit(density_i.reshape(-1, 1))
    bics.append(gmm_test.bic(density_i.reshape(-1, 1)))

plt.figure()
plt.plot(n_range, bics, 'o-')
plt.xlabel("Nombre de gaussiennes")
plt.ylabel("BIC")
plt.title("Sélection du nombre de composantes")
plt.show()

n_optimal = n_range[np.argmin(bics)]
print(f"Nombre optimal de gaussiennes (BIC) : {n_optimal}")


# In[26]:


from sklearn.mixture import GaussianMixture
from scipy.stats import norm

n_gauss = 4  # <-- change ici (3, 4, 5, ...)

fig, ax = plt.subplots(figsize=(8, 5))
density_i = densite_m[:, 20].astype(np.float32)
x = np.linspace(0, x_lim, 500)

# KDE (ta courbe d'origine)
kde = gaussian_kde(density_i)
ax.plot(x, kde(x), label="KDE", color='black', lw=2)

# --- Fit GMM à n_gauss composantes ---
gmm = GaussianMixture(n_components=n_gauss, n_init=10, random_state=0)
gmm.fit(density_i.reshape(-1, 1))

means = gmm.means_.flatten()
stds = np.sqrt(gmm.covariances_.flatten())
weights = gmm.weights_.flatten()

# Tri par moyenne croissante
order = np.argsort(means)
means, stds, weights = means[order], stds[order], weights[order]

cmap = plt.get_cmap('tab10')  # gère jusqu'à 10 couleurs distinctes
total = np.zeros_like(x)

for k in range(n_gauss):
    gauss_k = weights[k] * norm.pdf(x, means[k], stds[k])
    total += gauss_k
    ax.plot(x, gauss_k, '--', color=cmap(k),
            label=f"G{k+1}: μ={means[k]:.1f}, σ={stds[k]:.1f}, w={weights[k]:.2f}")

ax.plot(x, total, color='orange', lw=1.5, label="Somme GMM")

ax.set_xlabel("nb neighbors")
ax.set_ylabel("Density")
ax.set_xlim(0, x_lim)
ax.set_ylim(0, max(kde(x).max(), total.max()) * 1.1)
ax.set_title(f"KDE & GMM ({n_gauss} gaussiennes) for r={r[-1]:.0f}")
ax.legend(fontsize=8)


plt.show()


# In[20]:


fig, ax = plt.subplots(figsize=(8, 5))

# Choisis les 4 indices de colonnes que tu veux comparer (parmi les 32)
indices = [0, 10, 20, 31]

x = np.linspace(0, x_lim, 500)

for idx in indices:
    density_i = densite_m[:, idx].astype(np.float32)
    kde = gaussian_kde(density_i)

    if idx == indices[-1]:
        # Dernière courbe : en couleur, mise en avant
        ax.plot(x, kde(x), color='tab:red', lw=2.5, label=f"col {idx}")
    else:
        # Les autres : en gris
        ax.plot(x, kde(x), color='gray', lw=1.5, alpha=0.7, label=f"col {idx}")

ax.set_xlabel("nb neighbors")
ax.set_ylabel("KDE")
ax.set_xlim(0, x_lim)
ax.set_title(f"KDE comparison for r={r[-1]:.0f}")
ax.legend()

fig.savefig(
    os.path.join(r"C:\Users\LOCCO\Project_Curie\labmeeting\r_evol_A2Z0CCD70150", f"kde_comparison_r_{int(rid):04d}.png"),
    dpi=100
)
plt.show()


# In[22]:


fig, ax = plt.subplots(figsize=(8, 5))

indices = [0, 10, 20, 31]

x = np.linspace(0, x_lim, 500)

for idx in indices:
    density_i = densite_m[:, idx].astype(np.float32)
    kde = gaussian_kde(density_i)
    y = kde(x)
    y_norm = y / y.max()  # normalisation par le pic

    if idx == indices[-1]:
        ax.plot(x, y_norm, color='tab:red', lw=2.5, label=f"col {idx}")
    else:
        ax.plot(x, y_norm, color='gray', lw=1.5, alpha=0.7, label=f"col {idx}")

ax.set_xlabel("nb neighbors")
ax.set_ylabel("KDE (normalized)")
ax.set_xlim(0, x_lim)
ax.set_ylim(0, 1.05)
ax.set_title(f"Normalized KDE comparison for r={r[-1]:.0f}")
ax.legend()


plt.show()


# In[32]:


fig, ax = plt.subplots(figsize=(8, 5))

indices = [0, 10, 20, 28, 31]
colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(indices)))

x = np.linspace(0, x_lim, 500)

for idx, color in zip(indices, colors):
    density_i = densite_m[:, idx].astype(np.float32)
    kde = gaussian_kde(density_i)
    y = kde(x)
    y_norm = y / y.max()

    ax.plot(x, y_norm, color=color, lw=2, label=f"r={r[idx]}, KDE={y.max():.3f} à N={np.argmax(y)}")

ax.set_xlabel("N, Nombre de voisins compris dans la sphère")
ax.set_ylabel("KDE")
ax.set_xlim(0, x_lim)
ax.set_ylim(0, 1.05)
#ax.set_title(f"Evolution de l'estimation de distribution de densité pour différents r")
ax.legend()

plt.show()


# In[6]:


densite_m = np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\labmeeting\r_evol_A2Z0CCD70150\densitem.npy',
    mode='r'
)
x_lim = densite_m.max()
r  =  np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\labmeeting\r_evol_A2Z0CCD70150\r.npy',
    mode='r'
)
fig, ax = plt.subplots(figsize=(8, 5))

indices = [0, 10]
colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))

n_bins = np.linspace(0, x_lim, 150)  # bins fixes pour toutes les courbes

for idx, color in zip(indices, colors):
    density_i = densite_m[:, idx].astype(np.float32)

    counts, edges = np.histogram(density_i, bins=n_bins, density=True)
    counts_norm = counts / counts.max()
    centers = 0.5 * (edges[:-1] + edges[1:])

    ax.step(centers, counts_norm, where='mid', color=color, lw=2,
            label=f"r={r[idx]}")
    #max à N={centers[np.argmax(counts_norm)]:.0f}

ax.set_xlabel("N, Nombre de voisins compris dans la sphère")
ax.set_ylabel("Densité normalisée")
ax.set_xlim(0, x_lim)
ax.set_ylim(0, 1.05)
ax.legend()

plt.show()


# In[8]:


from scipy.stats import norm

densite_m = np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\labmeeting\r_evol_A2Z0CCD70150\densitem.npy',
    mode='r'
)
x_lim = densite_m.max()
r  =  np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\labmeeting\r_evol_A2Z0CCD70150\r.npy',
    mode='r'
)
fig, ax = plt.subplots(figsize=(8, 5))
indices = [0, 10, 20, 28, 31]
colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
n_bins = np.linspace(0, x_lim, 150)  # bins fixes pour toutes les courbes

for idx, color in zip(indices, colors):
    density_i = densite_m[:, idx].astype(np.float32)

    counts, edges = np.histogram(density_i, bins=n_bins, density=True)
    counts_norm = counts / counts.max()
    centers = 0.5 * (edges[:-1] + edges[1:])

    ax.step(centers, counts_norm, where='mid', color=color, lw=2,
            label=f"r={r[idx]}")

# Gaussienne ajoutée par-dessus, avec moyenne/sigma donnés manuellement
mu = 181      # <-- remplace par ta valeur
sigma = 101    # <-- remplace par ta valeur

x_gauss = np.linspace(0, x_lim, 500)
y_gauss = norm.pdf(x_gauss, mu, sigma)
y_gauss_norm = y_gauss / y_gauss.max() # normalisée comme les courbes (pic à 1)
y_gauss_norm = y_gauss_norm * 0.469
ax.plot(x_gauss, y_gauss_norm, color='black', lw=2, linestyle='--',
        label=f"Gaussienne obtenue par lecture graphique (μ={mu}, σ={sigma})")
plt.show()


# In[7]:


FWHM = (41-14)*2  # largeur totale à mi-hauteur que tu lis sur le graphe
sigma = FWHM / 2.3548
print(sigma)


# In[19]:


from scipy import stats
y = np.array([64,149,181])
x = np.array([np.sqrt(61),np.sqrt(84),np.sqrt(101)])
plt.scatter(x, y)
plt.show()
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print(f'Pente (slope): {slope:.4f}')
print(f'Ordonnée à l\'origine (intercept): {intercept:.4f}')
print(f'Coefficient de corrélation (r): {r_value:.4f}')
print(f'R²: {r_value**2:.4f}')
print(f'p-value: {p_value:.4e}')


# In[13]:


mumu = [0, 14 , 64 , 149, 181 ]
sigmama = [0, 22 , 61  , 84, 101 ]
normimi = [1, 0.606,0.462 ,0.454 , 0.469 ]


# In[15]:


from scipy.stats import norm

densite_m = np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\labmeeting\r_evol_A2Z0CCD70150\densitem.npy',
    mode='r'
)
x_lim = densite_m.max()
r  =  np.lib.format.open_memmap(
    r'C:\Users\LOCCO\Project_Curie\labmeeting\r_evol_A2Z0CCD70150\r.npy',
    mode='r'
)
# fig, ax = plt.subplots(figsize=(8, 5))
indices = [0, 10, 20, 28, 31]
colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
n_bins = np.linspace(0, x_lim, 150)  # bins fixes pour toutes les courbes

for k, (idx, color) in enumerate(zip(indices, colors)):
    fig, ax = plt.subplots(figsize=(8, 5))
    density_i = densite_m[:, idx].astype(np.float32)

    counts, edges = np.histogram(density_i, bins=n_bins, density=True)
    counts_norm = counts / counts.max()
    centers = 0.5 * (edges[:-1] + edges[1:])

    ax.step(centers, counts_norm, where='mid', color=color, lw=2,
            label=f"r={r[idx]}")

# Gaussienne ajoutée par-dessus, avec moyenne/sigma donnés manuellement
    mu = mumu[k]     # <-- remplace par ta valeur
    sigma = sigmama[k]    # <-- remplace par ta valeur
    normi =  normimi[k]

    x_gauss = np.linspace(0, x_lim, 500)
    y_gauss = norm.pdf(x_gauss, mu, sigma)
    y_gauss_norm = y_gauss / y_gauss.max() # normalisée comme les courbes (pic à 1)
    y_gauss_norm = y_gauss_norm * normi
    ax.plot(x_gauss, y_gauss_norm, color='black', lw=2, linestyle='--',
            label=f"Gaussienne obtenue par lecture graphique (μ={mu}, σ={sigma})")
    ax.legend()
    plt.show()

