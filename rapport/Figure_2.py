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
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import hsv_to_rgb


# In[2]:


data_treated = pd.read_csv(r'C:\Users\LOCCO\Project_Curie\labmeeting\datatreatedA2Z0CCD70150.csv',sep=',')
frame_treated_0 = data_treated['frame'].values
X_treated_0 = (data_treated[['x [nm]', 'y [nm]', 'z [nm]']].values).T
rho_treated_0 = data_treated['rho'].values
delta_treated_0 = data_treated['delta'].values


# In[3]:


data_treated = pd.read_csv(r'C:\Users\LOCCO\Project_Curie\labmeeting\datatreatedA2Z1CCD70150.csv',sep=',')
frame_treated_1 = data_treated['frame'].values
X_treated_1 = (data_treated[['x [nm]', 'y [nm]', 'z [nm]']].values).T
rho_treated_1 = data_treated['rho'].values
delta_treated_1 = data_treated['delta'].values


# In[4]:


data_treated = pd.read_csv(r'C:\Users\LOCCO\Project_Curie\labmeeting\datatreatedA1CCD70150.csv',sep=',')
frame_treated_2 = data_treated['frame'].values
X_treated_2 = (data_treated[['x [nm]', 'y [nm]', 'z [nm]']].values).T
rho_treated_2 = data_treated['rho'].values
delta_treated_2 = data_treated['delta'].values


# In[7]:


#Hemispherical angle distribution
def rho_distribution(ax, rho, title):

    rho_wrapped = np.concatenate([rho, rho + 180]) % 360
    theta = np.deg2rad(rho_wrapped)

    n, bins_rho, patches = ax.hist(theta, bins=360, range=(0, 2*np.pi))
    # Color each bar by its hue
    for patch, left_edge in zip(patches, bins_rho[:-1]):
        hue = left_edge / 180.0
        patch.set_facecolor(hsv_to_rgb([[hue, 1.0, 1.0]])[0])


    ax.set_title(title)

    return ax


# In[8]:


fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(15, 5),
    subplot_kw={'projection': 'polar'}
)

rho_distribution(ax1, rho_treated_0, 'Rho(°) Noyau 0')
rho_distribution(ax2, rho_treated_1, 'Rho(°) Noyau 1')
rho_distribution(ax3, rho_treated_2, 'Rho(°) Noyau 2')

plt.tight_layout()
plt.show()


# In[ ]:


def plot_semicircle_colorbar(ax=None, title='ρ (°)'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw=dict(projection=None))

    n_segments = 180
    theta = np.linspace(0, np.pi, n_segments + 1)  

    for i in range(n_segments):
        angle_deg = i  
        hue = angle_deg / 180.0
        color = hsv_to_rgb([[hue, 1.0, 1.0]])[0]

        # Wedge from theta[i] to theta[i+1]
        wedge = mpatches.Wedge(
            center=(0, 0),
            r=1.0,
            theta1=np.degrees(theta[i]),
            theta2=np.degrees(theta[i+1]),
            width=0.4,        # ring thickness
            color=color
        )
        ax.add_patch(wedge)

    # Tick labels at 0°, 45°, 90°, 135°, 180°
    for deg in [0, 45, 90, 135, 180]:
        rad = np.radians(deg)
        x = 1.15 * np.cos(rad)
        y = 1.15 * np.sin(rad)
        ax.text(x, y, f'{deg}°', ha='center', va='center', fontsize=9)

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.3, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=11)

    return ax



def add_scale_bar(ax, length_mum=1, fontsize=9):
    # Get current axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    length = length_mum*1000  # in micrometers, adjust as needed
    # Position: bottom-left corner with some padding
    x_start = xlim[0] + 0.05 * (xlim[1] - xlim[0])
    y_pos   = ylim[0] + 0.04 * (ylim[1] - ylim[0])
    x_end   = x_start + length

    # Draw bar
    ax.plot([x_start, x_end], [y_pos, y_pos], 'k-', linewidth=2, solid_capstyle='butt')
    # End ticks
    tick_h = 0.01 * (ylim[1] - ylim[0])
    ax.plot([x_start, x_start], [y_pos - tick_h, y_pos + tick_h], 'k-', linewidth=2)
    ax.plot([x_end,   x_end  ], [y_pos - tick_h, y_pos + tick_h], 'k-', linewidth=2)
    # Label
    ax.text((x_start + x_end) / 2, y_pos + 2 * tick_h,
            f'{length_mum} µm', ha='center', va='bottom', fontsize=fontsize,
            color='black')

def add_rho_distribution( rho, title, ax=None):

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(122, projection='polar')

    rho_wrapped = np.concatenate([rho, rho + 180]) % 360
    theta = np.deg2rad(rho_wrapped)

    n, bins_rho, patches = ax.hist(theta, bins=360, range=(0, 2*np.pi))
    # Color each bar by its hue
    for patch, left_edge in zip(patches, bins_rho[:-1]):
        hue = left_edge / 180.0
        patch.set_facecolor(hsv_to_rgb([[hue, 1.0, 1.0]])[0])


    ax.set_title(title)

    return ax



# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


def add_rho_distribution(ax, rho, title=''):
    """
    Histogramme polaire de rho sur [0°, 180°].
    """
    theta = np.deg2rad(rho)

    ax.hist(
        theta,
        bins=180,
        range=(0, np.pi)
    )

    ax.set_thetamin(0)
    ax.set_thetamax(180)

    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180]))
    ax.set_xticklabels(['0°', '45°', '90°', '135°', '180°'], fontsize=8)

    ax.set_yticklabels([])
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_title(title, fontsize=11, pad=12)

    return ax


def plot_rho_distributions_only(datasets, figsize=(15, 5), save_path=None):
    """
    Trace les 3 distributions de rho côte à côte.

    datasets : liste de 3 tuples (X, rho, titre)
               On n'utilise ici que rho.
    """
    assert len(datasets) == 3, "Il faut exactement 3 jeux de données."

    fig, axes = plt.subplots(
        1, 3,
        figsize=figsize,
        subplot_kw={'projection': 'polar'}
    )

    custom_titles = ["Noyau 0", "Noyau 1", "Noyau 2"]

    for ax, (_, rho, _), title in zip(axes, datasets, custom_titles):
        add_rho_distribution(ax, rho, title=title)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')

    return fig, axes


# ----------------------------------------------------------------------
# Exemple avec tes données
# ----------------------------------------------------------------------

datasets = [
    (X_treated_0, rho_treated_0, "Noyau 0"),
    (X_treated_1, rho_treated_1, "Noyau 1"),
    (X_treated_2, rho_treated_2, "Noyau 2")
]

fig, axes = plot_rho_distributions_only(datasets, figsize=(15, 5))
plt.show()

