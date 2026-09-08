#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Imports 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import hsv_to_rgb
import tifffile as tiff
from tqdm import tqdm



# In[2]:


data = pd.read_csv(r'C:\Users\LOCCO\Project_Curie\lambdadna\07_10_2026\lamelle_1\metadata.companion-partie_results_fr1to501_method=Propagation matrix_box-method=Fixed_invertRotationPolarizer_corr.csv',sep=',')
print(data.columns)
mask = tiff.imread(r"C:\Users\LOCCO\Project_Curie\lambdadna\07_10_2026\lamelle_1\Mask_lam1_acq3_0707.tif")    
mask = np.array(mask.transpose()) #Fiji écrit en y,x


# In[4]:


#Import data
frame = data['frame'].values
X = data[['x [nm]', 'y [nm]', 'z [nm]']].values
rho = data['rho'].values
delta = data['delta'].values
N_photons = data['intensity [u?]'].values
sigma = data[['sigmax [nm]', 'sigmay [nm]', 'sigmaz [nm]']].values
#Apply mask 12 = taille pixel, 5 = magnification imageJ
#mask_vect = (mask[(X[:, 0] / (120/5)).astype(int),(X[:, 1] / (120/5)).astype(int)] > 0)&(sigma[:,0] <= 240) & (sigma[:,1] <= 240) & (sigma[:,2] <= 720)
ix = np.clip((X[:, 0] / (120/5)).astype(int), 0, mask.shape[0] - 1)
iy = np.clip((X[:, 1] / (120/5)).astype(int), 0, mask.shape[1] - 1)

mask_vect = (mask[ix, iy] > 0) &(sigma[:,0] <= 240) & (sigma[:,1] <= 240) & (sigma[:,2] <= 720)

X_masked = X[mask_vect]
sigma_masked   = sigma[mask_vect]
rho_masked     = rho[mask_vect]
delta_masked   = delta[mask_vect]
frame_masked   = frame[mask_vect]

print('after masking free-hand roi & noisy outliers data set is ', len(X_masked), ' instead of', len(X))


# In[11]:


#Import data
frame = data['frame'].values
X = data[['x [nm]', 'y [nm]', 'z [nm]']].values
rho = data['rho'].values
delta = data['delta'].values
N_photons = data['intensity [u?]'].values
sigma = data[['sigmax [nm]', 'sigmay [nm]', 'sigmaz [nm]']].values
#Apply mask 12 = taille pixel, 5 = magnification imageJ
#mask_vect = (mask[(X[:, 0] / (120/5)).astype(int),(X[:, 1] / (120/5)).astype(int)] > 0)&(sigma[:,0] <= 240) & (sigma[:,1] <= 240) & (sigma[:,2] <= 720)
#ix = np.clip((X[:, 0] / (120/5)).astype(int), 0, mask.shape[0] - 1)
#iy = np.clip((X[:, 1] / (120/5)).astype(int), 0, mask.shape[1] - 1)

mask_vect = (sigma[:,0] <= 240) & (sigma[:,1] <= 240) & (sigma[:,2] <= 720)

X_masked = X[mask_vect]
sigma_masked   = sigma[mask_vect]
rho_masked     = rho[mask_vect]
delta_masked   = delta[mask_vect]
frame_masked   = frame[mask_vect]

print('after masking noisy outliers data set is ', len(X_masked), ' instead of', len(X))


# In[12]:


mask_delta = ( ~np.isnan(delta_masked.astype(float))  & (delta_masked >= 70)   & (delta_masked <= 150))

X_masked_b = X_masked[mask_delta]
sigma_masked_b   = sigma_masked[mask_delta]
rho_masked_b     = rho_masked[mask_delta]
delta_masked_b   = delta_masked[mask_delta]
frame_masked_b   = frame_masked[mask_delta]

print('after masking wobbling outliers data set is ', len(X_masked_b), ' instead of', len(X_masked))


# In[5]:


def recursive_call(not_counted,i,X, rho, delta, frame, previous_index,th_lat=50, th_axial=75):
    #i is index of the frame we are in 
    #Let's create a mask that takes into account only the next frame and that looks for points at <th_lat and <th_ax
    mask = (frame==frame[i]+1) & ((X[:,0]-X[i,0])**2+(X[:,1]-X[i,1])**2<th_lat**2) & ((X[:,2]-X[i,2])**2<th_axial**2)
    #Is there at least one True value in mask?
    if (mask==True).any():
        not_counted[i] = False
        return recursive_call(not_counted, np.where(mask)[0][0], X, rho, delta, frame, previous_index=np.concatenate((previous_index, np.where(mask)[0])),th_lat=50, th_axial=75)
    else:
        return previous_index.astype(int)


# In[6]:


#Commençons par obtenir les std obtenus avec les threshholds arbitraires voir si ça tient la route
#Init
nb_id = len(X_masked_b[:,0])
not_counted = np.ones(nb_id, dtype=bool)
stdx = []
stdy = []
stdz = []
stdrho = []
stddelta = []


th_lat= 15
th_axial = 30 #Obtenus après calcul de std


for i in tqdm(range(nb_id), desc="Processing points"):
    if not_counted[i]:
        indices =  recursive_call(not_counted,i,X_masked_b, rho_masked_b, delta_masked_b, frame_masked_b, previous_index=np.array([i]),th_lat=th_lat, th_axial= th_axial)
        if len(indices)>1:
            #Pour le calcul de threshhold*

            stdx.append(np.std([X_masked_b[j,0] for j in indices]))
            stdy.append(np.std([X_masked_b[j,1] for j in indices]))
            stdz.append(np.std([X_masked_b[j,2] for j in indices]))
            stdrho.append(np.std(rho_masked_b[indices]))
            stddelta.append(np.std(delta_masked_b[indices]))

stdx = np.array(stdx)
stdy = np.array(stdy)
stdz = np.array(stdz)
stdrho = np.array(stdrho)
stddelta = np.array(stddelta)

print(np.mean(stdx))
print(np.mean(stdy))
print(np.mean(stdz))
print(np.mean(stdrho))
print(np.mean(stddelta))



# In[21]:


std = pd.DataFrame({
    'stdx': stdx,
    'stdy': stdy,
    'stdz': stdz,
    'stdrho': stdrho,
    'stddelta': stddelta,
})

std.to_csv(r'C:\Users\LOCCO\Project_Curie\lambdadna\07_10_2026\lamelle_3\std_lam3_0707.csv', index=False)


# In[6]:


#Commençons par obtenir les std obtenus avec les threshholds arbitraires voir si ça tient la route
#Init
nb_id = len(X_masked_b[:,0])
not_counted = np.ones(nb_id, dtype=bool)

th_lat= 6
th_axial = 12 #Obtenus après calcul de std


for i in tqdm(range(nb_id), desc="Processing points"):
    if not_counted[i]:
        indices =  recursive_call(not_counted,i,X_masked_b, rho_masked_b, delta_masked_b, frame_masked_b, previous_index=np.array([i]),th_lat=th_lat, th_axial= th_axial)
        if len(indices)>1:

            #Collapse les doublons into their average position, keep it at the last index, and erase the others by setting them to NaN.
            X_masked_b[indices[-1], 0] = np.mean(X_masked_b[indices, 0])
            X_masked_b[indices[:-1], 0] = np.nan
            X_masked_b[indices[-1], 1] = np.mean(X_masked_b[indices, 1])
            X_masked_b[indices[:-1], 1] = np.nan
            X_masked_b[indices[-1], 2] = np.mean(X_masked_b[indices, 2])
            X_masked_b[indices[:-1], 2] = np.nan

            sigma_masked_b[indices[-1], 0] = np.mean(sigma_masked_b[indices, 0])
            sigma_masked_b[indices[:-1], 0] = np.nan
            sigma_masked_b[indices[-1], 1] = np.mean(sigma_masked_b[indices, 1])
            sigma_masked_b[indices[:-1], 1] = np.nan
            sigma_masked_b[indices[-1], 2] = np.mean(sigma_masked_b[indices, 2])
            sigma_masked_b[indices[:-1], 2] = np.nan

            rho_masked_b[indices[-1]] = np.mean(rho_masked_b[indices])
            rho_masked_b[indices[:-1]] = np.nan

            delta_masked_b[indices[-1]] = np.mean(delta_masked_b[indices])
            delta_masked_b[indices[:-1]] = np.nan

            frame_masked_b[indices[-1]] = np.mean(frame_masked_b[indices])
            frame_masked_b[indices[:-1]] = np.nan

print('removed ', len(np.where(np.isnan(X_masked[:,0]))[0]), ' over ', len(X_masked[:,0]))


X_treated = np.array([X_masked_b[~np.isnan(X_masked_b[:,0]), 0] , X_masked_b[~np.isnan(X_masked_b[:,1]), 1] ,X_masked_b[~np.isnan(X_masked_b[:,2]), 2]])
sigma_treated = np.array([sigma_masked_b[~np.isnan(sigma_masked_b[:,0]), 0] , sigma_masked_b[~np.isnan(sigma_masked_b[:,1]), 1] ,sigma_masked_b[~np.isnan(sigma_masked_b[:,2]), 2]])
rho_treated = rho_masked_b[~np.isnan(rho_masked_b)]
delta_treated = delta_masked_b[~np.isnan(delta_masked_b)]
frame_treated = frame_masked_b[~np.isnan(frame_masked_b)]


# In[41]:


os.makedirs(r'C:\Users\LOCCO\Project_Curie\labmeeting', exist_ok=True)


# In[7]:


data_filtered = pd.DataFrame({
    'frame': frame_treated,
    'x [nm]': X_treated[0, :],
    'y [nm]': X_treated[1, :],
    'z [nm]': X_treated[2, :],
    'rho': rho_treated,
    'delta' : delta_treated,
})

data_filtered.to_csv(r'C:\Users\LOCCO\Project_Curie\lambdadna\07_10_2026\lamelle_1\500_datatreated_lam1_acq3_0707.csv', index=False)


# In[8]:


#Import data treated
data_treated = pd.read_csv(r'C:\Users\LOCCO\Project_Curie\lambdadna\07_10_2026\lamelle_1\500_datatreated_lam1_acq3_0707.csv',sep=',')
frame_treated = data_treated['frame'].values
X_treated = (data_treated[['x [nm]', 'y [nm]', 'z [nm]']].values).T
rho_treated = data_treated['rho'].values
delta_treated = data_treated['delta'].values


# In[6]:


#Hemispherical angle distribution
def rho_distribution(rho):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(122, projection='polar')

    rho_wrapped = np.concatenate([rho, rho + 180]) % 360
    theta = np.deg2rad(rho_wrapped)

    n, bins_rho, patches = ax.hist(theta, bins=360, range=(0, 2*np.pi))
    # Color each bar by its hue
    for patch, left_edge in zip(patches, bins_rho[:-1]):
        hue = left_edge / 180.0
        patch.set_facecolor(hsv_to_rgb([[hue, 1.0, 1.0]])[0])

    plt.tight_layout()
    ax.set_title('Rho distribution (°)')

    return ax


# In[13]:


fig, ax = plt.subplots(figsize=(4, 2.5))
ax = rho_distribution(rho_masked_b)
plt.tight_layout()
plt.show()


# In[ ]:


import matplotlib
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import hsv_to_rgb


def plot_semicircle_colorbar(ax=None, title='ρ (°)'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw=dict(projection=None))

    n_segments = 180
    theta = np.linspace(0, np.pi, n_segments + 1)  # 0 to π (semi-circle)

    for i in range(n_segments):
        angle_deg = i  # 0 to 179 degrees
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

# --- Use it standalone ---
fig, ax = plt.subplots(figsize=(4, 2.5))
plot_semicircle_colorbar(ax)
plt.tight_layout()
plt.show()

from matplotlib.patches import FancyArrowPatch

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


fig = plt.figure(figsize=(14, 6))

# Main scatter
ax_scatter = fig.add_axes([0.05, 0.1, 0.7, 0.85])  # [left, bottom, width, height]
norm = matplotlib.colors.Normalize(vmin=0, vmax=180)
sc = ax_scatter.scatter(X_masked_b[:,0], X_masked_b[:,1],
                        c=rho_masked_b, cmap=matplotlib.colormaps['hsv'],
                        norm=norm, s=1)
ax_scatter.set_aspect('equal')
ax_scatter.set_title('Localizations colored by ρ')
# After your scatter:
add_scale_bar(ax_scatter, length_mum=1)  # 1 µm bar

# Semi-circle colorbar
ax_cbar = fig.add_axes([0.75, 0.2, 0.22, 0.6])
plot_semicircle_colorbar(ax_cbar, title='ρ (°)')

plt.show()


# In[ ]:


print(rho)


# In[10]:


get_ipython().run_line_magic('matplotlib', 'qt')
from matplotlib.path import Path
import matplotlib as mpl
from matplotlib.widgets import LassoSelector


plt.rcParams['figure.figsize'] = [15,15]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
norm = mpl.colors.Normalize(vmin=20., vmax=160.)
vals = rho_treated
sc = ax.scatter(X_treated[0, :], X_treated[1, :], X_treated[2, :], c=vals , cmap='hsv', norm=norm, s=0.1)
ax.axis('equal')
cb = plt.colorbar(sc)
x , y , z = X_treated[0, :], X_treated[1, :] , X_treated[2, :]
cb = plt.colorbar(sc)
cb.ax.invert_yaxis()
points = np.column_stack((x, y))
mask2 = np.zeros(len(x), dtype=bool) 

def onselect(verts):
    global mask2
    path = Path(verts)
    mask2 = path.contains_points(points)
    print(mask2)

lasso = LassoSelector(ax, onselect)
plt.show()

angles = np.deg2rad(rho_treated[mask2])
bins = 18
counts, bin_edges = np.histogram(angles, bins=bins, density=True)

bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
width = bin_edges[1] - bin_edges[0]
# Duplicate with π shift
bin_centers_full = np.concatenate([bin_centers, bin_centers + np.pi])
counts_full = np.concatenate([counts, counts])

# ----- Figure layout -----
fig = plt.figure(figsize=(10, 5))

# Polar subplot
ax1 = fig.add_subplot(1, 2, 1, projection='polar')

ax1.bar(bin_centers_full, counts_full, width=width, alpha=0.8)

ax1.set_theta_zero_location("E")   # 0° to the right
ax1.set_theta_direction(1)         # anti-clockwise
ax1.set_yticklabels([])
plt.tight_layout()
plt.show()


# In[ ]:


from matplotlib.patches import Rectangle
import matplotlib
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import hsv_to_rgb

def plot_semicircle_colorbar(ax=None, title='ρ (°)'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw=dict(projection=None))

    n_segments = 180
    theta = np.linspace(0, np.pi, n_segments + 1)  # 0 to π (semi-circle)

    for i in range(n_segments):
        angle_deg = i  # 0 to 179 degrees
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

# --- Use it standalone ---
fig, ax = plt.subplots(figsize=(4, 2.5))
plot_semicircle_colorbar(ax)
plt.tight_layout()
plt.show()

from matplotlib.patches import FancyArrowPatch

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


fig = plt.figure(figsize=(14, 6))

# Main scatter
ax_scatter = fig.add_axes([0.05, 0.1, 0.7, 0.85])  # [left, bottom, width, height]
norm = matplotlib.colors.Normalize(vmin=0, vmax=180)
sc = ax_scatter.scatter(X_treated[0,:], X_treated[1,:],
                        c=rho_treated, cmap=matplotlib.colormaps['hsv'],
                        norm=norm, s=0.01)
ax_scatter.set_aspect('equal')
ax_scatter.set_title('Localizations colored by ρ')
# After your scatter:
add_scale_bar(ax_scatter, length_mum=1)  # 1 µm bar

# Semi-circle colorbar
ax_cbar = fig.add_axes([0.75, 0.2, 0.22, 0.6])
plot_semicircle_colorbar(ax_cbar, title='ρ (°)')

plt.show()
from matplotlib.patches import Rectangle

y_center = 8000
y_thickness = 500
x_min, x_max = 11500, 16000

rect = Rectangle(
    (x_min, y_center - y_thickness / 2),  # coin bas-gauche
    x_max - x_min,                          # largeur limitée
    y_thickness,                             # hauteur = épaisseur de la slice
    linewidth=1.5,
    edgecolor='red',
    facecolor='red',
    alpha=0.15,
    zorder=5
)
ax_scatter.add_patch(rect)

# Lignes de bord
ax_scatter.axhline(y_center - y_thickness / 2, xmin=..., xmax=..., color='red', lw=1, ls='--', zorder=6)
ax_scatter.axhline(y_center + y_thickness / 2, xmin=..., xmax=..., color='red', lw=1, ls='--', zorder=6)

ax_scatter.plot([x_min, x_max], [y_center - y_thickness / 2, y_center - y_thickness / 2],
                color='white', lw=1, ls='--', zorder=6)
ax_scatter.plot([x_min, x_max], [y_center + y_thickness / 2, y_center + y_thickness / 2],
                color='white', lw=1, ls='--', zorder=6)

