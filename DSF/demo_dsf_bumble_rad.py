#!/usr/bin/env python
# coding: utf-8

# # Démo simulation DSF microscope Bumbleblee (bras radpol)
# 
# Toutes les données propres au microscope sont déjà présentes dans le dossier simu_dsf_bubmble_rad. 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'qt')
from simu_dsf import *
from numpy.random import normal, poisson
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit


# In[2]:


#Détails microscope

lambd = 617 #longueur d'onde de travailen nm
n1 = 1.52 #indice du verre et de l'huile de contact
n2 = 1.33 #indice de l'échantillon biologique est celui de l'eau
h = 6.626*10**(-34) # Planck 
c = 299792458 #vitesse de la lumière

#Nikon Eclipse Ti2-E
f_tube = 200
f_obj = 2
mag_obj = 100
NA = 1.4

#Système relai de lentilles
mag_total = 37.5

#Hamamatsu CMOS
l_pixel = 4.6 #taille pixel - µm
largeur_pixel = 4096 #largeur du capteur - pixel
hauteur_pixel = 2304 #hauteur du capteur - pixel


# In[3]:


4.578/mag_total


# In[3]:


N = 110 #discretization de la BFP, numériquement optimisé

#On peut également chercher à obtenir une taille de visualisation finale de la même taille que la FOV réelle en caméra, pour cela: 
#N_tot = max(largeur_pixel, hauteur_pixel) // pixel_sur_camera #On veut que l'image finale soit équivalente à la FOV
#N = padding_depuis_FOV(r_cut, Ntot, lambd, f_tube, f_obj, mag_obj, mag_total, l_pixel, n1)


# In[4]:


x, y, th1, phi, [Ex0, Ex1, Ex2], [Ey0, Ey1, Ey2], r, r_cut= vectorial_BFP(N, NA, n1,n2)


# In[5]:


#Padding
Npad,test, lp_prim = padding_depuis_BFP(r_cut, N, lambd, f_tube, f_obj, mag_obj, mag_total, l_pixel, n1)
th1 = pad(th1, Npad)
phi = pad(phi, Npad)
Ex0 = pad(Ex0, Npad)
Ex1 = pad(Ex1, Npad)
Ex2 = pad(Ex2, Npad)
Ey0 = pad(Ey0, Npad)
Ey1 = pad(Ey1, Npad)
Ey2 = pad(Ey2, Npad)


# In[7]:


lp_prim


# In[6]:


pixel_sur_camera_prim = lp_prim/mag_total # µm/pixel
pixel_sur_camera = l_pixel/mag_total # µm/pixel


# In[7]:


#Définitions des deux bras du radpol 
polar_projections = np.array([0,'radphi'])


# In[8]:


# 10 émetteurs espacés de 1µm en x et y, même z, rho, delta, N_photons
xp =  np.ones(5) * 0 #np.ones(10) * 0  #       # [-5, -3.89, -2.78, ..., 5] µm
yp =   np.ones(5) * 0 #      # [-5, -3.89, -2.78, ..., 5] µm
zp = np.ones(5) * 1.0            # z = 1 µm pour tous
d = np.array([-1.3, -1.8])         
rho = np.ones(5) * 30         # rho = 45° pour tous
eta = np.linspace(0, 90, 5) #np.ones(5) * 80          # eta distribué uniformément entre 0° et 90°
delta = np.ones(5) * 75.0     #np.linspace(10, 180, 10)           # delta = 10 pour tous
N_photons = np.ones(5) * 5000  # 5000 photons par dipôle

#Paramètres ensuite pour la visualisation
img_shape = th1.shape[-2:]          # (ny, nx)
cx, cy = (img_shape[1]) / 2, (img_shape[0]) / 2  # centre en pixels
xp_px = cx + xp / pixel_sur_camera_prim
yp_px = cy + yp / pixel_sur_camera_prim


# In[ ]:


print(cx, cy)


# In[9]:


#Obtention de la base de PSF, la matrice M
M = compute_M(xp,yp,zp,d,th1,phi,Ex0,Ex1,Ex2,Ey0,Ey1,Ey2, n1,n2, pixel_sur_camera, polar_projections = polar_projections, lambd=lambd)


# In[10]:


#Maintenant, ajoutons l'effet dipôle au lieu de simple émetteur et obtenons le plan image
#Att, la taille affichée constitue celle du padding. la vraie fov est de 4096x2304 pixels
psf = PSF(rho,eta,delta,d, M,N_photons)


# In[17]:


plt.close('all')

get_ipython().run_line_magic('matplotlib', 'qt')

def gaussian2d(xy, amp, x0, y0, sigma_x, sigma_y, offset):
    x, y = xy
    return (offset + amp * np.exp(
        -((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2))
    )).ravel()

def fit_psf_center(psf_rows, col, x0c, x1c, y0c, y1c, cx_im, cy_im,pixel_sur_camera):
    """Somme les 4 canaux et fit une gaussienne 2D sur le crop."""
    crop_sum = sum(psf_rows[row][col, y0c:y1c, x0c:x1c] for row in range(2))

    ny, nx = crop_sum.shape
    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y)

    p0 = [crop_sum.max(), nx/2, ny/2, 3, 3, crop_sum.min()]
    try:
        popt, _ = curve_fit(gaussian2d, (xx, yy), crop_sum.ravel(), p0=p0)
        # Centre en pixels dans le repère global
        x_fit_px = x0c + popt[1]
        y_fit_px = y0c + popt[2]
        # Conversion en µm (origine au centre de l'image)
        x_fit_um = (x_fit_px - cx_im) * pixel_sur_camera
        y_fit_um = (y_fit_px - cy_im) * pixel_sur_camera
        return x_fit_um, y_fit_um, popt
    except RuntimeError:
        print(f"Fit échoué pour émetteur {col}")
        return None, None, None


half = 15 #centre de psf qu'on attend des données du microscope
n_emetteurs = len(xp)
param_label = "eta"
param_unite = "°"
param_values = eta

row_labels = ['rad', 'phi', '0°', '90°']
psf_rows = [
    psf[:, 0, 0, :, :],
    psf[:, 0, 1, :, :],
    psf[:, 1, 0, :, :],
    psf[:, 1, 1, :, :],
]
n_rows = 4
'''

row_labels = ['rad', 'phi']
psf_rows = [
    psf[:, 0, 0, :, :],
    psf[:, 0, 1, :, :],

]

n_rows = 2
'''
#fig = plt.figure(figsize=(n_emetteurs * 1.8, n_rows * 1.8 + 0.6))
fig = plt.figure(figsize=(4, 6))

gs = fig.add_gridspec(n_rows + 1, n_emetteurs + 1,  # +1 colonne pour colorbar
                      width_ratios=[1] * n_emetteurs + [0.05],
                      height_ratios=[0.3] + [1] * n_rows,
                      hspace=0.05, wspace=0.05)

'''
# --- Flèche ---
ax_arrow = fig.add_subplot(gs[0, :-1])
ax_arrow.set_xlim(0, 1)
ax_arrow.set_ylim(0, 1)
ax_arrow.axis('off')
ax_arrow.annotate('', xy=(0.95, 0.5), xytext=(0.05, 0.5),
                  arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax_arrow.text(0.5, 0.85, f'{param_label} : {param_values[0]:.0f} {param_unite} → {param_values[-1]:.0f}{param_unite}',
              ha='center', va='center', fontsize=10)
'''
# --- Calcul vmin/vmax commun ---


vmin = min(r.min() for r in psf_rows)
vmax = max(r.max() for r in psf_rows)

# --- Crops ---
im_ref = None
fit_results = np.zeros((n_emetteurs, 2))

for col in range(n_emetteurs):
    xc = int(round(cx + xp[col] / pixel_sur_camera))  # attention x→col
    yc = int(round(cy + yp[col] / pixel_sur_camera))  # y→row

    x0, x1 = xc - half, xc + half
    y0, y1 = yc - half, yc + half

    # Clamp aux bords
    x0c, x1c = max(0, x0), min(img_shape[1], x1)
    y0c, y1c = max(0, y0), min(img_shape[0], y1)

    x_fit, y_fit, popt = fit_psf_center(psf_rows, col, x0c, x1c, y0c, y1c,cx, cy,pixel_sur_camera)

    fit_results[col] = (x_fit, y_fit)

    for row in range(n_rows):
        ax = fig.add_subplot(gs[row + 1, col])


        crop = psf_rows[row][col, y0c:y1c, x0c:x1c]

        im_ref = ax.imshow(crop, origin='lower', vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_xticks([])
        ax.set_yticks([])

        # Croix rouge — coordonnées dans le repère du crop
        xp_px_crop = cx + xp[col] / pixel_sur_camera - x0c
        yp_px_crop = cy + yp[col] / pixel_sur_camera - y0c
        #ax.plot(xp_px_crop, yp_px_crop, 'r+', markersize=6, markeredgewidth=1.0)


        # Croix verte = position fittée
        x_fit, y_fit = fit_results[col]
        if x_fit is not None:
            x_fit_crop = cx + x_fit / pixel_sur_camera - x0c
            y_fit_crop = cy + y_fit / pixel_sur_camera - y0c
            #ax.plot(x_fit_crop, y_fit_crop, 'g+', markersize=6, markeredgewidth=1.0)

        # Label ligne à gauche
        if col == 0:
            ax.set_ylabel(row_labels[row], fontsize=9, rotation=0,
                          labelpad=30, va='center')

        # Valeur param en dessous dernière ligne
        if row == n_rows - 1:
            ax.set_xlabel( f'Eta:{eta[col]:.0f}{param_unite}', fontsize=10)
            #f'Rho:{rho[col]:.0f}{param_unite} 

# --- Colorbar commune ---
cax = fig.add_subplot(gs[1:, -1])
fig.colorbar(im_ref, cax=cax)

# Dictionnaire de tous les paramètres
all_params = {
    'xp': (xp, 'µm'),
    'yp': (yp, 'µm'),
    'zp': (zp, 'µm'),
    'ρ':  (rho, '°'),
    'η':  (eta, '°'),
    'δ':  (delta, ''),
    'N_photons': (N_photons, ''),
}

# Construit le titre : paramètre variable omis, les autres à l'indice 0
title_parts = []
for name, (values, unit) in all_params.items():
    if name == param_label:
        continue  # on skip le paramètre qui varie
    title_parts.append(f'{name}={values[0]}{unit}')

#fig.suptitle('Emitters: ' + ', '.join(title_parts)+f' NA={NA}', fontsize=10)
plt.show()


# In[18]:


def gaussian2d(xy, amp, x0, y0, sigma_x, sigma_y, offset):
    x, y = xy
    return (offset + amp * np.exp(
        -((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2))
    )).ravel()


def fit_psf_center(psf, cx_im, cy_im, pixel_sur_camera_prim):
    #Pour une seule psf
    nx, ny = np.shape(psf)
    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y)

    idx_max = np.unravel_index(psf.argmax(), psf.shape)

    #plt.figure()
    #plt.imshow(psf)

    p0 = [psf.max(), idx_max[1], idx_max[0], 3, 3, psf.min()]
    #plt.plot(p0[1], p0[2], 'r+', markersize=10, label='p0')

    try:
        popt, _ = curve_fit(gaussian2d, (xx, yy), psf.ravel(), p0=p0)
        # Conversion en µm (origine au centre de l'image)
        x_fit_um = (popt[1] - cx_im) * pixel_sur_camera_prim
        y_fit_um = (popt[2] - cy_im) * pixel_sur_camera_prim
        #plt.plot(popt[1], popt[2], 'g+', markersize=10, label='p0')
        #plt.show()
        return x_fit_um, y_fit_um, popt, p0

    except RuntimeError:
        print(f"Fit échoué pour émetteur")
        return None, None, None


# In[36]:


print(np.shape(psf[0,0,0,:,:]))
print(np.shape(psf.sum(axis=1)))
print(np.shape(psf.sum(axis=1).sum(axis=1)))
plt.close("all")


# In[37]:


# 10 émetteurs espacés de 1µm en x et y, même z, rho, delta, N_photons
xp =  np.ones(1) * 0 #np.ones(10) * 0  #       # [-5, -3.89, -2.78, ..., 5] µm
yp =   np.ones(1) * 0#      # [-5, -3.89, -2.78, ..., 5] µm
zp = np.ones(1) * 1.0            # z = 1 µm pour tous
d = np.array([-1.3,-1.8])         
rho =  np.ones(1) * 0             # rho = 45° pour tous
eta = np.ones(1) * 45    # eta distribué uniformément entre 0° et 90°
delta = np.ones(1) * 180.0     #np.linspace(10, 180, 10)           # delta = 10 pour tous
N_photons = np.ones(1, dtype=int) * 5000  # 5000 photons par dipôle

#Paramètres ensuite pour la visualisation


#xp_px = cx + xp / pixel_sur_camera_prim
#yp_px = cy + yp / pixel_sur_camera_prim    
#yp = np.linspace(-(1+j), 1+j, 10)
M = compute_M(xp,yp,zp,d,th1,phi,Ex0,Ex1,Ex2,Ey0,Ey1,Ey2, n1,n2, pixel_sur_camera, polar_projections = polar_projections, lambd=lambd)
psf = PSF(rho,eta,delta,d, M,N_photons)
psf_2d = psf.sum(axis=1).sum(axis=1)[0, :, :]  # shape (208, 208)
idx_max = np.unravel_index(psf_2d.argmax(), psf_2d.shape)  # ← use psf_2d.shape, not psf.shape

print(f"PSF peak at pixel: {idx_max}")
print(f"Image center: {cx, cy}")
print(f"Difference: {idx_max[0] - cx, idx_max[1] - cy}")
x,y,p,p0 = fit_psf_center(psf.sum(axis=1).sum(axis=1)[0,:,:],cx,cy,pixel_sur_camera_prim)
print(f"Gaussian fitting: {cx + x / pixel_sur_camera_prim, cy + y / pixel_sur_camera_prim}")
#print(f"Positions d'émetteurs: {yp}")
#print(f"Erreur guess initial (argmax): {(((p0[2]-cy)*pixel_sur_camera_prim-yp[i])*100)/yp[i]}%"))
'''erreur[i] = abs(((y-yp[i])))
plt.plot(yp, erreur, '--', c=rgb_colors[j], label = f"yp E [{-(1+j)}, {1+j}]")
plt.grid()
plt.legend()
plt.title(f"Evolution de l'erreur pour N = {N} optimisé et longueur de pixel pris en compte, pour différents emplacements d'émetteurs")
plt.show()'''


# In[90]:


#Check fit gaussien
from matplotlib.colors import hsv_to_rgb
hues = np.linspace(0, 1, 10)
hsv_colors = np.stack((hues, np.ones_like(hues), np.ones_like(hues)), axis=1)
rgb_colors = hsv_to_rgb(hsv_colors)
N=110
#plt.close("all")
print(N)

test = 40
# 10 émetteurs espacés de 1µm en x et y, même z, rho, delta, N_photons
xp =  np.ones(test) * 0
yp =  np.ones(test) * 3 #np.ones(10) * 0  #       # [-5, -3.89, -2.78, ..., 5] µm
         # z = 1 µm pour tous
d = np.array([-1.3,-1.8])         
rho =  np.ones(test) * 0             # rho = 45° pour tous
eta = np.ones(test) * 90    # eta distribué uniformément entre 0° et 90°
delta = np.ones(test) * 180.0     #np.linspace(10, 180, 10)           # delta = 10 pour tous
N_photons = np.ones(test, dtype=int) * 5000  # 5000 photons par dipôle
erreur = np.ones(len(xp))
plt.figure()
zp = np.linspace(0,5, test)
M = compute_M(xp,yp,zp,d,th1,phi,Ex0,Ex1,Ex2,Ey0,Ey1,Ey2, n1,n2, pixel_sur_camera, polar_projections = polar_projections, lambd=lambd)
psf = PSF(rho,eta,delta,d, M,N_photons)
for i in range(len(xp)): 
    x,y,p,p0 = fit_psf_center(psf.sum(axis=1).sum(axis=1)[i,:,:],cx,cy,pixel_sur_camera_prim)
    #print(f"Positions d'émetteurs: {yp}")
    #print(f"Erreur guess initial (argmax): {(((p0[2]-cy)*pixel_sur_camera_prim-yp[i])*100)/yp[i]}%"))
    if yp[i]!=0:
        erreur[i] = ((y/yp[i]))
plt.plot(zp, erreur, '--', c=rgb_colors[0], label = f"yp = 3, y/yp = {erreur}")
plt.ylim(0.9,1.1)
plt.grid()
plt.xlabel("Position en z")
plt.legend()
plt.title(f"Tracé de y_fit_gauss/yp pour N = {N} en fonction de sa position en z")
plt.show()
        #print(f"Erreur Gaussien: {((y-yp[i])*100)/yp[i]}%  Erreur guess initial (argmax): {(((p0[2]-cy)*pixel_sur_camera_prim-yp[i])*100)/yp[i]}%")


# In[80]:


erreur


# In[72]:


for y_true in [-6, -3, 0, 3, 6]:
    # single emitter at each position
    yp_test = np.array([float(y_true)])
    xp =  np.ones(1) * 0 #np.ones(10) * 0  #       # [-5, -3.89, -2.78, ..., 5] µm
    zp = np.ones(1) * 1.0            # z = 1 µm pour tous
    d = np.array([-1.3,-1.8])         
    rho =  np.ones(1) * 0             # rho = 45° pour tous
    eta = np.ones(1) * 90    # eta distribué uniformément entre 0° et 90°
    delta = np.ones(1) * 180.0     #np.linspace(10, 180, 10)           # delta = 10 pour tous
    N_photons = np.ones(1, dtype=int) * 5000  # 5000 photons par dipôle
    M = compute_M(xp,yp_test,zp,d,th1,phi,Ex0,Ex1,Ex2,Ey0,Ey1,Ey2, n1,n2, pixel_sur_camera, polar_projections = polar_projections, lambd=lambd)
    psf_test = PSF(rho,eta,delta,d, M,N_photons)

    psf_2d = psf_test.sum(axis=1).sum(axis=1)
    plt.imshow(psf_2d[0,:,:])
    x_fit, y_fit, _, _ = fit_psf_center(psf_2d[0,:,:], cx, cy, pixel_sur_camera_prim)
    print(f"y_true={y_true:.1f} | y_fit={y_fit:.3f} | error={y_fit-y_true:.3f} | ratio={y_fit/y_true if y_true!=0 else 'N/A'}")


# In[74]:


1-pixel_sur_camera/pixel_sur_camera_prim


# In[19]:


fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.3, hspace=0.4)

psf_sum = psf.sum(axis=0)
vmin = psf_sum.min()
vmax = psf_sum.max()

#Dans le cas où on visualise toute la FOV, faut cropper le côté pas carré
    #Nx = int(np.ceil(FoV_x / pixel_sur_camera))
    #Ny = int(np.ceil(FoV_y / pixel_sur_camera))

    #cx, cy = N_total // 2, N_total // 2

    #psf_crop = psf[cy - Ny//2 : cy + Ny//2,cx - Nx//2 : cx + Nx//2]

titles = [[f'PSF d={d[i]}, {("rad" if p == "radphi" else f"{int(p)}°")}', 
           f'PSF d={d[i]}, {("phi" if p == "radphi" else f"{int(p)+90}°")}']
          for i, p in enumerate(polar_projections)]

axes = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)]


fit_x, fit_y = np.zeros(len(xp)),np.zeros(len(xp))

for i in range(len(xp)):
    fit_x[i],fit_y[i],p,p0 = fit_psf_center(psf.sum(axis=1).sum(axis=1)[0,:,:],cx,cy,pixel_sur_camera_prim)

for i in range(2):
    for j in range(2):
        im = axes[i][j].imshow(psf_sum[i, j, :, :], vmin=vmin, vmax=vmax, origin='lower')
        axes[i][j].set_title(titles[i][j], pad=8)
        axes[i][j].plot(cx + xp / pixel_sur_camera_prim, cy + yp / pixel_sur_camera_prim, 'r+', markersize=2, markeredgewidth=1)

        axes[i][j].plot(cx + fit_x / pixel_sur_camera_prim, cy + fit_y / pixel_sur_camera_prim, 'g+', markersize=2, markeredgewidth=1)

cax = fig.add_subplot(gs[:, 2])
fig.colorbar(im, cax=cax)

info_lines = [
    f"Emitter {k+1}: xp={xp[k]} µm, yp={yp[k]} µm, zp={zp[k]} µm,"
    f"ρ={rho[k]}°, η={eta[k]}°, δ={delta[k]}°, N_photons={N_photons[k]}"
    for k in range(len(xp))
]
fig.text(0.5, 0.01, '\n'.join(info_lines), ha='center', va='bottom',
         fontsize=11, family='monospace')

plt.subplots_adjust(bottom=0.25)
plt.show()


# In[71]:


yp_px ,xp_px


# In[72]:


fit_y, fit_x


# In[42]:


plt.close('all')



def gaussian2d(xy, amp, x0, y0, sigma_x, sigma_y, offset):
    x, y = xy
    return (offset + amp * np.exp(
        -((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2))
    )).ravel()

def fit_psf_center(psf_rows, col, x0c, x1c, y0c, y1c, cx_im, cy_im,pixel_sur_camera):
    """Somme les 4 canaux et fit une gaussienne 2D sur le crop."""
    crop_sum = sum(psf_rows[row][col, y0c:y1c, x0c:x1c] for row in range(2))

    ny, nx = crop_sum.shape
    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y)

    p0 = [crop_sum.max(), nx/2, ny/2, 3, 3, crop_sum.min()]
    try:
        popt, _ = curve_fit(gaussian2d, (xx, yy), crop_sum.ravel(), p0=p0)
        # Centre en pixels dans le repère global
        x_fit_px = x0c + popt[1]
        y_fit_px = y0c + popt[2]
        # Conversion en µm (origine au centre de l'image)
        x_fit_um = (x_fit_px - cx_im) * pixel_sur_camera
        y_fit_um = (y_fit_px - cy_im) * pixel_sur_camera
        return x_fit_um, y_fit_um, popt
    except RuntimeError:
        print(f"Fit échoué pour émetteur {col}")
        return None, None, None


half = 15 #centre de psf qu'on attend des données du microscope
n_emetteurs = len(xp)
param_label = "rho"
param_unite = "°"
param_values = rho

'''row_labels = ['rad', 'phi', '0°', '90°']
psf_rows = [
    psf[:, 0, 0, :, :],
    psf[:, 0, 1, :, :],
    psf[:, 1, 0, :, :],
    psf[:, 1, 1, :, :],
]'''

row_labels = ['rad', 'phi']
psf_rows = [
    psf[:, 0, 0, :, :],
    psf[:, 0, 1, :, :],

]

n_rows = 2
fig = plt.figure(figsize=(n_emetteurs * 1.8, n_rows * 1.8 + 0.6))

gs = fig.add_gridspec(n_rows + 1, n_emetteurs + 1,  # +1 colonne pour colorbar
                      width_ratios=[1] * n_emetteurs + [0.05],
                      height_ratios=[0.3] + [1] * n_rows,
                      hspace=0.05, wspace=0.05)

# --- Flèche ---
ax_arrow = fig.add_subplot(gs[0, :-1])
ax_arrow.set_xlim(0, 1)
ax_arrow.set_ylim(0, 1)
ax_arrow.axis('off')
ax_arrow.annotate('', xy=(0.95, 0.5), xytext=(0.05, 0.5),
                  arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax_arrow.text(0.5, 0.85, f'{param_label} : {param_values[0]:.0f} {param_unite} → {param_values[-1]:.0f}{param_unite}',
              ha='center', va='center', fontsize=10)

# --- Calcul vmin/vmax commun ---


vmin = min(r.min() for r in psf_rows)
vmax = max(r.max() for r in psf_rows)

# --- Crops ---
im_ref = None
fit_results = np.zeros((n_emetteurs, 2))

for col in range(n_emetteurs):
    xc = int(round(cx + xp[col] / pixel_sur_camera))  # attention x→col
    yc = int(round(cy + yp[col] / pixel_sur_camera))  # y→row

    x0, x1 = xc - half, xc + half
    y0, y1 = yc - half, yc + half

    # Clamp aux bords
    x0c, x1c = max(0, x0), min(img_shape[1], x1)
    y0c, y1c = max(0, y0), min(img_shape[0], y1)

    x_fit, y_fit, popt = fit_psf_center(psf_rows, col, x0c, x1c, y0c, y1c,cx, cy,pixel_sur_camera)

    fit_results[col] = (x_fit, y_fit)

    for row in range(n_rows):
        ax = fig.add_subplot(gs[row + 1, col])


        crop = psf_rows[row][col, y0c:y1c, x0c:x1c]

        im_ref = ax.imshow(crop, origin='lower', vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_xticks([])
        ax.set_yticks([])

        # Croix rouge — coordonnées dans le repère du crop
        xp_px_crop = cx + xp[col] / pixel_sur_camera - x0c
        yp_px_crop = cy + yp[col] / pixel_sur_camera - y0c
        #ax.plot(xp_px_crop, yp_px_crop, 'r+', markersize=6, markeredgewidth=1.0)


        # Croix verte = position fittée
        x_fit, y_fit = fit_results[col]
        if x_fit is not None:
            x_fit_crop = cx + x_fit / pixel_sur_camera - x0c
            y_fit_crop = cy + y_fit / pixel_sur_camera - y0c
            #ax.plot(x_fit_crop, y_fit_crop, 'g+', markersize=6, markeredgewidth=1.0)

        # Label ligne à gauche
        if col == 0:
            ax.set_ylabel(row_labels[row], fontsize=9, rotation=0,
                          labelpad=30, va='center')

        # Valeur param en dessous dernière ligne
        if row == n_rows - 1:
            ax.set_xlabel(f'{param_values[col]:.0f}{param_unite}', fontsize=8)


# --- Colorbar commune ---
cax = fig.add_subplot(gs[1:, -1])
fig.colorbar(im_ref, cax=cax)

# Dictionnaire de tous les paramètres
all_params = {
    'xp': (xp, 'µm'),
    'yp': (yp, 'µm'),
    'zp': (zp, 'µm'),
    'ρ':  (rho, '°'),
    'η':  (eta, '°'),
    'δ':  (delta, ''),
    'N_photons': (N_photons, ''),
}

# Construit le titre : paramètre variable omis, les autres à l'indice 0
title_parts = []
for name, (values, unit) in all_params.items():
    if name == param_label:
        continue  # on skip le paramètre qui varie
    title_parts.append(f'{name}={values[0]}{unit}')

fig.suptitle('Emitters: ' + ', '.join(title_parts)+f'NA={NA}', fontsize=10)
plt.show()


# In[ ]:


abs(fit_results[:,1]/yp) #N=80


# In[ ]:


abs(fit_results[:,1]/yp) #N=110 et lp corrige lors de l'affichage


# In[ ]:


abs(lp_prim/l_pixel)


# In[ ]:


abs(fit_results[:,1]/yp) #N=110


# In[ ]:


def fit_psf_center(psf_rows, col, x0c, x1c, y0c, y1c, cx_im, cy_im):
    crop_sum = sum(psf_rows[row][col, y0c:y1c, x0c:x1c] for row in range(4))
    ny, nx = crop_sum.shape
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    p0 = [crop_sum.max(), nx/2, ny/2, 3, 3, crop_sum.min()]
    try:
        popt, _ = curve_fit(gaussian2d, (xx, yy), crop_sum.ravel(), p0=p0)
        x_fit_um = (x0c + popt[1] - cx_im) * pixel_sur_camera
        y_fit_um = (y0c + popt[2] - cy_im) * pixel_sur_camera
        return x_fit_um, y_fit_um, popt
    except RuntimeError:
        print(f"Fit échoué pour émetteur {col}")
        return np.nan, np.nan, None

NA = np.linspace(0.7,1.4,10)
polar_projections = np.array(['radphi',0])

# Allez un emetteur tfacons je sais que c'est constant
xp =  np.ones(1) * 0 #np.ones(10) * 0  #       # [-5, -3.89, -2.78, ..., 5] µm
yp =  np.ones(1) * 5 #      # [-5, -3.89, -2.78, ..., 5] µm
zp = np.ones(1) * 1.0            # z = 1 µm pour tous
d = np.array([-1.3,-1.8])         
rho =  np.ones(1) * 0             # rho = 45° pour tous
eta = np.ones(1) * 90    # eta distribué uniformément entre 0° et 90°
delta = np.ones(1) * 180.0     #np.linspace(10, 180, 10)           # delta = 10 pour tous
N_photons = np.ones(1) * 5000  # 5000 photons par dipôle
half = 15

coeff = np.zeros(10)
test = np.zeros(10)

for i in range(10):
    x, y, th1, phi, [Ex0, Ex1, Ex2], [Ey0, Ey1, Ey2], r, r_cut= vectorial_BFP(N, NA[i], n1,n2)
    Npad, test[i] = padding_depuis_BFP(r_cut, N, lambd, f_tube, f_obj, mag_obj, mag_total, l_pixel, n1)
    th1 = pad(th1, Npad)
    phi = pad(phi, Npad)
    Ex0 = pad(Ex0, Npad)
    Ex1 = pad(Ex1, Npad)
    Ex2 = pad(Ex2, Npad)
    Ey0 = pad(Ey0, Npad)
    Ey1 = pad(Ey1, Npad)
    Ey2 = pad(Ey2, Npad)
    #Obtention de la base de PSF, la matrice M
    M = compute_M(xp,yp,zp,d,th1,phi,Ex0,Ex1,Ex2,Ey0,Ey1,Ey2, n1,n2, pixel_sur_camera, polar_projections = polar_projections, lambd=lambd)
    psf = PSF(rho,eta,delta,d, M,N_photons)
    psf_rows = [
    psf[:, 0, 0, :, :],
    psf[:, 0, 1, :, :],
    psf[:, 1, 0, :, :],
    psf[:, 1, 1, :, :],]

    #Paramètres ensuite pour la visualisation
    img_shape = psf.shape[-2:]
    cx_im, cy_im = img_shape[1] / 2, img_shape[0] / 2

    n_emetteurs = len(xp)
    fit_results = np.zeros((n_emetteurs, 2))

    for col in range(n_emetteurs):
        xc = int(round(cx_im + xp[col] / pixel_sur_camera))
        yc = int(round(cy_im + yp[col] / pixel_sur_camera))

        x0c, x1c = max(0, xc - half), min(img_shape[1], xc + half)
        y0c, y1c = max(0, yc - half), min(img_shape[0], yc + half)

        x_fit, y_fit, popt = fit_psf_center(psf_rows, col, x0c, x1c, y0c, y1c, cx_im, cy_im)


    coeff[i] = yp[0]/y_fit

plt.plot(NA, coeff, 'r--', markersize=2, markeredgewidth=1.5)

plt.suptitle('Evolution du rapport entre emplacement issue de la TF (fit gaussien) et celui donné par le grossissement total, en fonction de NA', fontsize=10)
plt.show()
plt.figure()
plt.plot(NA, test, 'g--', markersize=2, markeredgewidth=1.5)
plt.suptitle('Evolution de (2*np.pi*(mag_total/mag_obj)*f_tube_pad)/(k*l_pixel*Dx)-N)-Npadding, en fonction de NA', fontsize=10)

plt.show()


# In[ ]:


N = np.linspace(40,150,111)

plt.figure()
for i in N: 
    Npad,test = padding_depuis_BFP(r_cut, i, lambd, f_tube, f_obj, mag_obj, mag_total, l_pixel, n1)
    plt.plot(i, test,  'r+', markersize=2, markeredgewidth=1.5 )

plt.axhline(y=0)
plt.show()


# In[25]:


import pandas as pd
N = np.linspace(40,150,111)

erreur = np.ones(len(N))
lpp = np.ones(len(N))
plt.figure()
for i in N: 
    Npad,test, lp_prim = padding_depuis_BFP(r_cut, i, lambd, f_tube, f_obj, mag_obj, mag_total, l_pixel, n1)
    erreur[int(i-40)] = ((lp_prim-l_pixel)/l_pixel) * 100
    lpp[int(i-40)] = lp_prim

idx = np.argmin(np.abs(erreur))
plt.plot(N, erreur, color='steelblue', linewidth=1.5)
plt.plot(N, erreur, 'r+', markersize=2, markeredgewidth=1.5)
plt.axhline(y=0, color = "gray", linestyle = '--')
plt.annotate(f'({int(N[70])}, {erreur[70]:.5f})',
             xy=(N[70], erreur[70]),                   # position du point
             xytext=(N[70] + 5, erreur[70] + 0.5),     # position du texte
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=9)
plt.ylabel("Pourcentage d'erreur relative")
plt.xlabel("Discrétisation de la BFP")
#plt.title(f"Discrétisation de la BFP minimisant l'erreur d'arrondi entier pour la taille de pixel vaut {N[41]}")
#plt.show()

std = pd.DataFrame({
    'Discretisation_N': N,
    'Longuer_pixel_prim': lpp,
    'Erreur_relative': erreur,
})

std.to_csv(r'C:\Users\LOCCO\Project_Curie\discretization_lp_4.6um.csv', index=False)


# In[71]:


N = np.arange(20, 150)

polar_projections = np.array(['radphi',0])
# 10 émetteurs espacés de 1µm en x et y, même z, rho, delta, N_photons
xp =  np.ones(10) * 0 #np.ones(10) * 0  #       # [-5, -3.89, -2.78, ..., 5] µm
yp =   np.linspace(-1.5, 1.5, 10)#      # [-5, -3.89, -2.78, ..., 5] µm
zp = np.ones(10) * 1.0            # z = 1 µm pour tous
d = np.array([-1.3,-1.8])         
rho =  np.ones(10) * 0             # rho = 45° pour tous
eta = np.ones(10) * 90    # eta distribué uniformément entre 0° et 90°
delta = np.ones(10) * 180.0     #np.linspace(10, 180, 10)           # delta = 10 pour tous
N_photons = np.ones(10, dtype=int) * 5000  # 5000 photons par dipôle

test = np.ones(len(N))
dot_px = np.ones(len(N))
dot_mum = np.ones(len(N))


for i in N: 
    x, y, th1, phi, [Ex0, Ex1, Ex2], [Ey0, Ey1, Ey2], r, r_cut= vectorial_BFP(int(i), NA, n1,n2)

    Npad,test[int(i-N[0])], lp_prim = padding_depuis_BFP(r_cut, int(i), lambd, f_tube, f_obj, mag_obj, mag_total, l_pixel, n1)
    #erreur[int(i-40)] = ((lp_prim-l_pixel)/l_pixel) * 100

    th1 , phi , Ex0, Ex1 , Ex2, Ey0, Ey1, Ey2 = pad(th1, Npad), pad(phi, Npad) , pad(Ex0, Npad), pad(Ex1, Npad), pad(Ex2, Npad), pad(Ey0, Npad), pad(Ey1, Npad), pad(Ey2, Npad)

    pixel_sur_camera_prim = lp_prim/mag_total # µm/pixel
    pixel_sur_camera = l_pixel/mag_total # µm/pixel

    #Paramètres ensuite pour la visualisation
    img_shape = th1.shape[-2:]          # (ny, nx)
    cx, cy = img_shape[1] / 2, img_shape[0] / 2  # centre en pixels

    xp_px = cx + xp / pixel_sur_camera_prim
    yp_px = cy + yp / pixel_sur_camera_prim


    M = compute_M(xp,yp,zp,d,th1,phi,Ex0,Ex1,Ex2,Ey0,Ey1,Ey2, n1,n2, pixel_sur_camera, polar_projections = polar_projections, lambd=lambd)
    psf = PSF(rho,eta,delta,d, M,N_photons)
    fit_x, fit_y = np.zeros(len(xp)),np.zeros(len(xp))

    for j in range(len(xp)):
        fit_x[j],fit_y[j],p,p0 = fit_psf_center(psf.sum(axis=1).sum(axis=1)[j,:,:],cx,cy,pixel_sur_camera_prim)

    dot_mum[int(i-N[0])] =np.mean((fit_y - yp))



plt.plot(N, np.abs(dot_mum), 'r+')
plt.plot(N,  np.abs(dot_mum), color='steelblue', linewidth=1.5)
plt.grid()    
plt.ylabel("mean(fit(y)-y)")
plt.xlabel("Discrétisation de la BFP")
plt.title(f"Fit gaussien Evolution de l'erreur (loc en µm) en fonction de la discrétisation de la BFP")
plt.show()
plt.figure()
#idx = np.argmin(np.abs(erreur))
plt.plot(N, np.abs(dot_mum/(test*100)), color='steelblue', linewidth=1.5)
plt.plot(N, np.abs(dot_mum/(test*100)), 'r+', markersize=2, markeredgewidth=1.5)
plt.axhline(y=1, color = "gray", linestyle = '--')
plt.grid()
#plt.annotate(f'({int(N[41])}, {erreur[41]:.5f})',
#             xy=(N[41], erreur[41]),                   # position du point
#             xytext=(N[41] + 5, erreur[41] + 0.5),     # position du texte
#             arrowprops=dict(arrowstyle='->', color='black'),
#             fontsize=9)
plt.ylabel("Rapport entre erreur de localisation et erreur de d'arrondissement de N, en fonction de N")
plt.xlabel("Discrétisation de la BFP")
plt.title(f"erreur localisation / erreur arrondi")
plt.show()
plt.figure()
plt.plot(N, np.abs(test*100), color='steelblue', linewidth=1.5)
plt.plot(N, np.abs(test*100), 'r+', markersize=2, markeredgewidth=1.5)
plt.axhline(y=0, color = "gray", linestyle = '--')
plt.ylabel("Pourcentage d'erreur de d'arrondi ((2*np.pi*(mag_total/mag_obj)*f_tube_pad)/(k*l_pixel*Dx)-(N+Npadding))/(N+Npadding)*100")
plt.xlabel("Discrétisation de la BFP")
plt.title(f"Evolution de l'erreur d'arrondi selon N. Minimum à {np.argmin(np.abs(test))+N[0]}")
plt.show()



# In[25]:


plt.plot(N, test, color='steelblue', linewidth=1.5)
plt.plot(N, test, 'r+', markersize=2, markeredgewidth=1.5)
plt.axhline(y=0, color = "gray", linestyle = '--')
#plt.annotate(f'({int(N[41])}, {erreur[41]:.5f})',
#             xy=(N[41], erreur[41]),                   # position du point
#             xytext=(N[41] + 5, erreur[41] + 0.5),     # position du texte
#             arrowprops=dict(arrowstyle='->', color='black'),
#             fontsize=9)
plt.ylabel("Pourcentage d'erreur")
plt.xlabel("Discrétisation de la BFP")
#plt.title(f"Discrétisation de la BFP minimisant l'erreur d'arrondi entier pour la taille de pixel vaut {N[41]}")
plt.show()


# In[27]:


plt.figure()
plt.plot(N, test, color='steelblue', linewidth=1.5)
plt.plot(N, test, 'r+', markersize=2, markeredgewidth=1.5)
plt.axhline(y=0, color = "gray", linestyle = '--')
#plt.annotate(f'({int(N[41])}, {erreur[41]:.5f})',
#             xy=(N[41], erreur[41]),                   # position du point
#             xytext=(N[41] + 5, erreur[41] + 0.5),     # position du texte
#             arrowprops=dict(arrowstyle='->', color='black'),
#             fontsize=9)
plt.ylabel("Pourcentage d'erreur")
plt.xlabel("Discrétisation de la BFP")
#plt.title(f"Discrétisation de la BFP minimisant l'erreur d'arrondi entier pour la taille de pixel vaut {N[41]}")
plt.show()


# In[ ]:





# In[2]:


get_ipython().run_line_magic('matplotlib', 'qt')


from simu_dsf import *
from numpy.random import normal, poisson
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
#Détails microscope
def compute_M_complet(xp, yp, zp, d, th1, phi, Ex0, Ex1, Ex2, Ey0, Ey1, Ey2,n1 , n2, pixel_sur_camera, focale_l4, polar_projections=None, lambd=617, BFP_visualisation=False):

    ''' 
    inputs:
    xp, yp, zp: coordonées du (des) dipole(s) relatifs au plan focal
    d: distance entre la lamelle et le plan focal
    th1, phi: description du champ objet en coordonnées polaires dans la BFP (theta est relié à r et phi est la coordonnée azimuthale)
    Ex0, Ex1, Ex2: base génératrice du microscope des champs électriques polarisés en x au niveau de la BFP
    Ey0, Ey1, Ey2: base génératrice du microscope des champs électriques polarisés en y au niveau de la BFP
    phase_maskx, phase_masky: masques de phase respectifs
    #A corriger cette histoire de second plane
    polar_projections: array donnant les angles de projection si différents de x,y, ainsi que rad si polarisation radpol
    lambd: longueur d'onde
    visualisation: booléen pour activer la visualisation
    BFP_visualisation: booléen pour activer la visualisation du plan focal (sans fft)
    FOCALE L4 EN MM 
    returns:
    M : matrice de calibration, permet le calcul de la PSF
        (2, 3, 3, N_pix, N_pix) pour un seule émetteur (2: pola x et pola y)
        (N, 2, 3, 3, N_pix, N_pix) si N émetteurs
        (N, K, 2, 3, 3, N_pix, N_pix) with K being the indices for planes shifted (the shift is in second_plane (negative convention again)), and projected on polarizations that are in 
        polar_projections (+90)
    '''
    focale_l4 = focale_l4 * 10**3 #Conversion en microns
    a = Ex0.shape[-1]
    b = len(d)
    c = len(xp)

    if (Ex0.shape[-1]!=th1.shape[-1]):
        print(Ex0.shape, th1.shape)
        raise ValueError("Pb de dimensionnement")

    if (len(xp)!=len(yp))or(len(xp)!=len(zp))or(len(zp)!=len(yp)):
        print(len(xp),len(yp),len(zp))
        raise ValueError("Pb de dimensionnement dans les longueurs des arrays")

    if len(polar_projections)!=len(d):
        raise ValueError("Donner un type de projection (radpol/xy) par plan imagé (d)")

    if np.any(np.abs(xp) > (Ex0.shape[1] * pixel_sur_camera) / 2) or np.any(np.abs(yp) > (Ex0.shape[0] * pixel_sur_camera) / 2):
        raise ValueError( f"Coordonnées de l'émetteur en dehors du FoV autorisé par la discrétisation de la BFP : abs(x,y) < {(Ex0.shape[1] * pixel_sur_camera) / 2}. Adapter N ou les coordonnées de l'émetteur.")

    M = np.zeros((c,b,2,3,3,a,a), dtype =complex)

    #Differents plans focaux
    for i in range(b):
        #Quel type de projection en polarisation? Cas d'un radphi
        if polar_projections[i] == 'radphi':
            #for pl in range(len(second_plane)): ça ça sera pour plusieurs émetteurs
            ex0, ey0 = Ex0*np.cos(phi) +  Ey0*np.sin(phi), -Ex0*np.sin(phi) +  Ey0*np.cos(phi)
            ex1, ey1 = Ex1*np.cos(phi) +  Ey1*np.sin(phi), -Ex1*np.sin(phi) +  Ey1*np.cos(phi)
            ex2, ey2 = Ex2*np.cos(phi) +  Ey2*np.sin(phi), -Ex2*np.sin(phi) +  Ey2*np.cos(phi)

        #Au cas où on projette sur différents axes de polarisation (cas MFM....)
        else:
            angle = float(polar_projections[i]) * np.pi / 180
            ex0, ey0 = np.cos(angle)*Ex0 + np.sin(angle)*Ey0 , -np.sin(angle)*Ex0 + np.cos(angle)*Ey0
            ex1, ey1 = np.cos(angle)*Ex1 + np.sin(angle)*Ey1,  -np.sin(angle)*Ex1 + np.cos(angle)*Ey1
            ex2, ey2 = np.cos(angle)*Ex2 + np.sin(angle)*Ey2, -np.sin(angle)*Ex2 + np.cos(angle)*Ey2

        #Différents émetteurs
        for j in range(c):

            #Terme de phase total
            phase = np.exp(1j*(psi_d(th1, d[i], lambd, n1)+psi_z(th1, zp[j],  lambd, n1, n2)+psi_lat_complet(xp[j],yp[j],th1,phi, lambd, n1,focale_l4)))
            phase[np.isnan(phase)] = 0

            E00 = np.fft.fftshift(np.fft.fft2(ex0*phase))
            E01 = np.fft.fftshift(np.fft.fft2(ex1*phase))
            E02 = np.fft.fftshift(np.fft.fft2(ex2*phase))

            E10 = np.fft.fftshift(np.fft.fft2(ey0*phase))
            E11= np.fft.fftshift(np.fft.fft2(ey1*phase))
            E12 = np.fft.fftshift(np.fft.fft2(ey2*phase))

            if BFP_visualisation:
                ''' base de PSF au niveau de la BFP, avant propagation'''
                Ex_bfp = np.array([ex0*phase, ex1*phase, ex2*phase])
                Ey_bfp = np.array([ey0*phase, ey1*phase, ey2*phase])
                Mx = np.einsum('abc, ubc -> aubc', np.conj(Ex_bfp), Ex_bfp) 
                My = np.einsum('abc, ubc -> aubc', np.conj(Ey_bfp), Ey_bfp)

            else:
                '''base de PSF au niveau du plan image, après propagation'''
                Ex_im = np.array([E00, E01, E02])
                Ey_im = np.array([E10, E11, E12])

                Mx = np.einsum('abc, ubc -> aubc', np.conj(Ex_im), Ex_im) 
                My = np.einsum('abc, ubc -> aubc', np.conj(Ey_im), Ey_im)

            M[j,i,:,:,:,:,:] = np.array([Mx, My])

    return M

def gaussian2d(xy, amp, x0, y0, sigma_x, sigma_y, offset):
    x, y = xy
    return (offset + amp * np.exp(
        -((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2))
    )).ravel()


def fit_psf_center(psf, cx_im, cy_im, pixel_sur_camera_prim):
    #Pour une seule psf
    nx, ny = np.shape(psf)
    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y)

    idx_max = np.unravel_index(psf.argmax(), psf.shape)

    #plt.figure()
    #plt.imshow(psf)

    p0 = [psf.max(), idx_max[1], idx_max[0], 3, 3, psf.min()]
    #plt.plot(p0[1], p0[2], 'r+', markersize=10, label='p0')

    try:
        popt, _ = curve_fit(gaussian2d, (xx, yy), psf.ravel(), p0=p0)
        # Conversion en µm (origine au centre de l'image)
        x_fit_um = (popt[1] - cx_im) * pixel_sur_camera_prim
        y_fit_um = (popt[2] - cy_im) * pixel_sur_camera_prim
        #plt.plot(popt[1], popt[2], 'g+', markersize=10, label='p0')
        #plt.show()
        return x_fit_um, y_fit_um, popt, p0

    except RuntimeError:
        print(f"Fit échoué pour émetteur")
        return None, None, None

lambd = 617 #longueur d'onde de travailen nm
n1 = 1.52 #indice du verre et de l'huile de contact
n2 = 1.33 #indice de l'échantillon biologique est celui de l'eau
h = 6.626*10**(-34) # Planck 
c = 299792458 #vitesse de la lumière

#Nikon Eclipse Ti2-E
f_tube = 200
f_obj = 2
mag_obj = 100
NA = 1.4
f4 = 150 #mm
#Système relai de lentilles
mag_total = 37.5

#Hamamatsu CMOS
l_pixel = 4.6 #taille pixel - µm
largeur_pixel = 4096 #largeur du capteur - pixel
hauteur_pixel = 2304 #hauteur du capteur - pixel
N = 110 #discretization de la BFP, numériquement optimisé
x, y, th1, phi, [Ex0, Ex1, Ex2], [Ey0, Ey1, Ey2], r, r_cut= vectorial_BFP(N, NA, n1,n2)
#Padding
Npad,test, lp_prim = padding_depuis_BFP(r_cut, N, lambd, f_tube, f_obj, mag_obj, mag_total, l_pixel, n1)
th1 = pad(th1, Npad)
phi = pad(phi, Npad)
Ex0 = pad(Ex0, Npad)
Ex1 = pad(Ex1, Npad)
Ex2 = pad(Ex2, Npad)
Ey0 = pad(Ey0, Npad)
Ey1 = pad(Ey1, Npad)
Ey2 = pad(Ey2, Npad)

pixel_sur_camera_prim = lp_prim/mag_total # µm/pixel
pixel_sur_camera = l_pixel/mag_total # µm/pixel

test = 10

xp =  np.ones(test) * 0
yp = np.linspace(-5,5,test)
zp = np.ones(test)*1
d = np.array([-1.3,-1.8])         
rho =  np.ones(test) * 0             # rho = 45° pour tous
eta = np.ones(test) * 90    # eta distribué uniformément entre 0° et 90°
delta = np.ones(test) * 180.0     #np.linspace(10, 180, 10)           # delta = 10 pour tous
N_photons = np.ones(test, dtype=int) * 5000  # 5000 photons par dipôle
erreur = np.ones(len(xp))
erreur_complet = np.ones(len(xp))
img_shape = th1.shape[-2:]          # (ny, nx)
cx, cy = (img_shape[1]) / 2, (img_shape[0]) / 2  # centre en pixels
polar_projections = np.array(['radphi',0])

M = compute_M(xp,yp,zp,d,th1,phi,Ex0,Ex1,Ex2,Ey0,Ey1,Ey2, n1,n2, pixel_sur_camera,polar_projections = polar_projections, lambd=lambd)
psf = PSF(rho,eta,delta,d, M,N_photons)
M_complet = compute_M_complet(xp,yp,zp,d,th1,phi,Ex0,Ex1,Ex2,Ey0,Ey1,Ey2, n1,n2, pixel_sur_camera, f4,polar_projections = polar_projections, lambd=lambd)
psf_complet = PSF(rho,eta,delta,d, M_complet,N_photons)
for i in range(len(xp)): 
    x,y,p,p0 = fit_psf_center(psf.sum(axis=1).sum(axis=1)[i,:,:],cx,cy,pixel_sur_camera_prim)
    x_complet, y_complet, _,_ = fit_psf_center(psf_complet.sum(axis=1).sum(axis=1)[i,:,:],cx,cy, pixel_sur_camera_prim)
    #print(f"Positions d'émetteurs: {yp}")
    #print(f"Erreur guess initial (argmax): {(((p0[2]-cy)*pixel_sur_camera_prim-yp[i])*100)/yp[i]}%"))
    if yp[i]!=0:
        erreur[i] = ((y/yp[i]))
        erreur_complet[i] = ((y_complet/yp[i]))
plt.figure()
plt.plot(yp, erreur, '--', c='r', label = f" y/yp = {erreur}")
plt.plot(yp, erreur_complet, '--', c='g', label = f"phi_lat sans approximation petits angles: y_complet/yp = {erreur_complet}")
plt.ylim(0.9,1.1)
plt.grid()
plt.xlabel("Position en y")
plt.legend()
plt.title(f"Tracé de y_fit_gauss/yp pour N = {N} en fonction de sa position en y pour phi_lat avec et sans approximations petits angles")
plt.show()


# In[5]:


get_ipython().run_line_magic('matplotlib', 'qt')


from simu_dsf import *
from numpy.random import normal, poisson
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

def compute_M_hamming(xp, yp, zp, d, th1, phi, Ex0, Ex1, Ex2, Ey0, Ey1, Ey2,n1 , n2, pixel_sur_camera, polar_projections=None, lambd=617, BFP_visualisation=False):

    ''' 
    inputs:
    xp, yp, zp: coordonées du (des) dipole(s) relatifs au plan focal
    d: distance entre la lamelle et le plan focal
    th1, phi: description du champ objet en coordonnées polaires dans la BFP (theta est relié à r et phi est la coordonnée azimuthale)
    Ex0, Ex1, Ex2: base génératrice du microscope des champs électriques polarisés en x au niveau de la BFP
    Ey0, Ey1, Ey2: base génératrice du microscope des champs électriques polarisés en y au niveau de la BFP
    phase_maskx, phase_masky: masques de phase respectifs
    #A corriger cette histoire de second plane
    polar_projections: array donnant les angles de projection si différents de x,y, ainsi que rad si polarisation radpol
    lambd: longueur d'onde
    visualisation: booléen pour activer la visualisation
    BFP_visualisation: booléen pour activer la visualisation du plan focal (sans fft)
    FOCALE L4 EN MM 
    returns:
    M : matrice de calibration, permet le calcul de la PSF
        (2, 3, 3, N_pix, N_pix) pour un seule émetteur (2: pola x et pola y)
        (N, 2, 3, 3, N_pix, N_pix) si N émetteurs
        (N, K, 2, 3, 3, N_pix, N_pix) with K being the indices for planes shifted (the shift is in second_plane (negative convention again)), and projected on polarizations that are in 
        polar_projections (+90)
    '''

    a = Ex0.shape[-1]
    b = len(d)
    c = len(xp)

    if (Ex0.shape[-1]!=th1.shape[-1]):
        print(Ex0.shape, th1.shape)
        raise ValueError("Pb de dimensionnement")

    if (len(xp)!=len(yp))or(len(xp)!=len(zp))or(len(zp)!=len(yp)):
        print(len(xp),len(yp),len(zp))
        raise ValueError("Pb de dimensionnement dans les longueurs des arrays")

    if len(polar_projections)!=len(d):
        raise ValueError("Donner un type de projection (radpol/xy) par plan imagé (d)")

    if np.any(np.abs(xp) > (Ex0.shape[1] * pixel_sur_camera) / 2) or np.any(np.abs(yp) > (Ex0.shape[0] * pixel_sur_camera) / 2):
        raise ValueError( f"Coordonnées de l'émetteur en dehors du FoV autorisé par la discrétisation de la BFP : abs(x,y) < {(Ex0.shape[1] * pixel_sur_camera) / 2}. Adapter N ou les coordonnées de l'émetteur.")

    M = np.zeros((c,b,2,3,3,a,a), dtype =complex)

    #Differents plans focaux
    for i in range(b):
        #Quel type de projection en polarisation? Cas d'un radphi
        if polar_projections[i] == 'radphi':
            #for pl in range(len(second_plane)): ça ça sera pour plusieurs émetteurs
            ex0, ey0 = Ex0*np.cos(phi) +  Ey0*np.sin(phi), -Ex0*np.sin(phi) +  Ey0*np.cos(phi)
            ex1, ey1 = Ex1*np.cos(phi) +  Ey1*np.sin(phi), -Ex1*np.sin(phi) +  Ey1*np.cos(phi)
            ex2, ey2 = Ex2*np.cos(phi) +  Ey2*np.sin(phi), -Ex2*np.sin(phi) +  Ey2*np.cos(phi)

        #Au cas où on projette sur différents axes de polarisation (cas MFM....)
        else:
            angle = float(polar_projections[i]) * np.pi / 180
            ex0, ey0 = np.cos(angle)*Ex0 + np.sin(angle)*Ey0 , -np.sin(angle)*Ex0 + np.cos(angle)*Ey0
            ex1, ey1 = np.cos(angle)*Ex1 + np.sin(angle)*Ey1,  -np.sin(angle)*Ex1 + np.cos(angle)*Ey1
            ex2, ey2 = np.cos(angle)*Ex2 + np.sin(angle)*Ey2, -np.sin(angle)*Ex2 + np.cos(angle)*Ey2

        #Différents émetteurs
        for j in range(c):

            #Terme de phase total
            phase = np.exp(1j*(psi_d(th1, d[i], lambd, n1)+psi_z(th1, zp[j],  lambd, n1, n2)+psi_lat(xp[j],yp[j],th1,phi, lambd, n1)))
            phase[np.isnan(phase)] = 0

            hamming_1d = np.hamming(a)
            hamming_2d = np.outer(hamming_1d, hamming_1d)
            phase = phase * hamming_2d 

            E00 = np.fft.fftshift(np.fft.fft2(ex0*phase))
            E01 = np.fft.fftshift(np.fft.fft2(ex1*phase))
            E02 = np.fft.fftshift(np.fft.fft2(ex2*phase))

            E10 = np.fft.fftshift(np.fft.fft2(ey0*phase))
            E11= np.fft.fftshift(np.fft.fft2(ey1*phase))
            E12 = np.fft.fftshift(np.fft.fft2(ey2*phase))

            if BFP_visualisation:
                ''' base de PSF au niveau de la BFP, avant propagation'''
                Ex_bfp = np.array([ex0*phase, ex1*phase, ex2*phase])
                Ey_bfp = np.array([ey0*phase, ey1*phase, ey2*phase])
                Mx = np.einsum('abc, ubc -> aubc', np.conj(Ex_bfp), Ex_bfp) 
                My = np.einsum('abc, ubc -> aubc', np.conj(Ey_bfp), Ey_bfp)

            else:
                '''base de PSF au niveau du plan image, après propagation'''
                Ex_im = np.array([E00, E01, E02])
                Ey_im = np.array([E10, E11, E12])

                Mx = np.einsum('abc, ubc -> aubc', np.conj(Ex_im), Ex_im) 
                My = np.einsum('abc, ubc -> aubc', np.conj(Ey_im), Ey_im)

            M[j,i,:,:,:,:,:] = np.array([Mx, My])

    return M


def gaussian2d(xy, amp, x0, y0, sigma_x, sigma_y, offset):
    x, y = xy
    return (offset + amp * np.exp(
        -((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2))
    )).ravel()


def fit_psf_center(psf, cx_im, cy_im, pixel_sur_camera_prim):
    #Pour une seule psf
    nx, ny = np.shape(psf)
    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y)

    idx_max = np.unravel_index(psf.argmax(), psf.shape)

    #plt.figure()
    #plt.imshow(psf)

    p0 = [psf.max(), idx_max[1], idx_max[0], 3, 3, psf.min()]
    #plt.plot(p0[1], p0[2], 'r+', markersize=10, label='p0')

    try:
        popt, _ = curve_fit(gaussian2d, (xx, yy), psf.ravel(), p0=p0)
        # Conversion en µm (origine au centre de l'image)
        x_fit_um = (popt[1] - cx_im) * pixel_sur_camera_prim
        y_fit_um = (popt[2] - cy_im) * pixel_sur_camera_prim
        #plt.plot(popt[1], popt[2], 'g+', markersize=10, label='p0')
        #plt.show()
        return x_fit_um, y_fit_um, popt, p0

    except RuntimeError:
        print(f"Fit échoué pour émetteur")
        return None, None, None

lambd = 617 #longueur d'onde de travailen nm
n1 = 1.52 #indice du verre et de l'huile de contact
n2 = 1.33 #indice de l'échantillon biologique est celui de l'eau
h = 6.626*10**(-34) # Planck 
c = 299792458 #vitesse de la lumière

#Nikon Eclipse Ti2-E
f_tube = 200
f_obj = 2
mag_obj = 100
NA = 1.4
f4 = 150 #mm
#Système relai de lentilles
mag_total = 37.5

#Hamamatsu CMOS
l_pixel = 4.6 #taille pixel - µm
largeur_pixel = 4096 #largeur du capteur - pixel
hauteur_pixel = 2304 #hauteur du capteur - pixel
N = 110 #discretization de la BFP, numériquement optimisé
x, y, th1, phi, [Ex0, Ex1, Ex2], [Ey0, Ey1, Ey2], r, r_cut= vectorial_BFP(N, NA, n1,n2)
#Padding
Npad,test, lp_prim = padding_depuis_BFP(r_cut, N, lambd, f_tube, f_obj, mag_obj, mag_total, l_pixel, n1)
th1 = pad(th1, Npad)
phi = pad(phi, Npad)
Ex0 = pad(Ex0, Npad)
Ex1 = pad(Ex1, Npad)
Ex2 = pad(Ex2, Npad)
Ey0 = pad(Ey0, Npad)
Ey1 = pad(Ey1, Npad)
Ey2 = pad(Ey2, Npad)

pixel_sur_camera_prim = lp_prim/mag_total # µm/pixel
pixel_sur_camera = l_pixel/mag_total # µm/pixel

test = 10

xp =  np.ones(test) * 0
yp = np.linspace(-5,5,test)
zp = np.ones(test)*1
d = np.array([-1.3,-1.8])         
rho =  np.ones(test) * 0             # rho = 45° pour tous
eta = np.ones(test) * 90    # eta distribué uniformément entre 0° et 90°
delta = np.ones(test) * 180.0     #np.linspace(10, 180, 10)           # delta = 10 pour tous
N_photons = np.ones(test, dtype=int) * 5000  # 5000 photons par dipôle
erreur = np.ones(len(xp))
erreur_complet = np.ones(len(xp))
img_shape = th1.shape[-2:]          # (ny, nx)
cx, cy = (img_shape[1]) / 2, (img_shape[0]) / 2  # centre en pixels
polar_projections = np.array(['radphi',0])

M = compute_M(xp,yp,zp,d,th1,phi,Ex0,Ex1,Ex2,Ey0,Ey1,Ey2, n1,n2, pixel_sur_camera,polar_projections = polar_projections, lambd=lambd)
psf = PSF(rho,eta,delta,d, M,N_photons)
M_complet = compute_M_hamming(xp,yp,zp,d,th1,phi,Ex0,Ex1,Ex2,Ey0,Ey1,Ey2, n1,n2, pixel_sur_camera,polar_projections = polar_projections, lambd=lambd)
psf_complet = PSF(rho,eta,delta,d, M_complet,N_photons)
for i in range(len(xp)): 
    x,y,p,p0 = fit_psf_center(psf.sum(axis=1).sum(axis=1)[i,:,:],cx,cy,pixel_sur_camera_prim)
    x_complet, y_complet, _,_ = fit_psf_center(psf_complet.sum(axis=1).sum(axis=1)[i,:,:],cx,cy, pixel_sur_camera_prim)
    #print(f"Positions d'émetteurs: {yp}")
    #print(f"Erreur guess initial (argmax): {(((p0[2]-cy)*pixel_sur_camera_prim-yp[i])*100)/yp[i]}%"))
    if yp[i]!=0:
        erreur[i] = ((y/yp[i]))
        erreur_complet[i] = ((y_complet/yp[i]))
plt.figure()
plt.plot(yp, erreur, '--', c='r', label = f" y/yp = {erreur}")
plt.plot(yp, erreur_complet, '--', c='g', label = f" y_hamming/yp = {erreur_complet}")
plt.ylim(0.9,1.1)
plt.grid()
plt.xlabel("Position en y")
plt.legend()
plt.title(f"Tracé de y_fit_gauss/yp pour N = {N} t avec et sans hamming (carré) de la fft")
plt.show()

