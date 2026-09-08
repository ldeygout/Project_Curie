#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'qt')
from simu_dsf import *
from numpy.random import normal, poisson
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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

#Définitions des deux bras du radpol 
polar_projections = np.array(['radphi',0])


# In[ ]:


N = 110 #discretization de la BFP, numériquement optimisé


# In[ ]:


x, y, th1, phi, [Ex0, Ex1, Ex2], [Ey0, Ey1, Ey2], r, r_cut= vectorial_BFP(N, NA, n1,n2)
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


# In[ ]:


# 11 émetteurs espacés de 1µm en x et y, même z, rho, delta, N_photons
xp =  np.ones(11) * 0 #np.ones(10) * 0  #       # [-5, -3.89, -2.78, ..., 5] µm
yp =   np.linspace(-5, 5, 11)#      # [-5, -3.89, -2.78, ..., 5] µm
zp = np.ones(11) * 1.0            # z = 1 µm pour tous
d = np.array([-1.3,-1.8])         
rho =  np.ones(11) * 0             # rho = 45° pour tous
eta = np.ones(11) * 90    # eta distribué uniformément entre 0° et 90°
delta = np.ones(11) * 180.0     #np.linspace(10, 180, 10)           # delta = 10 pour tous
N_photons = np.ones(11, dtype=int) * 5000  # 5000 photons par dipôle

#Paramètres ensuite pour la visualisation
img_shape = th1.shape[-2:]          # (ny, nx)
cx, cy = img_shape[1] / 2, img_shape[0] / 2  # centre en pixels
xp_px = cx + xp / pixel_sur_camera
yp_px = cy + yp / pixel_sur_camera


# In[ ]:


#Obtention de la base de PSF, la matrice M
M = compute_M(xp,yp,zp,d,th1,phi,Ex0,Ex1,Ex2,Ey0,Ey1,Ey2, n1,n2, pixel_sur_camera, polar_projections = polar_projections, lambd=lambd)
#Maintenant, ajoutons l'effet dipôle au lieu de simple émetteur et obtenons le plan image
#Att, la taille affichée constitue celle du padding. la vraie fov est de 4096x2304 pixels
psf = PSF(rho,eta,delta,d, M,N_photons)


# In[ ]:


from scipy.optimize import curve_fit

def gaussian2d(xy, amp, x0, y0, sigma_x, sigma_y, offset):
    x, y = xy
    return (offset + amp * np.exp(
        -((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2))
    )).ravel()

def fit_psf_center(psf_rows, col, x0c, x1c, y0c, y1c, cx_im, cy_im):
    """Somme les 4 canaux et fit une gaussienne 2D sur le crop."""
    crop_sum = sum(psf_rows[row][col, y0c:y1c, x0c:x1c] for row in range(4))

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

