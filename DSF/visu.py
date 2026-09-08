# In[1]: Imports 
from scipy.optimize import curve_fit
import numpy as np

# In[1]: Fittings 
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
    

# In[1]: visualisation

half = 15 #centre de psf qu'on attend des données du microscope
n_emetteurs = len(xp)
param_label = "yp"
param_unite = "µm"
param_values = yp

row_labels = ['rad', 'phi', '0°', '90°']
psf_rows = [
    psf[:, 0, 0, :, :],
    psf[:, 0, 1, :, :],
    psf[:, 1, 0, :, :],
    psf[:, 1, 1, :, :],
]

n_rows = 4
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
    xc = int(round(cy + xp[col] / pixel_sur_camera))  # attention x→col
    yc = int(round(cx + yp[col] / pixel_sur_camera))  # y→row

    x0, x1 = xc - half, xc + half
    y0, y1 = yc - half, yc + half

    # Clamp aux bords
    x0c, x1c = max(0, x0), min(img_shape[1], x1)
    y0c, y1c = max(0, y0), min(img_shape[0], y1)
    
    x_fit, y_fit, popt = fit_psf_center(psf_rows, col, x0c, x1c, y0c, y1c,cx, cy)
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
        ax.plot(xp_px_crop, yp_px_crop, 'r+', markersize=6, markeredgewidth=1.0)


        # Croix verte = position fittée
        x_fit, y_fit = fit_results[col]
        if x_fit is not None:
            x_fit_crop = cx + x_fit / pixel_sur_camera - x0c
            y_fit_crop = cy + y_fit / pixel_sur_camera - y0c
            ax.plot(x_fit_crop, y_fit_crop, 'g+', markersize=6, markeredgewidth=1.0)

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

# In[1]: Imports 