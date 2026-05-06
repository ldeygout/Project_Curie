#!/usr/bin/env python
# coding: utf-8

# # DSF Simulation pour Bumblebee (radpol)
# 
# Très fortement inspirée du travail d'Amaury Autric dans https://github.com/aautric/4polarMFM_these/blob/main/simu_PSF_polarMFM.py

# In[1]: Imports 

import numpy as np
from numpy.random import normal, poisson
import matplotlib.pyplot as plt



# In[3]: Emetteurs générateurs -> base de grillage à la BFP

def vectorial_BFP(N, NA, n1,n2):
    '''
    input:
    N: taille du grillage souhaité à la BFP

    returns: 
    x, y :  grillage au niveau de la BFP
    th1: angles d'incidence dans le milieu d'immersion (huile) correspondant à chaque point du grillage 
    r, phi: coordonnées polaires du grillage
    EX = [EX0, EX1, EX2] : polarisation en x pour les trois émetteurs générateurs (ie selon ux, uy, uz) pour chaque point du grillage
    EY = [EY0, EY1, EY2] :  polarisation en y pour (ux, uy, uz) pour chaque point du grillage
    r_cut : rayon de cutoff à la BFP
    '''
    #Définitions
    r_cut = min(NA/n1, n2/n1) #rayon de cutoff à la BFP normalisé par f_obj*tan(angle_ouverture)
    x, y = np.meshgrid(np.linspace(-r_cut,r_cut,N), np.linspace(-r_cut,r_cut,N))
    r = np.sqrt(x**2+y**2)
    phi = np.arctan2(y,x)
    phi[r>=r_cut] = 0

    th1 = np.zeros((N,N))
    th2 = np.zeros((N,N))
    th1[r<r_cut] = np.arcsin(r[r<r_cut]) # r = sin(th1) cf pg 116 Lousie
    th2[r<r_cut] = np.arcsin((n1/n2)*r[r<r_cut]) #angles du milieu d'échantillon (avant réfraction) correspondant à chaque point du grillage

    #Coefficients de Fresnel pour chaque point du grillage
    costh2 = np.cos(th2).astype(complex)
    Ts = (2*n2*costh2) / (n2*costh2 + n1*np.cos(th1))
    Tp = (2*n2*costh2) / (n2*np.cos(th1) + n1*costh2)

    #Champs électriques à la BFP pour les trois émetteurs générateurs
    Ex0 = ((n1/n2) * ((np.cos(th1)/costh2)*Ts*(np.sin(phi)**2) + Tp*(np.cos(phi)**2)*np.cos(th1)))/np.sqrt(np.cos(th1))
    Ex1 = (-((n1*np.sin(2*phi))/(2*n2))*((np.cos(th1)*Ts)/costh2 - Tp*np.cos(th1)))/np.sqrt(np.cos(th1))
    Ex2 = (-((n1/n2)**2)*(np.cos(th1)/costh2)*Tp*np.cos(phi)*np.sin(th1))/np.sqrt(np.cos(th1))
    Ex0[r>=r_cut], Ex1[r>=r_cut], Ex2[r>=r_cut]= 0,0,0

    Ey0 = (-0.5*np.sin(2*phi)*(n1/n2)*((np.cos(th1)/costh2)*Ts - Tp*np.cos(th1)))/np.sqrt(np.cos(th1))
    Ey1 = ((n1/n2)*((np.cos(th1)/costh2)*Ts*(np.cos(phi)**2)+Tp*np.cos(th1)*(np.sin(phi)**2)))/np.sqrt(np.cos(th1))
    Ey2 = (-((n1/n2)**2)*(np.cos(th1)/costh2)*Tp*np.sin(phi)*np.sin(th1))/np.sqrt(np.cos(th1))
    Ey0[r>=r_cut], Ey1[r>=r_cut], Ey2[r>=r_cut]= 0,0,0

    return x, y, th1, phi, [Ex0, Ex1, Ex2], [Ey0, Ey1, Ey2], r, r_cut

# In[ ]:Phases cumulées, inspirée de la Thèse de Louise p. 117 et de la thèse d'Amaury p. 13

#Terme de phase issu d'un désaxage latéral de l'émetteur 
#Une translation latérale correspond à multiplier par une exponentielle complexe dans le plan de Fourier (BFP)
def psi_lat(x,y,theta,phi, lambd, n1):
    '''
    input: 
    x,y : position de l'émetteur dans le plan focal (désaxage)
    theta, phi : description du champ objet en coordonnées polaires dans la BFP (theta est relié à r et phi est la coordonnée azimuthale)
    lambd : longueur d'onde en µm

    returns:
    psi_lat : terme de phase à ajouter
    '''
    lambd = 10**(-3)*lambd
    k = (2*np.pi*n1)/(lambd)
    return np.sin(theta)*(x*np.cos(phi)+y*np.sin(phi))*k

#Terme de phase issu d'un décalage axial de l'émetteur au plan focal, défini à partir de l'interface
def psi_z(theta,z, lambd, n1, n2):
    '''
    input: 
    z : position de l'émetteur selon l'axe optique(décalage) à partir de l'interface
    theta: description du champ objet en coordonnées polaires dans la BFP (theta est relié à r, pas besoin de la coordonnée azimuthale phi)
    lambd : longueur d'onde en µm


    returns:
    psi_z : terme de phase à ajouter
    '''
    lambd = 10**(-3)*lambd
    return 2*np.pi*n2*z*np.sqrt(1-(n1*np.sin(theta)/n2)**2)/lambd

#Terme de phase issu de la propagation dans le milieu biologique avant la réfraction avec la lamelle d'observation
def psi_d(theta, d, lambd, n1):
    '''
    input: 
    theta: description du champ objet en coordonnées polaires dans la BFP (theta est relié à r, pas besoin de la coordonnée azimuthale phi)
    d : distance entre la lamelle d'observation et le plan focal (milieu biologique)
    PAR CONVENTION d EST ALGEBRIQUEMENT NEGATIF
    lambd : longueur d'onde en µm

    returns:
    psi_d : terme de phase à ajouter
    '''
    lambd = 10**(-3)*lambd
    if d<0:
        return 2*np.pi*n1*np.cos(theta)*d/lambd
    else:
        return ValueError('d doit être négatif')

# In[ ]: Pour un meilleur traitement, éviter les effets de bord d'une TF finie en faisant du padding. Cf p. 14 thèse Amaury

def padding_depuis_BFP(r_cut, N, lambd, f_tube, f_obj, mag_obj, mag_total, l_pixel, n1):
    '''
    input:
    r_cut: rayon de cutoff à la BFP
    N: nombre de pixels de discrétisation de la BFP
    l_pixel: taille du pixel en micromètres
    NA: ouverture numérique
    mag_obj: grandissement objectif
    lambd: longueur d'onde en micromètres
    mag_total: grandissemtn total du microscope

    returns:
    Npadding: le nombre de zeros à ajouter de chaque côté pour que la taille du pixel dans l'espace réel corresponde à la taille du pixel en caméra
    '''
    #On passe tout en µm
    lambd = 0.001*lambd 
    f_tube_pad = f_tube*1000 
    f_obj_pad = f_obj*1000
    k = (2*np.pi*n1)/(lambd)

    #Cf thèse Amaury p.14, discrétization de la BFP
    Dx = 2*r_cut*f_obj_pad/N 

    #On calcule Ntot correspondant à la discretization qu'on a fait de la BFP, on en déduit Npadding en enlevant N
    Npadding = int((2*np.pi*(mag_total/mag_obj)*f_tube_pad)/(k*l_pixel*Dx)) - N

    if Npadding%2==1:
        Npadding=Npadding+1

    return Npadding, ((2*np.pi*(mag_total/mag_obj)*f_tube_pad)/(k*l_pixel*Dx)-N)-Npadding

def padding_depuis_FOV(r_cut, Ntot, lambd, f_tube, f_obj, mag_obj, mag_total, l_pixel, n1):
    '''
    input:
    r_cut: rayon de cutoff à la BFP
    Ntot: nombre de pixels total dans le FOV
    l_pixel: taille du pixel en micromètres
    NA: ouverture numérique
    mag_obj: grandissement objectif
    lambd: longueur d'onde en micromètres
    mag_total: grandissemtn total du microscope

    returns:
    Npadding: le nombre de zeros à ajouter de chaque côté pour que la taille du pixel dans l'espace réel corresponde à la taille du pixel en caméra
    '''
    #On passe tout en µm
    lambd = 0.001*lambd 
    f_tube_pad = f_tube*1000 
    f_obj_pad = f_obj*1000
    k = (2*np.pi*n1)/(lambd)

    #Cf thèse Amaury p.14, discrétization de la BFP
    N_bfp = int(Ntot/((2*np.pi*(mag_total/mag_obj)*f_tube_pad)/(k*l_pixel*2*r_cut*f_obj_pad)-1))

    if N_bfp%2==1:
        N_bfp=N_bfp+1

    return N_bfp


def pad(a, n):
    '''
    inputs: 
    a : array à pad
    n : nombre de zeros à ajouter de chaque côté

    returns:
    b : array paddée
    '''
    n0 = a.shape[1]
    type = a.dtype

    if len(a.shape)==2:
        b = np.zeros((n+n0, n+n0)).astype(type)
        b[:] = 0.
        b[n//2:-n//2,n//2:-n//2] = a

    if len(a.shape)==3:
        b = np.zeros((a.shape[0], n+n0, n+n0)).astype(type)
        b[:] = 0.
        b[:,n//2:-n//2,n//2:-n//2] = a

    return b

# In[ ]: Maintenant on rentre dans le dur: calculons M, la matrice qui nous permettra de faire le lien entre la BFP et la PSF. M est construite à partir de la base génératrice au niveau de la BFP, EX et EY (microscope dependant). 

def compute_M(xp, yp, zp, d, th1, phi, Ex0, Ex1, Ex2, Ey0, Ey1, Ey2,n1 , n2, pixel_sur_camera, polar_projections=None, lambd=617, BFP_visualisation=False):

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

# In[ ]:Visualisons la PSF d'un dipole ayant une orientation. 

def PSF(rho, eta, delta, d, M, N_photons=1000):
    '''
    inputs: 
    rho: inclinaison du dipôle projeté sur le plan focal 
    eta: angle entre le dipôle et l'axe optique z
    delta: demi-angle du cône de wobbling du dipole
    M: matrice de calibration permmetant le passage de la BFP du microscope à la PSF
    N_photons: nombre de photons émis par le dipôle

    returns:
    psf : plan image en polarisation x et y, shape (2, Nx, Ny)
        Pour N émetteurs, shape is (N, 2, Nx, Ny)
        Pour plusieurs plans d'étude, shape is (N, nbre_plans, 2_polar, Nx, Ny)

    Cette fonction correspond à l'équation 11 de la thèse d'Amaury
    '''
    #Passage de degrés à radians
    rho = rho * np.pi / 180
    eta = eta * np.pi / 180
    delta = delta * np.pi / 180

    if (len(rho)!=len(eta))or(len(rho)!=len(delta))or(len(eta)!=len(delta))or(len(rho)!=len(N_photons)):
        print(len(rho),len(eta),len(delta))
        raise ValueError("Pb de dimensionnement dans les longueurs des arrays")

    if len(rho)!=M.shape[0]:
        raise ValueError("Pb de dimensionnement dans les longueurs des arrays")

    a = len(rho)
    b = len(d)
    psf_arr = np.zeros((a,b,2, M.shape[-1],M.shape[-1]))

    for j in range(a):
        #Expression de R non triviale: la rotation en 3D n'est pas un espace commmutatif cf Thèse Louise p.107
        R = np.array([
                [(np.sin(rho[j])**2)*(1 - np.cos(eta[j])) + np.cos(eta[j]), 
                np.sin(rho[j]) * np.cos(rho[j]) * (np.cos(eta[j]) - 1), 
                np.cos(rho[j]) * np.sin(eta[j])],
                [np.sin(rho[j]) * np.cos(rho[j]) * (np.cos(eta[j]) - 1), 
                (np.cos(rho[j])**2)*(1 - np.cos(eta[j])) + np.cos(eta[j]), 
                np.sin(rho[j]) * np.sin(eta[j])],
                [-np.cos(rho[j]) * np.sin(eta[j]), 
                -np.sin(rho[j]) * np.sin(eta[j]), 
                np.cos(eta[j])]
            ])

        #Valeurs propres après diagonalisation de l'intégrale sur le cône de wobbling, cf Thèse Louise p. 108
        lamb_delta = np.array([
                (1 - np.cos(delta[j] / 2)) * (np.cos(delta[j] / 2) + 2) / 6, 
                (1 - np.cos(delta[j] / 2)) * (np.cos(delta[j] / 2) + 2) / 6, 
                ((np.cos(delta[j] / 2)**3 - 1) / (np.cos(delta[j] / 2) - 1)) / 3
            ])

        #Plusieurs plans d'imagerie
        for i in range(b):
            #np.real pour faire sauter ranger le psf en type float, et faire sauter les erreurs d'arrondis près
            psfx = np.real(np.einsum('a, auv -> uv', lamb_delta, np.moveaxis(np.diagonal(np.einsum('ab, bcuv -> acuv', R.T, np.einsum('abuv, bc -> acuv', M[j,i,0,:,:,:,:], R)), axis1=0, axis2=1), -1, 0)))
            psfy = np.real(np.einsum('a, auv -> uv', lamb_delta, np.moveaxis(np.diagonal(np.einsum('ab, bcuv -> acuv', R.T, np.einsum('abuv, bc -> acuv', M[j,i,1,:,:,:,:], R)), axis1=0, axis2=1), -1, 0)))
            psf = np.array([psfx, psfy])
            psf[psf<0] = 0 #Au cas où

            #Car on a pas pris en compte la norme du dipôle depuis le début
            norm = np.sum(psf)
            psf_arr[j,i,:,:,:] = psf * (N_photons[j] / norm)

    return psf_arr

