#!/usr/bin/env python
# coding: utf-8

# Brute force search - Set up optique
# 
# On a indépendament: 
# 
# M = M_obj * f4 * (f2/f1f3)
# 
# BFP = bfp_obj/f_tl *(f1f3/f2)
# 
# C'est à dire: M* BFP = cste *f4
# 
# C'est ce qu'on cherche à résoudre. 

# In[ ]:


#Toutes les distances sont en mm

import numpy as np

#Données de l'objectif
f_tubelens = 200
M_obj = 100 
na = 1.4
n =1.52
f_obj = 2
bfp_obj = 2* np.tan(np.arcsin(na/n)) * f_obj

cste = M_obj*bfp_obj/f_tubelens

#Optiques
ca_vawp = 20
ca_vwp = 21.5
ca_woll =19.5

#Definitions des intervalles de recherche pour M, BFP et F4
M_min, M_max = 27, 40
bfp_min, bfp_max = 5, min(ca_vawp, ca_vwp, ca_woll)
f4 = 150

grillage_M = 1
grillage_bfp = 0.1
grillage_f4 = 50

M_val = np.arange(M_min, M_max + grillage_M, grillage_M)
bfp_val = np.arange(bfp_min, bfp_max + grillage_bfp, grillage_bfp)


#Brute force pour trouver les solutions: bfp = cste*f4 / M
solution = (0,0,0)

for a in M_val:
    for c in bfp_val:

        #solutions.append((a, b, c))

#print(min)
#print(f"Nombre de solutions : {len(solutions)}")
#print(solutions)

#Nombre de solutions : 5
#[(np.int64(36), np.int64(150), np.float64(19.709006869985004)), (np.int64(37), np.int64(150), np.float64(19.176331008634058)), (np.int64(38), np.int64(150), np.float64(18.671690718933164)), (np.int64(39), np.int64(150), np.float64(18.192929418447697)), (np.int64(40), np.int64(150), np.float64(17.738106182986506))]


# On a trouvé que pour f4 = 150 et M=37 on a une BFP2 = 19.2 qui est compatibel avec les optiques
# 
# Retrouvons les focales maintenant (brute force)
# 
# F2/F1F3 = M/(M_obj*F4)

# In[8]:


#

solutions = []
M_obj = 100 
target = 38/(M_obj*150)

optiques_possibles = [75,80,100,125,150,200,250,300]

for a in optiques_possibles:
    for b in optiques_possibles:
        for c in optiques_possibles:
            if abs(a/(b*c) - target) < 1e-4:  # exact
                solutions.append((a, b, c))

#F2 = a, F1 et F3 = b ou c
print(solutions)



# In[ ]:




