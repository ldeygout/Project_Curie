#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports 
get_ipython().run_line_magic('matplotlib', 'qt')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 10))

# Table
l = 88
h = 45
table = [(0, h), (l, 0), (l, h), (0, 0)]


#Carre Amaury
a = 10
carre = [(0, h-l), (a, 0), (l, h), (0, 0)]

# Points intérieurs (modifiables)
points = [
    (2, 1),
    (3, 2),
    (4, 1.5),
    (5, 3),
    (6, 2),
    (7, 1),
    (4, 4),
    (6, 4.5)
]



# Grille
ax.set_xticks(np.arange(0, a + 1, 1))
ax.set_yticks(np.arange(0, b + 1, 1))
ax.grid(True)

# Rectangle
ax.add_patch(Rectangle((0, 0), a, b,
                       fill=False, edgecolor="black", linewidth=2))

# Carré dans le coin haut gauche
ax.add_patch(Rectangle((0, b - c), c, c,
                       fill=False, edgecolor="blue", linewidth=2))

# Points
xs, ys = zip(*points)
ax.scatter(xs, ys, color="red", zorder=5)

for i, (x, y) in enumerate(points, start=1):
    ax.text(x + 0.1, y + 0.1, f"P{i}", fontsize=9)

# Segments entre points successifs
for i in range(len(points) - 1):
    ax.plot(
        [points[i][0], points[i + 1][0]],
        [points[i][1], points[i + 1][1]],
        color="green",
        linewidth=1.5
    )

# Fermeture optionnelle
ax.plot(
    [points[-1][0], points[0][0]],
    [points[-1][1], points[0][1]],
    color="green",
    linestyle="--",
    linewidth=1
)

ax.set_xlim(0, a)
ax.set_ylim(0, b)
ax.set_aspect("equal")
ax.set_title("Rectangle, carré, points intérieurs et segments")

plt.show()

