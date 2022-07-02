# Sanity check that we are taking a uniform sample on sphere

import matplotlib.pyplot as plt
import numpy as np

from plot import points_3d

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')

vecs = []
for _ in range(5000):
    phi = np.random.uniform(0, 2 * np.pi)
    theta = np.arccos(1-2*np.random.uniform(0, 1))
    vecs.append([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])

points_3d(ax=ax, points=np.array(vecs))

ax.axes.set_xlim3d(left=-2.0, right=2.0)
ax.axes.set_ylim3d(bottom=-2.0, top=2.0)
ax.axes.set_zlim3d(bottom=-2.0, top=2.0)
plt.show()


