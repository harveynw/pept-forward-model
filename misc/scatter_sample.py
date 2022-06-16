# Sanity check that our scattering rotations are correct

import matplotlib.pyplot as plt
import numpy as np

from plot import arrow_3d

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')

arrow_3d(ax=ax, origin=[0, 0, 0], dir=[0, 0, 1], color='red')

for _ in range(10):
    phi = np.random.uniform(low=0, high=2 * np.pi)
    theta = np.random.vonmises(mu=0, kappa=10)

    rot_theta = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

    rot_phi = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi), np.cos(phi), 0],
        [0, 0, 1]
    ])

    arrow_3d(ax=ax, origin=[0, 0, 0], dir=np.matmul(rot_phi, rot_theta).dot([0, 0, 1]))

ax.axes.set_xlim3d(left=-2.0, right=2.0)
ax.axes.set_ylim3d(bottom=-2.0, top=2.0)
ax.axes.set_zlim3d(bottom=-2.0, top=2.0)
plt.show()