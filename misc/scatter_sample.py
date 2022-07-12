# Sanity check that our scattering rotations are correct

import matplotlib.pyplot as plt
import numpy as np

from plot import arrow_3d, points_3d

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')

e_theta = np.array([0, 1, 0])
e_phi = np.array([1, 0, 0])
n = np.array([0, 0, 1])
change_of_basis = np.array([e_phi, e_theta, n]).transpose()

arrow_3d(ax=ax, origin=[0, 0, 0], dir=1.5*n, color='red')

points = []
for _ in range(1000):
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

    points += [change_of_basis.dot(np.matmul(rot_phi, rot_theta).dot([0, 0, 1]))]

points_3d(ax=ax, points=np.array(points))
ax.axes.set_xlabel('x')
ax.axes.set_ylabel('y')
ax.axes.set_xlim3d(left=-2.0, right=2.0)
ax.axes.set_ylim3d(bottom=-2.0, top=2.0)
ax.axes.set_zlim3d(bottom=0.0, top=2.0)
# plt.show()

plt.savefig('scatter_angles.eps', format='eps')
