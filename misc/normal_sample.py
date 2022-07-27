# Sanity check that we are taking a uniform sample for the normal vector defining the LOR

import matplotlib.pyplot as plt
import numpy as np

from geometry import atan2
from plot import points_3d, arrow_3d

vecs, opposite_vecs = [], []
for _ in range(5000):
    # phi = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, 2*np.pi)
    theta = np.arccos(1 - 2 * np.random.uniform(0, 1))

    # # These vectors are incorrect for some angles
    # e_phi = np.array([
    #     np.cos(phi),
    #     np.sin(phi),
    #     0.0
    # ])
    # e_theta = np.array([
    #     np.sin(theta) * np.cos(phi + np.pi/2),
    #     np.sin(theta) * np.sin(phi + np.pi/2),
    #     np.cos(theta)
    # ])
    #
    # n = np.cross(e_phi, e_theta)
    # n = n / np.linalg.norm(n)

    # n = np.array([
    #     np.cos(phi)*np.sin(theta),
    #     np.sin(phi)*np.sin(theta),
    #     np.cos(theta)
    # ])
    # e_phi = np.array([
    #     np.sin(phi+np.pi/2),
    #     np.cos(phi+np.pi/2),
    #     0
    # ])
    # e_theta = np.array([
    #     np.sin(theta+np.pi/2)*np.cos(phi + (np.pi if theta > np.pi/2 else 0)),
    #     np.sin(theta+np.pi/2)*np.sin(phi + (np.pi if theta > np.pi/2 else 0)),
    #     (-1 if theta > np.pi/2 else 1) * np.cos(np.pi/2 - theta)
    # ])

    theta_hat = np.arcsin(2 * np.random.uniform(0, 1) - 1)
    theta = theta_hat + np.pi/2
    # These vectors are incorrect for some angles
    e_phi = np.array([
        np.cos(phi),
        np.sin(phi),
        0.0
    ])
    e_theta = np.array([
        np.sin(theta_hat) * np.cos(phi + np.pi/2),
        np.sin(theta_hat) * np.sin(phi + np.pi/2),
        np.cos(theta_hat)
    ])

    n = np.cross(e_phi, e_theta)
    n = n / np.linalg.norm(n)


    vecs.append(n)
    opposite_vecs.append(-n)

    # Debug plot
    # fig = plt.figure(figsize=(10, 5))
    # ax = fig.add_subplot(111, projection='3d')
    # arrow_3d(ax, [0, 0, 0], e_phi, color='green', label=r'$\bf{e}_{\varphi}$')
    # arrow_3d(ax, [0, 0, 0], e_theta, color='red', label=r'$\bf{e}_{\theta}$')
    # arrow_3d(ax, [0, 0, 0], n, color='blue', label=r'$\bf{n}$')
    # ax.axes.set_xlim3d(left=-1.0, right=1.0), ax.axes.set_ylim3d(bottom=-1.0, top=1.0), \
    # ax.axes.set_zlim3d(bottom=-1.0, top=1.0)
    # plt.title(f"phi={phi}, theta={theta}")
    # plt.legend()
    # plt.show()

    # vecs.append(e_phi)
    # opposite_vecs.append(e_theta)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
points_3d(ax=ax, points=np.array(vecs))

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
points_3d(ax=ax, points=np.array(opposite_vecs), color='r')

plt.show()

samples = vecs + opposite_vecs
sample_theta = [np.arccos(p[2]) for p in samples]
sample_phi = [atan2(p[0], p[1]) for p in samples]

fig = plt.figure(figsize=(10, 5))
plt.hist2d(x=sample_phi, y=sample_theta, bins=100)
plt.colorbar()
plt.show()


