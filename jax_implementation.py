from typing import List

import numpy as onp
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

import matplotlib.pyplot as plt
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

from model import CylinderDetector


def atan2(y, x):
    # return jnp.arctan2(y, x) + jnp.pi
    angle = jnp.arctan2(y, x)
    return jnp.where(angle > 0.0, angle, 2 * jnp.pi + angle)


def F_lambdas_hat(R, varphi, x, y):
    c_1 = x * jnp.cos(varphi) + y * jnp.sin(varphi)
    c_2 = jnp.sqrt(c_1 ** 2 - (x ** 2 + y ** 2 - R ** 2))
    return -c_1 + c_2, -c_1 - c_2


def F_phi_1(R, varphi, x, y):
    l_1, _ = F_lambdas_hat(R, varphi, x, y)
    return atan2(y + l_1 * jnp.cos(varphi), x + l_1 * jnp.sin(varphi))


def F_phi_2(R, varphi, x, y):
    _, l_2 = F_lambdas_hat(R, varphi, x, y)
    return atan2(y + l_2 * jnp.cos(varphi), x + l_2 * jnp.sin(varphi))


def F_z_1(R, varphi, theta, x, y, z):
    l_1, _ = F_lambdas_hat(R, varphi, x, y)
    return z + l_1 * jnp.cos(theta) / jnp.sin(theta)


def F_z_2(R, varphi, theta, x, y, z):
    _, l_2 = F_lambdas_hat(R, varphi, x, y)
    return z + l_2 * jnp.cos(theta) / jnp.sin(theta)


def G_phi(R, phi_1, x, y):
    c_x, c_y = x - R * jnp.cos(phi_1), y - R * jnp.sin(phi_1)
    omega = -2 * R * (jnp.cos(phi_1) * c_x + jnp.sin(phi_1) * c_y) / (c_x ** 2 + c_y ** 2)
    return atan2(R * jnp.sin(phi_1) + omega * c_y, R * jnp.cos(phi_1) + omega * c_x)


def G_z(R, phi_1, z_1, x, y, z):
    c_x, c_y = x - R * jnp.cos(phi_1), y - R * jnp.sin(phi_1)
    omega = -2 * R * (jnp.cos(phi_1) * c_x + jnp.sin(phi_1) * c_y) / (c_x ** 2 + c_y ** 2)
    return z_1 + omega * (z - z_1)


def G_varphi_theta(R, phi_1, z_1, x, y, z):
    p = jnp.array([x, y, z])
    incident = jnp.array([R*jnp.cos(phi_1), R*jnp.sin(phi_1), z_1])
    diff = p - incident
    l_1 = jnp.sqrt(jnp.dot(diff, diff))

    varphi = atan2(diff[1], diff[0])
    theta = jnp.arccos((z_1 - z)/l_1)

    return varphi, theta


def jacobian_phi_1(R, varphi, x, y):
    return jnp.abs(grad(F_phi_1, 1)(R, varphi, x, y))


def jacobian_z_1(R, varphi, theta, x, y, z):
    return jnp.abs(grad(F_z_1, 2)(R, varphi, theta, x, y, z))


def greater_than(x, threshold, gamma):
    return 0.5 * jnp.tanh((x - threshold) * gamma) + 0.5


def smaller_than(x, threshold, gamma):
    return 0.5 * jnp.tanh(-(x - threshold) * gamma) + 0.5


def detector_proj(R, min_phi, max_phi, min_z, max_z, x, y, z):
    # Clockwise
    # p1 = (G_phi(R, max_phi, x, y), G_z(R, max_phi, max_z, x, y, z))
    # p2 = (G_phi(R, max_phi, x, y), G_z(R, max_phi, min_z, x, y, z))
    # p3 = (G_phi(R, min_phi, x, y), G_z(R, min_phi, min_z, x, y, z))
    # p4 = (G_phi(R, min_phi, x, y), G_z(R, min_phi, max_z, x, y, z))

    return G_phi(R, max_phi, x, y), G_phi(R, min_phi, x, y), \
           G_z(R, max_phi, max_z, x, y, z), G_z(R, max_phi, min_z, x, y, z), \
           G_z(R, min_phi, min_z, x, y, z), G_z(R, min_phi, max_z, x, y, z)


def characteristic_function(R, min_phi, max_phi, min_z, max_z, phi_1, z_1, x, y, z):
    gamma = 5

    phi_2, z_2 = G_phi(R, phi_1, x, y), G_z(R, phi_1, z_1, x, y, z)

    cond_1 = greater_than(phi_2, min_phi, gamma)
    cond_2 = smaller_than(phi_2, max_phi, gamma)
    cond_3 = greater_than(z_2, min_z, gamma)
    cond_4 = smaller_than(z_2, max_z, gamma)

    return cond_1 * cond_2 * cond_3 * cond_4


def projected_inside_detector(R, min_phi, max_phi, min_z, max_z, phi_1, z_1, x, y, z):
    return characteristic_function(R, min_phi, max_phi, min_z, max_z, phi_1, z_1, x, y, z)


# def inside_projected_detector(R, min_phi, max_phi, min_z, max_z, phi_2, z_2, x, y, z):
#     gamma = 500
#
#     phi_bound_max, phi_bound_min, z1, z2, z3, z4 = detector_proj(R, min_phi, max_phi, min_z, max_z, x, y, z)
#
#     # Phi bounds
#     phi_bound_1 = greater_than(phi_2, phi_bound_min, gamma)
#     phi_bound_2 = smaller_than(phi_2, phi_bound_max, gamma)
#
#     bound = jnp.where(phi_bound_min > phi_bound_max,
#                       phi_bound_1 + phi_bound_2,
#                       phi_bound_1 * phi_bound_2)
#     t = jnp.where(phi_bound_min > phi_bound_max,
#                   (phi_2 - phi_bound_min) / (phi_bound_max + 2 * jnp.pi - phi_bound_min),
#                   (phi_2 - phi_bound_min) / (phi_bound_max - phi_bound_min))
#
#     # Z bounds
#     bound = bound * greater_than(z_2, z1 + t * (z4 - z1), gamma)
#     bound = bound * smaller_than(z_2, z2 + t * (z2 - z3), gamma)
#
#     return bound  # in [0,1]


def evaluate_integrand(R, detector_i: tuple, detector_j: tuple, x, y, z):
    i_phi_min, i_phi_max, i_z_min, i_z_max = detector_i
    j_phi_min, j_phi_max, j_z_min, j_z_max = detector_j

    centroid_phi = (i_phi_max - i_phi_min) / 2.0
    centroid_z = (i_z_max - i_z_min) / 2.0

    # Characteristic
    char = characteristic_function(R=R, min_phi=j_phi_min, max_phi=j_phi_max,
                                   min_z=j_z_min, max_z=j_z_max, phi_1=centroid_phi, z_1=centroid_z,
                                   x=x, y=y, z=z)

    # Other parts of integrand
    centroid_varphi, centroid_theta = G_varphi_theta(R, centroid_phi, centroid_z, x, y, z)
    j_1 = jacobian_phi_1(R, centroid_varphi, x, y)
    j_2 = jacobian_z_1(R, centroid_varphi, centroid_theta, x, y, z)

    return (1 / (4 * jnp.pi)) * char * jnp.sin(centroid_theta) * j_1 * j_2


def compute_joint_probability(R, detector_i: tuple, detector_j: tuple, x, y, z):
    i_phi_min, i_phi_max, i_z_min, i_z_max = detector_i

    # Integral estimate = Volume * integrand(centroid)
    V = (i_phi_max - i_phi_min) * (i_z_max - i_z_min)

    return V * evaluate_integrand(R=R, detector_i=detector_i, detector_j=detector_j,
                                  x=x, y=y, z=z)


def compute_marginal_probability(R, detector_i: tuple, detectors: List[tuple], x, y, z):
    term_1 = 0.0
    for d in detectors:
        term_1 += compute_joint_probability(R, detector_i, d, x, y, z)
    term_1 = 2 * term_1

    term_2 = compute_joint_probability(R, detector_i, detector_i, x, y, z)
    return term_1 - term_2


def plot_proj_area(min_phi=jnp.pi / 2 - 0.05, max_phi=jnp.pi / 2 + 0.05, min_z=0.20, max_z=0.30, x=0.0, y=0.0, z=0.25):
    jit_test = jit(projected_inside_detector)

    fig, ax = plt.subplots()
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts}'
    })
    fig.set_size_inches(8, 20)

    shp = (314 * 2, 50 * 2)
    phi_vals = onp.linspace(0, 2 * onp.pi, num=shp[0])
    z_vals = onp.linspace(0, 0.5, num=shp[1])

    img = onp.zeros((len(phi_vals), len(z_vals)))

    for i, phi_samp in enumerate(phi_vals):
        for j, z_samp in enumerate(z_vals):
            img[i, j] = jit_test(R=0.5,
                                 min_phi=min_phi,
                                 max_phi=max_phi,
                                 min_z=min_z,
                                 max_z=max_z,
                                 phi_1=phi_samp, z_1=z_samp,
                                 x=x, y=y, z=z)

    ax.imshow(img.transpose(), origin='lower')

    # Plot original detector
    d_dphi = (max_phi - min_phi) / (2 * onp.pi) * shp[0]
    d_dz = (max_z - min_z) / 0.5 * shp[1]
    rect = patches.Rectangle((min_phi / (2 * onp.pi) * shp[0], min_z / 0.5 * shp[1]), d_dphi, d_dz,
                             linewidth=1, edgecolor='r', facecolor='none', label=r'$\mathcal{D}_j$ boundary')
    ax.add_patch(rect)
    ax.legend()

    ax.set_xlabel(r'Horizontal $\phi_1 \in [0, 2\pi]$')
    ax.set_ylabel(r'Vertical $z_1 \in [0, 0.5]$')
    ax.set_title(r'$\mathbb{I}_j(\phi_1, z_1)$')

    return fig, ax


def plot_marginal(d: CylinderDetector, x, y, z):
    R = d.dim_radius_cm
    jitted_eval = jit(evaluate_integrand)
    # evaluate_integrand(R, i_phi_min, i_phi_max, i_z_min, i_z_max,
    #                    j_phi_min, j_phi_max, j_z_min, j_z_max,
    #                    x, y, z)

    fig, ax = plt.subplots()
    n_x, n_y = d.n_detector_cells()
    img = onp.zeros((n_x, n_y))

    detector_cells = []

    for i in range(n_x):
        for j in range(n_y):
            idx = i + j * n_x
            centre_phi, centre_z = d.detector_cell_from_index(idx).vertices().mean(axis=0)
            v = jitted_eval(R, centre_phi, centre_z, x, y, z)
            img[i, j] = v

    plt_im = ax.imshow(img.transpose(), origin='lower')

    # Plot original detector
    ax.set_xlabel(r'Horizontal $\phi_2 \in [0, 2\pi]$')
    ax.set_ylabel(r'Vertical $z_2 \in [0, 0.5]$')
    ax.set_title(r'Marginal Distribution')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(plt_im, cax=cax, orientation='vertical')

    return fig, ax


if __name__ == '__main__':
    # f = jit(jacobian_z_1)
    # print('Start')
    # for i in range(1000):
    #     f(0.5, jnp.pi, 0.1, 0.1 + 0.1 * i/1000, 0.1, 0.25)
    #     # jacobian_z_1(0.5, jnp.pi, 0.1, 0.1 + 0.1 * i/1000, 0.1, 0.25)

    # for v in detector_proj(R=0.5, min_phi=jnp.pi-0.05, max_phi=jnp.pi+0.05, min_z=0.20, max_z=0.30, x=0.0, y=0.0, z=0.25):
    #     print(onp.array(v))

    # plot_proj_area()

    # plot_proj_area(jnp.pi/4, jnp.pi/4 + jnp.pi/10, min_z=0.1, max_z=0.2, x=0.1, y=0.05)
    # plt.show()

    d = CylinderDetector()
    n_x, n_y = d.n_detector_cells()
    n = n_x * n_y
    detectors = [d.detector_cell_from_index(idx) for idx in range(n)]
    detectors_boundaries = [cell.x_range() + cell.y_range() for cell in detectors]

    print(detectors_boundaries)

