import random
import numpy as onp
import jax.numpy as jnp
import matplotlib.pyplot as plt

from typing import List
from jax import grad, jit, vmap
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from model import CylinderDetector


# Detectors are defined throughout as array([min_phi, min_z, d_phi, d_z])
# where they occupy [min_phi, min_phi + d_phi] x [min_z, min_z + d_z]


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
    incident = jnp.array([R * jnp.cos(phi_1), R * jnp.sin(phi_1), z_1])
    diff = incident - p
    l_1 = jnp.sqrt(jnp.dot(diff, diff))

    varphi = atan2(diff[1], diff[0])
    theta = jnp.arccos((z_1 - z) / l_1)

    return varphi, theta


def jacobian_phi_1(R, varphi, x, y):
    return 1.0 / jnp.abs(grad(F_phi_1, 1)(R, varphi, x, y))


def jacobian_z_1(R, varphi, theta, x, y, z):
    return 1.0 / jnp.abs(grad(F_z_1, 2)(R, varphi, theta, x, y, z))


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


def characteristic_function(R, detector_j, phi_1, z_1, x, y, z):
    gamma = 500

    j_phi_min, j_z_min, d_phi, d_z = detector_j

    phi_2, z_2 = G_phi(R, phi_1, x, y), G_z(R, phi_1, z_1, x, y, z)

    cond_1 = greater_than(phi_2, j_phi_min, gamma)
    cond_2 = smaller_than(phi_2, j_phi_min + d_phi, gamma)
    cond_3 = greater_than(z_2, j_z_min, gamma)
    cond_4 = smaller_than(z_2, j_z_min + d_z, gamma)

    return cond_1 * cond_2 * cond_3 * cond_4


def projected_inside_detector(R, detector_j, phi_1, z_1, x, y, z):
    return characteristic_function(R, detector_j, phi_1, z_1, x, y, z)


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


def evaluate_integrand(R, detector_i, detector_j, x, y, z):
    i_phi_min, i_z_min, d_phi, d_z = detector_i

    centroid_phi = i_phi_min + d_phi / 2.0
    centroid_z = i_z_min + d_z / 2.0

    # Characteristic
    char = characteristic_function(R=R, detector_j=detector_j, phi_1=centroid_phi, z_1=centroid_z,
                                   x=x, y=y, z=z)

    # Other parts of integrand
    centroid_varphi, centroid_theta = G_varphi_theta(R, centroid_phi, centroid_z, x, y, z)
    j_1 = jacobian_phi_1(R, centroid_varphi, x, y)
    j_2 = jacobian_z_1(R, centroid_varphi, centroid_theta, x, y, z)

    return (1 / (4 * jnp.pi)) * char * jnp.sin(centroid_theta) * j_1 * j_2


def compute_joint_probability(R, detector_i: tuple, detector_j: tuple, x, y, z):
    i_phi_min, i_z_min, d_phi, d_z = detector_i

    # Integral estimate = Volume * integrand(centroid)
    V = d_phi * d_z

    return V * evaluate_integrand(R=R, detector_i=detector_i, detector_j=detector_j,
                                  x=x, y=y, z=z)


def compute_marginal_probability(R, detector_i: jnp.array, detectors: jnp.array, x, y, z):
    joint = jit(compute_joint_probability)
    joint_vmapped = jit(vmap(compute_joint_probability,
                             in_axes=(None, None, 0, None, None, None), out_axes=0))

    # term_1 = 2 * jnp.sum(joint_vmapped(R, detector_i, detectors, x, y, z))
    # term_2 = joint(R, detector_i, detector_i, x, y, z)
    # return term_1 - term_2

    return jnp.sum(joint_vmapped(R, detector_i, detectors, x, y, z))


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


def plot_marginal(x=0.0, y=0.0, z=0.25):
    d = CylinderDetector()

    n_phi, n_z = d.n_detector_cells()
    n = n_phi * n_z
    d_phi, d_z = 2.0 * onp.pi / n_phi, d.dim_height_cm / n_z

    detector_quads = [d.detector_cell_from_index(idx) for idx in range(n)]
    detectors = jnp.array([list(quad.min()) + [d_phi, d_z] for quad in detector_quads])

    marginal_mapped = jit(vmap(compute_marginal_probability, in_axes=(None, 0, None, None, None, None), out_axes=0))

    vals = marginal_mapped(d.dim_radius_cm, detectors, detectors, x, y, z)
    img = onp.array(vals).reshape((n_z, n_phi))

    fig, ax = plt.subplots()
    plt_im = ax.imshow(img, origin='lower')
    ax.set_xlabel(r'Horizontal $\phi \in [0, 2\pi]$')
    ax.set_ylabel(r'Vertical $z \in [0, 0.5]$')
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

    # d = CylinderDetector()
    #
    # n_phi, n_z = d.n_detector_cells()
    # n = n_phi * n_z
    # d_phi, d_z = 2.0 * onp.pi / n_phi, d.dim_height_cm / n_z
    #
    # detector_quads = [d.detector_cell_from_index(idx) for idx in range(n)]
    # detectors = jnp.array([list(quad.min()) + [d_phi, d_z] for quad in detector_quads])
    #
    # print(compute_marginal_probability(R=d.dim_radius_cm,
    #                                    detector_i=detectors[random.randint(0, n - 1), :],
    #                                    detectors=detectors,
    #                                    x=0.0,
    #                                    y=0.0,
    #                                    z=0.25))

    plot_marginal()
    plt.show()

    plot_marginal(0.1, 0.1, 0.25)
    plt.show()