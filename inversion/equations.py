import random

import jax.lax
import numpy as onp
import jax.numpy as np
import matplotlib.pyplot as plt

from jax import grad, jit, vmap, random
from jax._src.random import PRNGKey
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from model import CylinderDetector, StaticParticle


# Detectors are defined throughout as array([min_phi, min_z, d_phi, d_z])
# where they occupy [min_phi, min_phi + d_phi] x [min_z, min_z + d_z]


def atan2(y, x):
    angle = np.arctan2(y, x)
    return np.where(angle > 0.0, angle, 2 * np.pi + angle)


def F_lambdas_hat(R, varphi, X):
    x, y, _ = X
    c_1 = x * np.cos(varphi) + y * np.sin(varphi)
    c_2 = np.sqrt(c_1 ** 2 - (x ** 2 + y ** 2 - R ** 2))
    return -c_1 + c_2, -c_1 - c_2


def F_lambdas(R, varphi, theta, X):
    l_1_hat, l_2_hat = F_lambdas_hat(R, varphi, X)
    return 1.0 / np.sin(theta) * l_1_hat, 1.0 / np.sin(theta) * l_2_hat


def F_phi_1(R, varphi, X):
    x, y, _ = X
    l_1, _ = F_lambdas_hat(R, varphi, X)
    return atan2(y + l_1 * np.sin(varphi), x + l_1 * np.cos(varphi))


def F_phi_2(R, varphi, X):
    x, y, _ = X
    _, l_2 = F_lambdas_hat(R, varphi, X)
    return atan2(y + l_2 * np.sin(varphi), x + l_2 * np.cos(varphi))


def F_z_1(R, varphi, theta, X):
    x, y, z = X
    l_1, _ = F_lambdas_hat(R, varphi, X)
    return z + l_1 * np.cos(theta) / np.sin(theta)


def F_z_2(R, varphi, theta, X):
    _, _, z = X
    _, l_2 = F_lambdas_hat(R, varphi, X)
    return z + l_2 * np.cos(theta) / np.sin(theta)


def G_phi(R, phi_1, X):
    x, y, z = X
    R_x, R_y = x - R * np.cos(phi_1), y - R * np.sin(phi_1)
    omega = -2 * (R_x * R * np.cos(phi_1) + R_y * R * np.sin(phi_1)) / (R_x ** 2 + R_y ** 2)
    return atan2(R * np.sin(phi_1) + omega * R_y, R * np.cos(phi_1) + omega * R_x)


def G_z(R, phi_1, z_1, X):
    x, y, z = X
    R_x, R_y = x - R * np.cos(phi_1), y - R * np.sin(phi_1)
    omega = -2 * (R_x * R * np.cos(phi_1) + R_y * R * np.sin(phi_1)) / (R_x ** 2 + R_y ** 2)
    return z_1 + omega * (z - z_1)


def G_varphi_theta_1(R, phi_1, z_1, X):
    incident = np.array([R * np.cos(phi_1), R * np.sin(phi_1), z_1])
    diff = incident - X
    mu_1 = np.sqrt(np.dot(diff, diff))

    varphi = atan2(diff[1], diff[0])
    theta = np.arccos((z_1 - X[2]) / mu_1)

    return varphi, theta


def G_varphi_theta_2(R, phi_2, z_2, X):
    incident = np.array([R * np.cos(phi_2), R * np.sin(phi_2), z_2])
    diff = incident - X
    mu_2 = -np.sqrt(np.dot(diff, diff))

    varphi = atan2(-diff[1], -diff[0])
    theta = np.arccos((z_2 - X[2]) / mu_2)

    return varphi, theta


def jacobian_phi_1(R, varphi, X):
    return 1.0 / np.abs(grad(F_phi_1, 1)(R, varphi, X))


def jacobian_z_1(R, varphi, theta, X):
    return 1.0 / np.abs(grad(F_z_1, 2)(R, varphi, theta, X))


def jacobian_phi_2(R, varphi, X):
    return 1.0 / np.abs(grad(F_phi_2, 1)(R, varphi, X))


def jacobian_z_2(R, varphi, theta, X):
    return 1.0 / np.abs(grad(F_z_2, 2)(R, varphi, theta, X))


def greater_than(x, threshold, gamma):
    return 0.5 * np.tanh((x - threshold) * gamma) + 0.5


def smaller_than(x, threshold, gamma):
    return 0.5 * np.tanh(-(x - threshold) * gamma) + 0.5


def detector_proj(R, min_phi, max_phi, min_z, max_z, X):
    # Clockwise
    return G_phi(R, max_phi, X), G_phi(R, min_phi, X), \
           G_z(R, max_phi, max_z, X), G_z(R, max_phi, min_z, X), \
           G_z(R, min_phi, min_z, X), G_z(R, min_phi, max_z, X)


def characteristic_function(R, detector_j, phi_1, z_1, X, gamma):
    j_phi_min, j_z_min, d_phi, d_z = detector_j

    phi_2, z_2 = G_phi(R, phi_1, X), G_z(R, phi_1, z_1, X)

    cond_1 = greater_than(phi_2, j_phi_min, gamma)
    cond_2 = smaller_than(phi_2, j_phi_min + d_phi, gamma)
    cond_3 = greater_than(z_2, j_z_min, gamma)
    cond_4 = smaller_than(z_2, j_z_min + d_z, gamma)

    return cond_1 * cond_2 * cond_3 * cond_4


def evaluate_joint_integrand(R, sample_point, detector_j, X, gamma):
    phi_1, z_1 = sample_point

    # Characteristic
    char = characteristic_function(R=R, detector_j=detector_j, phi_1=phi_1, z_1=z_1, X=X, gamma=gamma)

    # Other parts of integrand
    varphi, theta = G_varphi_theta_1(R, phi_1, z_1, X)
    j_1 = jacobian_phi_1(R, varphi, X)
    j_2 = jacobian_z_1(R, varphi, theta, X)

    return (1 / (2 * np.pi)) * char * np.sin(theta) * j_1 * j_2


def evaluate_i_integrand(R, sample_point, X):
    phi_1, z_1 = sample_point
    varphi, theta = G_varphi_theta_1(R, phi_1, z_1, X)
    j_1 = jacobian_phi_1(R, varphi, X)
    j_2 = jacobian_z_1(R, varphi, theta, X)

    return (1 / (2 * np.pi)) * np.sin(theta) * j_1 * j_2


def evaluate_j_integrand(R, sample_point, X):
    phi_2, z_2 = sample_point
    varphi, theta = G_varphi_theta_2(R, phi_2, z_2, X)
    j_1 = jacobian_phi_2(R, varphi, X)
    j_2 = jacobian_z_2(R, varphi, theta, X)

    return (1 / (2 * np.pi)) * np.sin(theta) * j_1 * j_2


@jit
def compute_joint_probability(R, detector_i: tuple, detector_j: tuple, X: np.array, gamma, unifs: np.array):
    # P(i, j) Monte Carlo Integration
    i_phi_min, i_z_min, d_phi, d_z = detector_i
    _, _, z = X

    detector_corner = np.array([i_phi_min, i_z_min])
    detector_diff = np.array([d_phi, d_z])

    V = d_phi * d_z

    integrand = 0
    n_samples = 0.0
    for unif in unifs:
        # Sample point is corner + unif * diff
        sample = detector_corner + unif * detector_diff
        integrand += smaller_than(z, sample[1], gamma) * evaluate_joint_integrand(R, sample, detector_j, X, gamma)
        n_samples += 1.0
    integrand = 1 / n_samples * integrand

    return V * integrand


@jit
def compute_joint_probability_v2(R, detector_i: tuple, detector_j: tuple, X: np.array, gamma, unifs: np.array):
    # P(i, j) Monte Carlo Integration, this is faster to compile to XLA than compute_joint_probability
    i_phi_min, i_z_min, d_phi, d_z = detector_i
    _, _, z = X

    detector_corner = np.array([i_phi_min, i_z_min])
    detector_diff = np.array([d_phi, d_z])

    V = d_phi * d_z
    n_samples = unifs.shape[0]

    def loop_fun(i, v):
        sample = detector_corner + unifs[i, :] * detector_diff
        return v + smaller_than(z, sample[1], gamma) * evaluate_joint_integrand(R, sample, detector_j, X, gamma)

    integrand = jax.lax.fori_loop(lower=0, upper=n_samples,
                                  body_fun=loop_fun, init_val=0.0)
    integrand = 1.0 / n_samples * integrand

    return V * integrand


@jit
def compute_i_marginal_probability(R, detector_i: tuple, X: np.array):
    # P(i, .) midpoint rule integration
    i_phi_min, i_z_min, d_phi, d_z = detector_i

    centroid = np.array([i_phi_min + d_phi / 2.0, i_z_min + d_z / 2.0])

    return d_phi * d_z * evaluate_i_integrand(R, centroid, X)


@jit
def compute_j_marginal_probability(R, detector_j: tuple, X: np.array):
    # P(., j) midpoint rule integration
    j_phi_min, j_z_min, d_phi, d_z = detector_j

    centroid = np.array([j_phi_min + d_phi / 2.0, j_z_min + d_z / 2.0])

    return d_phi * d_z * evaluate_j_integrand(R, centroid, X)


@jit
def compute_marginal_probability(R, detector_i: np.array, detectors: np.array, X: np.array, gamma, unifs: np.array):
    joint_vmapped_1 = vmap(compute_joint_probability_v2, in_axes=(None, None, 0, None, None, None), out_axes=0)
    joint_vmapped_2 = vmap(compute_joint_probability_v2, in_axes=(None, 0, None, None, None, None), out_axes=0)

    return np.sum(joint_vmapped_1(R, detector_i, detectors, X, gamma, unifs)) \
           + np.sum(joint_vmapped_2(R, detectors, detector_i, X, gamma, unifs))


def plot_marginal(X=None, gamma=500):
    if X is None:
        X = np.array([0.0, 0.0, 0.0])
    else:
        X = np.array(X)

    d = CylinderDetector()

    n_phi, n_z = d.n_detector_cells()
    n = n_phi * n_z
    d_phi, d_z = d.del_detector_cells()

    detector_quads = [d.detector_cell_from_index(idx) for idx in range(n)]
    detectors = np.array([list(quad.min()) + [d_phi, d_z] for quad in detector_quads])

    key = random.PRNGKey(0)
    unifs = random.uniform(key=key, shape=(100, 2))

    marginal_mapped = jit(vmap(compute_marginal_probability, in_axes=(None, 0, None, None, None, None), out_axes=0))

    print('Evaluating marginal over entire detector')
    detectors_batched = np.array_split(detectors, 300)

    vals_batched = []
    for batch in tqdm(detectors_batched):
        vals_batched += [marginal_mapped(d.dim_radius_cm, batch, detectors, X, gamma, unifs)]

    vals = np.concatenate(vals_batched)

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
    #  Marginal Example Plots

    experiments = [
        {'position': [0.0, 0.0, 0.0], 'name': 'marginal_1'},
        {'position': [0.1, 0.1, 0.0], 'name': 'marginal_2'},
        {'position': [0.1, 0.1, 0.18], 'name': 'marginal_3'},
    ]

    for exp in experiments:
        plot_marginal(X=exp['position'])
        plt.savefig(f'figures/marginal/{exp["name"]}.eps', format='eps', bbox_inches='tight')
        plt.savefig(f'figures/marginal/{exp["name"]}.png', format='png', bbox_inches='tight')

    # plot_marginal()
    # plt.savefig('figures/comparison/marginal_1.eps', format='eps', bbox_inches='tight')
    # plt.savefig('figures/comparison/marginal_1.png', format='png', bbox_inches='tight')

    # plot_marginal(X=[0.1, 0.1, 0.0])
    # plt.savefig('figures/comparison/marginal_2.eps', format='eps', bbox_inches='tight')
    # plt.savefig('figures/comparison/marginal_2.png', format='png', bbox_inches='tight')

    # plot_marginal(X=[0.1, 0.1, 0.18])
    # plt.show()
    # plt.savefig('figures/comparison/marginal_3.eps', format='eps', bbox_inches='tight')
    # plt.savefig('figures/comparison/marginal_3.png', format='png', bbox_inches='tight')

    # #  Gradient eval on joint test
    # d = CylinderDetector()
    # n_phi, n_z = d.n_detector_cells()
    # n = n_phi * n_z
    # d_phi, d_z = 2.0 * onp.pi / n_phi, d.dim_height_cm / n_z
    #
    # detector_quads = [d.detector_cell_from_index(idx) for idx in range(n)]
    # detectors = np.array([list(quad.min()) + [d_phi, d_z] for quad in detector_quads])
    #
    # joint_grad = jit(grad(compute_joint_probability, 3))
    #
    # det_i, det_j = random.choice(detectors), random.choice(detectors)
    #
    # print('Compiled')
    #
    # random_x = np.array([np.array([onp.random.random()*0.1,
    #                        onp.random.random()*0.1,
    #                        onp.random.random()*0.1]) for _ in range(100000)])
    #
    # print('Compiling grad vec')
    # joint_grad_vec = jit(vmap(joint_grad, (None, None, None, 0), 0))
    # print('Finished')
    #
    # vals = onp.sum(joint_grad_vec(d.dim_radius_cm, det_i, det_j, random_x), axis=0)
    #
    # # print(joint_grad(d.dim_radius_cm, det_i, det_j, np.array([0.0, 0.0, 0.25])))
    # # print(joint_grad(d.dim_radius_cm, det_i, det_j, np.array([0.05, 0.09, 0.15])))
    # # print(joint_grad(d.dim_radius_cm, det_i, det_j, np.array([-0.03, 0.01, 0.35])))
    #
    # print(vals)
