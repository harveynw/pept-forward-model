import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt

from jax import jit, vmap, random
from mpl_toolkits.axes_grid1 import make_axes_locatable

from jax_implementation import compute_joint_probability
from model import CylinderDetector, StaticParticle


@jit
def _solid_angle_integrand(R, H, X, varphi):
    x, y, z = X

    R_varphi = x * np.cos(varphi) + y * np.sin(varphi)

    theta_1 = (-R_varphi + np.sqrt(R_varphi ** 2 - (x ** 2 + y ** 2 - R ** 2))) / (H / 2.0 - z)
    theta_2 = (-R_varphi - np.sqrt(R_varphi ** 2 - (x ** 2 + y ** 2 - R ** 2))) / (-H / 2.0 - z)

    theta_min = np.max(np.array([theta_1, theta_2]))

    return np.cos(np.arctan(theta_min))


@jit
def G_solid_angle_approx(R, H, X):
    # Trapezoidal Rule
    d_varphi = 2 * np.pi / 10
    varphi = np.arange(start=d_varphi, stop=2 * np.pi, step=d_varphi)
    integrand_vmap = vmap(_solid_angle_integrand, (None, None, None, 0), 0)
    f_x_0 = _solid_angle_integrand(R, H, X, 0.0)
    f_x_N = _solid_angle_integrand(R, H, X, 2 * np.pi)
    return (1 / (2 * np.pi)) * (d_varphi / 2.0) * (f_x_0 + f_x_N + 2.0 * np.sum(integrand_vmap(R, H, X, varphi)))


@jit
def single_dimensional_likelihood(R, H, rate, T, detections_i, detections_j, X, gamma, unifs):
    joint_vmapped = vmap(compute_joint_probability, (None, 0, 0, None, None, None), 0)
    joint_evaluations = joint_vmapped(R, detections_i, detections_j, X, gamma, unifs)

    # term_1 = np.sum(np.log(joint_evaluations))
    term_1 = np.sum(np.log(rate * T * (joint_evaluations + 0.05) ))
    term_2 = - rate * T * G_solid_angle_approx(R, H, X)
    return term_1 + term_2


def lors_to_jax(d: CylinderDetector, lors: list):
    d_phi, d_z = d.del_detector_cells()

    detections_i = np.array([
        list(quad.min()) + [d_phi, d_z] for quad in [d.detector_cell_from_index(lor[0]) for lor in lors]
    ])
    detections_j = np.array([
        list(quad.min()) + [d_phi, d_z] for quad in [d.detector_cell_from_index(lor[1]) for lor in lors]
    ])

    return detections_i, detections_j


def eval_single_dimensional_likelihood(d: CylinderDetector, activity: float, T: float, lors: list, gamma: float,
                                       X: np.array):
    # Uniform samples on [0,1] x [0,1] for estimating integral
    key = random.PRNGKey(0)
    unifs = random.uniform(key=key, shape=(5, 2))

    # Collecting LoRs into JAX Arrays
    detections_i, detections_j = lors_to_jax(d, lors)

    R, H = d.dim_radius_cm, d.dim_height_cm

    if np.ndim(X) == 1:
        return single_dimensional_likelihood(R=R, H=H, rate=activity, T=T,
                                             detections_i=detections_i, detections_j=detections_j, X=X,
                                             gamma=gamma, unifs=unifs)
    else:
        # Vmap case
        likelihood_mapped = jit(vmap(single_dimensional_likelihood,
                                     (None, None, None, None, None, None, 0, None, None), 0))
        return likelihood_mapped(R, H, activity, T, detections_i, detections_j, X, gamma, unifs)


def single_dimensional_likelihood_plot(d: CylinderDetector, activity: float, T: float, lors: list, gamma: float):
    fig, (ax1, ax2) = plt.subplots(ncols=2)

    R, H = d.dim_radius_cm, d.dim_height_cm
    n_samps = 100
    x, y, z = onp.linspace(-R, R, n_samps), onp.linspace(-R, R, n_samps), onp.linspace(-H / 2.0, H / 2.0, n_samps)
    d_x, d_y, d_z = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    img_1, img_2 = onp.full((n_samps, n_samps), -onp.inf), onp.full((n_samps, n_samps), -onp.inf)

    # Plotting over (x, y, z=0.0)
    points = [[x_s+d_x/2, y_s+d_y/2, 0.0] for x_s in x for y_s in y if onp.sqrt(x_s ** 2 + y_s ** 2) < R]
    indices = [[i, j] for i, x_s in enumerate(x) for j, y_s in enumerate(y) if onp.sqrt(x_s ** 2 + y_s ** 2) < R]
    vals = eval_single_dimensional_likelihood(d, activity, T, lors, gamma, np.array(points))
    for idx, (i, j) in enumerate(indices):
        img_1[i, j] = vals[idx]

    print('Sampled x-y plot')

    # Plotting over (x, y=0.0, z)
    points = [[x_s+d_x/2, 0.0, z_s+d_z/2] for x_s in x for z_s in z]
    indices = [[i, j] for i, x_s in enumerate(x) for j, z_s in enumerate(z)]
    vals = eval_single_dimensional_likelihood(d, activity, T, lors, gamma, np.array(points))
    for idx, (i, j) in enumerate(indices):
        img_2[i, j] = vals[idx]

    print('Sampled x-z plot')

    ax1.imshow(img_1.transpose(), origin='lower')
    ax2.imshow(img_2.transpose(), origin='lower')
    ax1.set_xlabel('x'), ax1.set_ylabel('y')
    ax2.set_xlabel('x'), ax2.set_ylabel('z')

    return fig, (ax1, ax2)


def G_solid_angle_plot(beta_ratio):
    det = CylinderDetector()
    det.dim_radius_cm = beta_ratio * det.dim_height_cm

    fig, ax = plt.subplots()
    x_samples = onp.linspace(-det.dim_radius_cm + 0.0001, det.dim_radius_cm - 0.0001, int(200 * beta_ratio))
    z_samples = onp.linspace(-det.dim_height_cm / 2 + 0.0001, det.dim_height_cm / 2.0 - 0.0001, 100)

    im = onp.zeros((len(x_samples), len(z_samples)))
    for i in range(len(x_samples)):
        for j in range(len(z_samples)):
            im[i, j] = G_solid_angle_approx(det.dim_radius_cm, det.dim_height_cm,
                                            X=np.array([x_samples[i], 0.0, z_samples[j]]))

    plt_im = ax.imshow(im.transpose(), origin='lower', cmap='Blues')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(plt_im, cax=cax, orientation='vertical')

    ax.set_title(rf'$\beta={beta_ratio}$')

    return fig, ax


if __name__ == '__main__':
    # d = CylinderDetector()
    # print(G_solid_angle_approx(R=d.dim_radius_cm, H=d.dim_height_cm, X=np.array([0.1, 0.1, 0.0])))
    # print(G_solid_angle_approx(R=d.dim_radius_cm, H=d.dim_height_cm, X=np.array([-0.1, -0.1, 0.0])))
    # print(G_solid_angle_approx(R=d.dim_radius_cm, H=d.dim_height_cm, X=np.array([0.0, 0.0, 0.1])))
    # print(G_solid_angle_approx(R=d.dim_radius_cm, H=d.dim_height_cm, X=np.array([0.0, 0.0, 0.0])))
    #
    # fig, ax = G_solid_angle_plot(beta_ratio=1.0)
    # plt.show()
    #
    # fig, ax = G_solid_angle_plot(beta_ratio=0.5)
    # plt.show()
    #
    # fig, ax = G_solid_angle_plot(beta_ratio=0.25)
    # plt.show()

    # Setup particle and set to no scattering
    det = CylinderDetector()
    p = StaticParticle()
    p.scatter_rate = 0.000001
    T, activity = 1.0, 10 ** 4
    X = np.array(p.get_position_cartesian())

    # Simulate Dataset
    lors, scatters = p.simulate_emissions(detector=det, n_lor=int(T * activity))
    print(f'Simulations finished, LoRs={len(lors)}, Scatters={scatters}')

    args = {'d': det, 'activity': activity, 'T': T, 'gamma': 2.5, 'lors': lors}

    print(eval_single_dimensional_likelihood(**args, X=X))
    print(eval_single_dimensional_likelihood(**args, X=np.array([0.01, 0.0, 0.0])))
    print(eval_single_dimensional_likelihood(**args, X=np.array([-0.01, 0.0, 0.0])))
    print(eval_single_dimensional_likelihood(**args, X=np.array([0.0, 0.01, 0.0])))
    print(eval_single_dimensional_likelihood(**args, X=np.array([0.0, -0.01, 0.0])))


    print(eval_single_dimensional_likelihood(**args, X=np.array([0.0, 0.0, 0.1])))
    print(eval_single_dimensional_likelihood(**args, X=np.array([0.0, 0.0, -0.1])))
    print(eval_single_dimensional_likelihood(**args, X=np.array([0.24, 0.0, 0.0])))

    single_dimensional_likelihood_plot(**args)
    plt.show()
