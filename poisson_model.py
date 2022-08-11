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

    theta_1 = (-R_varphi + np.sqrt(R_varphi**2 - (x**2 + y**2 - R**2)))/(H / 2.0 - z)
    theta_2 = (-R_varphi - np.sqrt(R_varphi**2 - (x**2 + y**2 - R**2)))/(-H / 2.0 - z)

    theta_min = np.max(np.array([theta_1, theta_2]))

    return np.cos(np.arctan(theta_min))

@jit
def G_solid_angle_approx(R, H, X):
    # Trapezoidal Rule
    d_varphi = 2*np.pi/10
    varphi = np.arange(start=d_varphi, stop=2*np.pi, step=d_varphi)
    integrand_vmap = vmap(_solid_angle_integrand, (None, None, None, 0), 0)
    f_x_0 = _solid_angle_integrand(R, H, X, 0.0)
    f_x_N = _solid_angle_integrand(R, H, X, 2*np.pi)
    return (1 / (2*np.pi)) * (d_varphi / 2.0) * (f_x_0 + f_x_N + 2.0*np.sum(integrand_vmap(R, H, X, varphi)))


@jit
def single_dimensional_likelihood(R, H, rate, T, detections_i, detections_j, X, gamma, unifs):
    joint_vmapped = vmap(compute_joint_probability, (None, 0, 0, None, None, None), 0)
    joint_evaluations = joint_vmapped(R, detections_i, detections_j, X, gamma, unifs)

    # term_1 = np.sum(np.log(joint_evaluations))
    term_1 = np.sum(np.log(joint_evaluations))
    term_2 = - rate * T * G_solid_angle_approx(R, H, X)
    # return term_1 + term_2
    return term_1


def G_solid_angle_plot(beta_ratio):
    det = CylinderDetector()
    det.dim_radius_cm = beta_ratio*det.dim_height_cm

    fig, ax = plt.subplots()
    x_samples = onp.linspace(-det.dim_radius_cm + 0.0001, det.dim_radius_cm - 0.0001, int(200*beta_ratio))
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
    T, activity = 1.0, 10**4
    X = np.array(p.get_position_cartesian())

    # Simulate Dataset
    lors, scatters = p.simulate_emissions(detector=det, n_lor=int(T*activity))
    d_phi, d_z = det.del_detector_cells()
    detections_i = np.array([
        list(quad.min()) + [d_phi, d_z] for quad in [det.detector_cell_from_index(lor[0]) for lor in lors]
    ])
    detections_j = np.array([
        list(quad.min()) + [d_phi, d_z] for quad in [det.detector_cell_from_index(lor[1]) for lor in lors]
    ])
    print(f'Simulations finished, LoRs={len(lors)}, Scatters={scatters}')

    # Evaluate likelihood
    def eval_likelihood(X_test):
        key = random.PRNGKey(0)
        unifs = random.uniform(key=key, shape=(100, 2))
        likelihood = single_dimensional_likelihood(R=det.dim_radius_cm, H=det.dim_height_cm, rate=activity, T=T,
                                                   detections_i=detections_i, detections_j=detections_j, X=X_test,
                                                   gamma=5.0, unifs=unifs)
        return likelihood

    print(eval_likelihood(X_test=X))
    print(eval_likelihood(X_test=np.array([0.0, 0.0, 0.1])))
    print(eval_likelihood(X_test=np.array([0.0, 0.0, -0.1])))
    print(eval_likelihood(X_test=np.array([0.24, 0.0, 0.0])))

