import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt


from mpl_toolkits.axes_grid1 import make_axes_locatable
from inversion.inference import eval_single_dimensional_likelihood
from inversion.integrals import G_integral
from model import CylinderDetector, StaticParticle


def G_solid_angle_plot(beta_ratio):
    det = CylinderDetector()
    det.dim_radius_cm = beta_ratio * det.dim_height_cm

    fig, ax = plt.subplots()
    x_samples = onp.linspace(-det.dim_radius_cm + 0.0001, det.dim_radius_cm - 0.0001, int(200 * beta_ratio))
    z_samples = onp.linspace(-det.dim_height_cm / 2 + 0.0001, det.dim_height_cm / 2.0 - 0.0001, 100)

    im = onp.zeros((len(x_samples), len(z_samples)))
    for i in range(len(x_samples)):
        for j in range(len(z_samples)):
            im[i, j] = G_integral(det.dim_radius_cm,
                                  det.dim_height_cm,
                                  X=np.array([x_samples[i],
                                              0.0, z_samples[j]]))

    plt_im = ax.imshow(im.transpose(), origin='lower', cmap='Blues')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(plt_im, cax=cax, orientation='vertical')

    ax.set_title(rf'$\beta={beta_ratio}$')

    return fig, ax


def single_dimensional_likelihood_plot(d: CylinderDetector, activity: float, T: float, lors: list, gamma: float):
    fig, (ax1, ax2) = plt.subplots(ncols=2)

    R, H = d.dim_radius_cm, d.dim_height_cm
    n_samps = 100
    x, y, z = onp.linspace(-R, R, n_samps), onp.linspace(-R, R, n_samps), onp.linspace(-H / 2.0, H / 2.0, n_samps)
    d_x, d_y, d_z = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    img_1, img_2 = onp.full((n_samps, n_samps), -onp.inf), onp.full((n_samps, n_samps), -onp.inf)

    # Plotting over (x, y, z=0.0)
    points = [[x_s + d_x / 2, y_s + d_y / 2, 0.0] for x_s in x for y_s in y if onp.sqrt(x_s ** 2 + y_s ** 2) < R]
    indices = [[i, j] for i, x_s in enumerate(x) for j, y_s in enumerate(y) if onp.sqrt(x_s ** 2 + y_s ** 2) < R]
    vals = eval_single_dimensional_likelihood(d, activity, T, lors, gamma, np.array(points), scattering=True)
    for idx, (i, j) in enumerate(indices):
        img_1[i, j] = vals[idx]
        # img_1[i, j] = points[idx][0]

    print('Sampled x-y plot')

    # Plotting over (x, y=0.0, z)
    points = [[x_s + d_x / 2, 0.0, z_s + d_z / 2] for x_s in x for z_s in z]
    indices = [[i, j] for i, x_s in enumerate(x) for j, z_s in enumerate(z)]
    vals = eval_single_dimensional_likelihood(d, activity, T, lors, gamma, np.array(points), scattering=True)
    for idx, (i, j) in enumerate(indices):
        img_2[i, j] = vals[idx]

    print('Sampled x-z plot')

    ax1.imshow(img_1.transpose(), origin='lower')
    ax2.imshow(img_2.transpose(), origin='lower')
    ax1.set_xlabel('x'), ax1.set_ylabel('y')
    ax2.set_xlabel('x'), ax2.set_ylabel('z')

    return fig, (ax1, ax2)


def scattering_experiment_plot(d: CylinderDetector, p: StaticParticle, activity, T, gamma):
    # TODO, This needs to be reimplemented, passing down a known scattering rate

    # Generate dataset
    lors, scatters = p.simulate_emissions(detector=d, n_lor=int(T * activity))

    # Eval likelihood over slices of detector
    fig, ax = single_dimensional_likelihood_plot(d=d, activity=activity, T=T, gamma=gamma, lors=lors)
    fig.suptitle(rf'Likelihood, particle={p.to_str_cartesian()}, scattering rate $\mu={p.scatter_rate}$')

    return fig, ax