import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt

from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from inversion.equations import characteristic_function
from inversion.inference import create_likelihood
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


def single_dimensional_likelihood_plot(d: CylinderDetector, activity: float, T: float, lors: list, gamma: float, mu: float):
    fig, (ax1, ax2) = plt.subplots(ncols=2)

    likelihood, _ = create_likelihood(d, activity, T, lors, gamma, mu, mc_samples=10, mapped=True)

    R, H = d.dim_radius_cm, d.dim_height_cm
    n_samps = 100
    x, y, z = onp.linspace(-R, R, n_samps), onp.linspace(-R, R, n_samps), onp.linspace(-H / 2.0, H / 2.0, n_samps)
    d_x, d_y, d_z = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    img_1, img_2 = onp.full((n_samps, n_samps), -onp.inf), onp.full((n_samps, n_samps), -onp.inf)

    # Plotting over (x, y, z=0.0)
    points = [[x_s + d_x / 2, y_s + d_y / 2, 0.0] for x_s in x for y_s in y if onp.sqrt(x_s ** 2 + y_s ** 2) < R]
    indices = [[i, j] for i, x_s in enumerate(x) for j, y_s in enumerate(y) if onp.sqrt(x_s ** 2 + y_s ** 2) < R]
    vals = likelihood(np.array(points))
    for idx, (i, j) in enumerate(indices):
        img_1[i, j] = vals[idx]
        # img_1[i, j] = points[idx][0]

    print('Sampled x-y plot')

    # Plotting over (x, y=0.0, z)
    points = [[x_s + d_x / 2, 0.0, z_s + d_z / 2] for x_s in x for z_s in z]
    indices = [[i, j] for i, x_s in enumerate(x) for j, z_s in enumerate(z)]
    vals = likelihood(np.array(points))
    for idx, (i, j) in enumerate(indices):
        img_2[i, j] = vals[idx]

    print('Sampled x-z plot')

    ax1.imshow(img_1.transpose(), origin='lower')
    ax2.imshow(img_2.transpose(), origin='lower')
    ax1.set_xlabel('x'), ax1.set_ylabel('y')
    ax2.set_xlabel('x'), ax2.set_ylabel('z')

    return fig, (ax1, ax2)


def single_dimensional_likelihood_plot_only_horizontal(d: CylinderDetector, activity: float, T: float, lors: list, gamma: float, mu: float):
    fig, ax = plt.subplots()

    likelihood, _ = create_likelihood(d, activity, T, lors, gamma, mu, mc_samples=10, mapped=True)

    R, H = d.dim_radius_cm, d.dim_height_cm
    n_samps = 100
    x, y, z = onp.linspace(-R, R, n_samps), onp.linspace(-R, R, n_samps), onp.linspace(-H / 2.0, H / 2.0, n_samps)
    d_x, d_y, d_z = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    img_1, img_2 = onp.full((n_samps, n_samps), -onp.inf), onp.full((n_samps, n_samps), -onp.inf)

    # Plotting over (x, y, z=0.0)
    points = [[x_s + d_x / 2, y_s + d_y / 2, 0.0] for x_s in x for y_s in y if onp.sqrt(x_s ** 2 + y_s ** 2) < R]
    indices = [[i, j] for i, x_s in enumerate(x) for j, y_s in enumerate(y) if onp.sqrt(x_s ** 2 + y_s ** 2) < R]
    vals = likelihood(np.array(points))
    for idx, (i, j) in enumerate(indices):
        img_1[i, j] = vals[idx]
        # img_1[i, j] = points[idx][0]

    print('Sampled x-y plot')

    ax.imshow(img_1.transpose(), origin='lower')
    ax.set_xlabel('x'), ax.set_ylabel('y')

    return fig, ax


def scattering_experiment_plot(d: CylinderDetector, p: StaticParticle, activity, T, gamma) -> (plt.Figure, plt.axis):
    # Generate dataset
    lors, scatters = p.simulate_emissions(detector=d, n_emissions=int(T * activity))

    # Eval likelihood over slices of detector
    fig, ax = single_dimensional_likelihood_plot(d=d, activity=activity, T=T, gamma=gamma, lors=lors, mu=p.scatter_rate)
    # fig.suptitle(rf'Likelihood, particle={p.to_str_cartesian()}, scattering rate $\mu={p.scatter_rate}$')

    return fig, ax


def scattering_experiment_plot_only_horizontal(d: CylinderDetector, p: StaticParticle, activity, T, gamma) -> (plt.Figure, plt.axis):
    # Generate dataset
    lors, scatters = p.simulate_emissions(detector=d, n_emissions=int(T * activity))

    # Eval likelihood over slices of detector
    fig, ax = single_dimensional_likelihood_plot_only_horizontal(d=d, activity=activity, T=T, gamma=gamma, lors=lors, mu=p.scatter_rate)

    return fig, ax


def plot_proj_area(min_phi, min_z, d_phi, d_z, p: StaticParticle, d: CylinderDetector, gamma: float):
    # Plots characteristic_function(R, detector_j, phi_1, z_1, X, gamma)
    detector_j = np.array([min_phi, min_z, d_phi, d_z])
    X = np.array(p.get_position_cartesian())

    fig, ax = plt.subplots()
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts}'
    })
    # fig.set_size_inches(8, 20)

    shp = (314 * 2, 50 * 2)
    phi_vals = onp.linspace(0, 2 * onp.pi, num=shp[0])
    z_vals = onp.linspace(0, 0.5, num=shp[1])

    img = onp.zeros((len(phi_vals), len(z_vals)))

    for i, phi_samp in enumerate(phi_vals):
        for j, z_samp in enumerate(z_vals):
            img[i, j] = characteristic_function(R=d.dim_radius_cm,
                                                detector_j=detector_j,
                                                phi_1=phi_samp, z_1=z_samp,
                                                X=X, gamma=gamma)

    ax.imshow(img.transpose(), origin='lower')

    # Plot original detector
    d_dphi = d_phi / (2 * onp.pi) * shp[0]
    d_dz = d_z / 0.5 * shp[1]
    rect = patches.Rectangle((min_phi / (2 * onp.pi) * shp[0], min_z / 0.5 * shp[1]), d_dphi, d_dz,
                             linewidth=1, edgecolor='r', facecolor='none', label=r'$\mathcal{D}_j$ boundary')
    ax.add_patch(rect)
    ax.legend()

    ax.set_xlabel(r'Horizontal $\phi_1 \in [0, 2\pi]$')
    ax.set_ylabel(r'Vertical $z_1 \in [0, 0.5]$')
    ax.set_title(r'$\mathbb{I}_j(\phi_1, z_1)$')

    return fig, ax


if __name__ == '__main__':
    # d = CylinderDetector()
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
    # det = CylinderDetector()
    # p = StaticParticle()
    # p.scatter_rate = 3.0
    # T = 1.0
    # X = np.array(p.get_position_cartesian())
    #
    # experiments = [
    #     {
    #         'pos': [0.0, 0.0, 0.0], 'scatter_rate': 0.001, 'activity': 10**4,
    #         'name': 'no_scatter_01', 'title': None
    #     },
    #     {
    #         'pos': [0.1, 0.1, 0.0], 'scatter_rate': 0.001, 'activity': 10**4,
    #         'name': 'no_scatter_02', 'title': None
    #     },
    #     {
    #         'pos': [-0.1, 0.0, 0.1], 'scatter_rate': 0.001, 'activity': 10**4,
    #         'name': 'no_scatter_03', 'title': None
    #     },
    #     {
    #         'pos': [0.1, 0.0, 0.0], 'scatter_rate': 1.5, 'activity': 10**4,
    #         'name': 'scatter_mu_1_5', 'title': None
    #     },
    #     {
    #         'pos': [0.1, 0.0, 0.0], 'scatter_rate': 3.0, 'activity': 10**4,
    #         'name': 'scatter_mu_3_0', 'title': None
    #     },
    #     {
    #         'pos': [0.1, 0.0, 0.0], 'scatter_rate': 8.0, 'activity': 10**4,
    #         'name': 'scatter_mu_8_0', 'title': None
    #     },
    # ]
    #
    # for exp in experiments:
    #     p.set_position_cartesian(*exp['pos'])
    #     p.scatter_rate = exp['scatter_rate']
    #     fig, ax = scattering_experiment_plot(d=det, p=p, activity=exp['activity'], T=T, gamma=50.0)
    #
    #     if exp['title']:
    #         title = exp['title']
    #         title = title.replace('PARTICLE_POS', p.get_position_cartesian())
    #         ax.set_title(title)
    #
    #     plt.savefig(f'figures/likelihood/{exp["name"]}.png', format='png')
    #     plt.savefig(f'figures/likelihood/{exp["name"]}.eps', format='eps', bbox_inches='tight')


    # p = StaticParticle()
    # d = CylinderDetector()
    # p_r, p_theta, p_z = 0.21, 0.32, 0.11
    # p.set_position_cylindrical(r=p_r, theta=p_theta, z=p_z)
    # for gamma in [5, 50, 500]:
    #     fig, ax = plot_proj_area(min_phi=np.pi / 2 - 0.05,
    #                              min_z=0.39,
    #                              d_phi=0.1,
    #                              d_z=0.1,
    #                              p=p, d=d,
    #                              gamma=gamma)
    #     plt.show()
    #     ax.set_title(ax.get_title() + r', particle at ' + p.to_str_cylindrical(True))
    #     plt.savefig(f'figures/projection_area/gamma_{gamma}.png', format='png')
    #     plt.savefig(f'figures/projection_area/gamma_{gamma}.eps', format='eps', bbox_inches='tight')

    det = CylinderDetector()
    p = StaticParticle()
    p.scatter_rate = 3.0
    T = 1.0
    X = np.array(p.get_position_cartesian())

    experiments = [
        {
            'pos': [0.1, 0.1, 0.0], 'scatter_rate': 6.0, 'activity': 10**4,
            'name': 'experiment_high_scattering', 'title': None
        },
        {
            'pos': [0.1, 0.1, 0.0], 'scatter_rate': 0.001, 'activity': 100,
            'name': 'experiment_low_lors', 'title': None
        },
    ]

    for exp in experiments:
        p.set_position_cartesian(*exp['pos'])
        p.scatter_rate = exp['scatter_rate']
        fig, ax = scattering_experiment_plot_only_horizontal(d=det, p=p, activity=exp['activity'], T=T, gamma=50.0)

        if exp['title']:
            title = exp['title']
            title = title.replace('PARTICLE_POS', p.get_position_cartesian())
            ax.set_title(title)

        plt.savefig(f'figures/likelihood/{exp["name"]}.png', format='png')
        plt.savefig(f'figures/likelihood/{exp["name"]}.eps', format='eps', bbox_inches='tight')
