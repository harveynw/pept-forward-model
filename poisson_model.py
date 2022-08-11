import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt

from jax import jit, vmap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from model import CylinderDetector


@jit
def _solid_angle_integrand(R, H, X, varphi):
    x, y, z = X

    R_varphi = x * np.cos(varphi) + y * np.sin(varphi)

    theta_1 = (R_varphi - np.sqrt(R_varphi**2 - (x**2 + y**2 - R**2)))/(z - H / 2.0)
    theta_2 = (R_varphi + np.sqrt(R_varphi**2 - (x**2 + y**2 - R**2)))/(z + H / 2.0)

    theta_min = np.min(np.array([theta_1, theta_2]))

    return np.arctan(theta_min)

@jit
def G_solid_angle_approx(R, H, X):
    # Trapezoidal Rule
    d_varphi = 2.0*np.pi/10
    varphi = np.arange(start=d_varphi, stop=2*np.pi, step=d_varphi)
    integrand_vmap = vmap(_solid_angle_integrand, (None, None, None, 0), 0)
    f_x_0 = _solid_angle_integrand(R, H, X, 0.0)
    f_x_N = _solid_angle_integrand(R, H, X, 2*np.pi)
    return 1 / (2*np.pi) * d_varphi / 2.0 * (f_x_0 + f_x_N + 2.0*np.sum(integrand_vmap(R, H, X, varphi)))


# @jit
# def single_particle_likelihood(R, i_n: np.array, j_n: np.array, X: np.array):


if __name__ == '__main__':
    d = CylinderDetector()
    print(G_solid_angle_approx(R=d.dim_radius_cm, H=d.dim_height_cm, X=np.array([0.1, 0.1, 0.0])))
    print(G_solid_angle_approx(R=d.dim_radius_cm, H=d.dim_height_cm, X=np.array([-0.1, -0.1, 0.0])))
    print(G_solid_angle_approx(R=d.dim_radius_cm, H=d.dim_height_cm, X=np.array([0.0, 0.0, 0.1])))
    print(G_solid_angle_approx(R=d.dim_radius_cm, H=d.dim_height_cm, X=np.array([0.0, 0.0, 0.0])))

    d.dim_height_cm = d.dim_radius_cm

    fig, ax = plt.subplots()
    x_samples = onp.linspace(-d.dim_radius_cm+0.0001, d.dim_radius_cm-0.0001, 200)
    z_samples = onp.linspace(-d.dim_height_cm/2.0+0.0001, d.dim_height_cm/2.0-0.0001, 100)

    im = onp.zeros((len(x_samples), len(z_samples)))
    for i in range(len(x_samples)):
        for j in range(len(z_samples)):
            im[i, j] = G_solid_angle_approx(d.dim_radius_cm, d.dim_height_cm,
                                            X=np.array([x_samples[i], 0.0, z_samples[j]]))

    plt_im = ax.imshow(im.transpose(), origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(plt_im, cax=cax, orientation='vertical')
    plt.show()

