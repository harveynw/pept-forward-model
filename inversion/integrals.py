import jax.numpy as np

from jax import jit, vmap
from jax_implementation import F_lambdas


@jit
def _G_solid_angle_integrand(R, H, X, varphi):
    x, y, z = X

    R_varphi = x * np.cos(varphi) + y * np.sin(varphi)

    theta_1 = (-R_varphi + np.sqrt(R_varphi ** 2 - (x ** 2 + y ** 2 - R ** 2))) / (H / 2.0 - z)
    theta_2 = (-R_varphi - np.sqrt(R_varphi ** 2 - (x ** 2 + y ** 2 - R ** 2))) / (-H / 2.0 - z)

    theta_max = np.max(np.array([theta_1, theta_2]))

    return np.cos(np.arctan(theta_max))


@jit
def _H_solid_angle_integrand(R, H, X, varphi):
    x, y, z = X

    d = x * np.cos(varphi) + y * np.sin(varphi)
    r_varphi = -d + np.sqrt(d ** 2 - (x ** 2 + y ** 2 - R ** 2))

    theta_min = np.arctan(r_varphi / (H / 2.0 - z))
    theta_max = np.arctan(r_varphi / (-H / 2.0 - z)) + np.pi

    return np.cos(theta_min) - np.cos(theta_max)


@jit
def H_integral(R, H, X):
    # Trapezoidal Rule Approximation of Solid Angle Ratio
    d_varphi = 2 * np.pi / 100
    varphi = np.arange(start=d_varphi, stop=2 * np.pi, step=d_varphi)
    integrand_vmap = vmap(_H_solid_angle_integrand, (None, None, None, 0), 0)
    f_x_0 = _H_solid_angle_integrand(R, H, X, 0.0)
    f_x_N = _H_solid_angle_integrand(R, H, X, 2 * np.pi)
    return (1 / (2 * np.pi)) * (d_varphi / 2.0) * (f_x_0 + f_x_N + 2.0 * np.sum(integrand_vmap(R, H, X, varphi)))


@jit
def G_integral(R, H, X):
    # Trapezoidal Rule Approximation of Solid Angle Ratio
    d_varphi = 2 * np.pi / 100
    varphi = np.arange(start=d_varphi, stop=2 * np.pi, step=d_varphi)
    integrand_vmap = vmap(_G_solid_angle_integrand, (None, None, None, 0), 0)
    f_x_0 = _G_solid_angle_integrand(R, H, X, 0.0)
    f_x_N = _G_solid_angle_integrand(R, H, X, 2 * np.pi)
    return (1 / (2 * np.pi)) * (d_varphi / 2.0) * (f_x_0 + f_x_N + 2.0 * np.sum(integrand_vmap(R, H, X, varphi)))


@jit
def _scattering_density_integrand(R, varphi, theta, X):
    mu = 3.0
    l_1, l_2 = F_lambdas(R, varphi, theta, X)
    e_1, e_2 = np.exp(-mu * l_1), np.exp(mu * l_2)

    return (1 / (2 * np.pi)) * np.sin(theta) * np.array([
        (1 - e_1) * (1 - e_2),
        (1 - e_1) * e_2,
        e_1 * (1 - e_2),
        e_1 * e_2
    ])


@jit
def _scattering_density_inner_integral(R, varphi, thetas, X):
    integrand_vmap = vmap(_scattering_density_integrand, (None, None, 0, None), 0)

    f_x_inner = np.sum(integrand_vmap(R, varphi, thetas, X), axis=0)
    f_x_0 = _scattering_density_integrand(R, varphi, 0.0, X)
    f_x_N = _scattering_density_integrand(R, varphi, np.pi / 2, X)

    return f_x_0 + 2 * f_x_inner + f_x_N


@jit
def scattering_density(R, X):
    # Returns a 4-dim vector for the probability of each possible value of S = [S_1, S_2]
    # Numericallly integrated by double application of trapezium rule

    d_varphi = 2 * np.pi / 100
    varphis = np.arange(start=d_varphi, stop=2 * np.pi, step=d_varphi)

    d_theta = (np.pi / 2) / 100
    thetas = np.arange(start=d_theta, stop=np.pi / 2, step=d_theta)

    integrand_vmap = vmap(_scattering_density_inner_integral, (None, 0, None, None), 0)

    f_x_inner = np.sum(integrand_vmap(R, varphis, thetas, X), axis=0)
    f_x_0 = _scattering_density_inner_integral(R, 0.0, thetas, X)
    f_x_N = _scattering_density_inner_integral(R, 2 * np.pi, thetas, X)

    return d_varphi / 2.0 * d_theta / 2.0 * (f_x_0 + 2 * f_x_inner + f_x_N)