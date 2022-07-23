from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from geometry import Quadrilateral
from model import Detector, CylinderDetector, StaticParticle
from plot import detector_plot


def atan2(x1, x2):
    # Returns argument of the complex number x2+x1*i in the range [0, 2pi]
    angle = np.arctan2(x1, x2)
    return angle if angle > 0.0 else 2 * np.pi + angle


def monte_carlo_2d_integral(x_sample_range: tuple, y_sample_range: tuple, integrand: Callable[[float, float], float],
                            rejection_func: Callable[[float, float], bool], n_samples: int = 100):
    # Performs a monte carlo estimate ∫∫f dV ≈ V*(1/n)*∑f(x,y)
    samples = []
    v_count = 0
    for _ in range(n_samples):
        point = np.array([
            np.random.uniform(*x_sample_range),
            np.random.uniform(*y_sample_range)
        ])

        if rejection_func(*point):
            continue

        v_count += 1
        samples += [integrand(*point)]

    v_est = (x_sample_range[1] - x_sample_range[0]) * (y_sample_range[1] - y_sample_range[0]) * (v_count / n_samples)
    return v_est * 1 / len(samples) * sum(samples) if len(samples) > 0 else 0.0


def phi_2(R: float, X: np.ndarray, phi_1: float):
    # Compute phi_2 given (phi_1, x)
    x, y, _ = X
    phi = atan2(R * np.sin(phi_1) - y, R * np.cos(phi_1) - x)

    mu_2_const = x * np.cos(phi) + y * np.sin(phi)
    mu_2 = -(mu_2_const)
    mu_2 -= np.sqrt(np.square((mu_2_const)) - (np.square(x) + np.square(y) - np.square(R)))

    return atan2(y + mu_2 * np.cos(phi), x + mu_2 * np.sin(phi))


def z_2(R: float, X: np.ndarray, phi_1: float, z_1: float):
    # Compute z_2 given (phi_1, z_1, x)
    x, y, z = X
    phi = atan2(R * np.sin(phi_1) - y, R * np.cos(phi_1) - x)

    l_1 = np.linalg.norm(X - np.array([R * np.cos(phi_1), R * np.sin(phi_1), z_1]))
    theta = np.arccos((z_1 - z) / l_1)

    l_solved = np.roots([
        np.square(np.sin(theta)),
        2 * np.sin(theta) * (x * np.cos(phi) + y * np.sin(phi)),
        np.square(x) + np.square(y) - np.square(R)
    ])
    l_solved.sort()
    l_2, l_1_check = l_solved

    return z + l_2 * np.cos(theta)


def projection_region(R: float, x: np.ndarray, detector_phi: tuple, detector_z: tuple) -> (tuple, tuple):
    # Projects a detector cell through x, returning a new region on the detector
    phi_range = [phi_2(R, x, detector_phi[0]), phi_2(R, x, detector_phi[1])]
    phi_range.sort()

    z_bounds_1 = [z_2(R, x, detector_phi[0], detector_z[0]),
                  z_2(R, x, detector_phi[0], detector_z[1])]
    z_bounds_1.sort()

    z_bounds_2 = [z_2(R, x, detector_phi[1], detector_z[0]),
                  z_2(R, x, detector_phi[1], detector_z[1])]
    z_bounds_2.sort()

    return Quadrilateral([phi_range[0], z_bounds_1[0]],
                         [phi_range[0], z_bounds_1[1]],
                         [phi_range[1], z_bounds_2[1]],
                         [phi_range[1], z_bounds_2[0]])


def joint_probability(d: CylinderDetector, x: np.ndarray, i: int, j: int, n_samples: int = 1000):
    # Step 1: Compute Integral Region
    i_region = d.detector_cell_from_index(i)
    j_region = d.detector_cell_from_index(j)

    j_proj_region = projection_region(R=d.dim_radius_cm, x=x,
                                      detector_phi=j_region.x_range(),
                                      detector_z=j_region.y_range())

    phi_min = min([*i_region.x_range(), *j_proj_region.x_range()])
    phi_max = max([*i_region.x_range(), *j_proj_region.x_range()])
    z_min = min([*i_region.y_range(), *j_proj_region.y_range()])
    z_max = max([*i_region.y_range(), *j_proj_region.y_range()])

    # Step 2: Check if there is no intersection between (i) and back-project(j)
    if not i_region.intersects(j_proj_region):
        return 0.0

    # Debug Integration region
    # fig, ax = detector_plot(d.dim_height_cm)
    # i_region.plot(ax, 'r')
    # j_proj_region.plot(ax, 'b')
    # ax.set_title('Integration regions')
    # plt.show()

    # Step 3: Perform integration
    R = d.dim_radius_cm

    def integrand(phi, z):
        frac = np.square(z - x[2])
        frac /= np.square(x[0] - R * np.cos(phi)) + np.square(x[1] - R * np.sin(phi)) + np.square(z - x[2])
        return 1 / (4 * np.pi) * np.sqrt(1 - frac)

    def rejection_func(phi, z):
        return not (i_region.inside([phi, z]) and j_proj_region.inside([phi, z]))

    return monte_carlo_2d_integral(x_sample_range=(phi_min, phi_max),
                                   y_sample_range=(z_min, z_max),
                                   integrand=integrand,
                                   rejection_func=rejection_func,
                                   n_samples=n_samples)


def marginal_probability(d: CylinderDetector, x: np.ndarray, i: int, n_samples: int = 1000):
    # Step 1: Compute Integral Region
    i_region = d.detector_cell_from_index(i)

    # Debug Integration region
    # fig, ax = detector_plot(d.dim_height_cm)
    # i_region.plot(ax, 'r')
    # j_proj_region.plot(ax, 'b')
    # ax.set_title('Integration regions')
    # plt.show()

    # Step 2: Compute both terms (integration)
    R = d.dim_radius_cm

    def integrand(phi, z):
        frac = np.square(z - x[2])
        frac /= np.square(x[0] - R * np.cos(phi)) + np.square(x[1] - R * np.sin(phi)) + np.square(z - x[2])
        return 1 / (2 * np.pi) * np.sqrt(1 - frac)

    def rejection_func(phi, z):
        return not (0.0 <= z_2(R=R, X=x, phi_1=phi, z_1=z) <= d.dim_height_cm)

    second_term = joint_probability(d, x, i, i, n_samples)
    first_term = monte_carlo_2d_integral(x_sample_range=i_region.x_range(),
                                         y_sample_range=i_region.y_range(),
                                         integrand=integrand,
                                         rejection_func=rejection_func,
                                         n_samples=n_samples)

    return first_term - second_term


if __name__ == '__main__':
    d = CylinderDetector()
    # d.detectors_width = 0.05
    # d.detectors_height = 0.05

    p = StaticParticle()
    p.set_position_cylindrical(r=0.96 * d.dim_radius_cm, theta=np.pi, z=d.dim_height_cm / 2)
    # p.set_position_cylindrical(r=0.0, theta=1.5, z=d.dim_height_cm/2)

    n_x, n_y = d.n_detector_cells()

    #
    # JOINT PROBABILITY PLOT
    #
    # from_cell = int(n_x/4) + int(n_y/2)*n_x
    # from_cell_region = d.detector_cell_from_index(from_cell)
    # fig, ax = detector_plot(d.dim_height_cm)
    # from_cell_region.plot(ax)
    # ax.set_title('From region')
    # plt.show()
    #
    # integral_values = np.zeros((n_x, n_y))
    # xv, yv = np.meshgrid(range(n_x), range(n_y), indexing='ij')
    # for i in range(n_x):
    #     print(f'{i}/{n_x}')
    #     for j in range(n_y):
    #         x, y = xv[i, j], yv[i, j]
    #         idx = x + y*n_x
    #         integral_values[i, j] = joint_probability(d, p.get_position_cartesian(), from_cell, idx, 1000)
    #
    # # Checking we a viewing this right
    # # integral_values[0, 0] = 1.0
    # # integral_values[n_x-1, n_y-1] = 0.5
    # plt.imshow(integral_values.transpose(), origin='lower')
    # plt.colorbar()
    # plt.show()

    #
    # MARGINAL PROBABILITY PLOT
    #
    integral_values = np.zeros((n_x, n_y))
    xv, yv = np.meshgrid(range(n_x), range(n_y), indexing='ij')
    for i in range(n_x):
        print(f'{i}/{n_x}')
        for j in range(n_y):
            x, y = xv[i, j], yv[i, j]
            idx = x + y * n_x
            integral_values[i, j] = marginal_probability(d, p.get_position_cartesian(), idx, 10)

    plt.imshow(integral_values.transpose(), origin='lower')
    plt.title(f'Marginal Probability given {p}')
    plt.colorbar()
    plt.show()

    print(np.sum(integral_values))