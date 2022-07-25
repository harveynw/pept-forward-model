from typing import Callable, List

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from geometry import Quadrilateral, MultiQuadrilateral
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


def monte_carlo_2d_integral_multi_region(x_sample_range: tuple, y_sample_range: tuple,
                                         integrand: Callable[[float, float], float],
                                         rejection_funcs: List[Callable[[float, float], bool]], n_samples: int = 100):
    # Performs same function as monte_carlo_2d_integral, but over multiple 2d integration regions (specified by a list
    # of rejection functions) by reusing samples. Result is a list of integration estimates.
    k = len(rejection_funcs)
    samples = np.zeros(k)
    samples_n_count = np.zeros(k)

    for _ in range(n_samples):
        point = np.array([
            np.random.uniform(*x_sample_range),
            np.random.uniform(*y_sample_range)
        ])

        accept = ~np.array([f(*point) for f in rejection_funcs], dtype=bool)
        samples_n_count = samples_n_count + accept

        evaluated = integrand(*point)
        samples = samples + accept * evaluated

    w, h = x_sample_range[1] - x_sample_range[0], y_sample_range[1] - y_sample_range[0]
    v_est = w * h * (samples_n_count / n_samples)
    return v_est * np.divide(1, samples_n_count, out=np.zeros(samples_n_count.shape, dtype=float),
                             where=samples_n_count != 0) * samples


# def phi_2(R: float, X: np.ndarray, phi_1: float):
#     # Compute phi_2 given (phi_1, x)
#     x, y, _ = X
#     phi = atan2(R * np.sin(phi_1) - y, R * np.cos(phi_1) - x)
#
#     mu_2_const = x * np.cos(phi) + y * np.sin(phi)
#     mu_2 = -(mu_2_const)
#     mu_2 -= np.sqrt(np.square((mu_2_const)) - (np.square(x) + np.square(y) - np.square(R)))
#
#     return atan2(y + mu_2 * np.cos(phi), x + mu_2 * np.sin(phi))

def phi_2(R: float, X: np.ndarray, phi_1: float):
    a_1 = R * np.cos(phi_1)
    a_2 = R * np.sin(phi_1)

    b_1 = X[0] - a_1
    b_2 = X[1] - a_2

    sols = np.roots([
        b_1 ** 2 + b_2 ** 2,
        2 * (a_1 * b_1 + a_2 * b_2),
        a_1 ** 2 + a_2 ** 2 - R ** 2
    ])
    sols.sort()
    mu = sols[1]  # Has to be the positive root
    return atan2(a_2 + mu * b_2, a_1 + mu * b_1)


def z_2(R: float, X: np.ndarray, phi_1: float, z_1: float):
    a_1 = R * np.cos(phi_1)
    a_2 = R * np.sin(phi_1)

    b_1 = X[0] - a_1
    b_2 = X[1] - a_2

    sols = np.roots([
        b_1 ** 2 + b_2 ** 2,
        2 * (a_1 * b_1 + a_2 * b_2),
        a_1 ** 2 + a_2 ** 2 - R ** 2
    ])
    sols.sort()
    mu = sols[1]  # Has to be the positive root
    return z_1 + mu * (X[2] - z_1)


def projection_region(R: float, x: np.ndarray, detector_phi: tuple, detector_z: tuple):
    # Projects a detector cell through x, returning one or two regions mod 2pi

    phi_a, phi_b = [phi_2(R, x, detector_phi[0]), phi_2(R, x, detector_phi[1])]

    z_a = [z_2(R, x, detector_phi[0], detector_z[0]),
           z_2(R, x, detector_phi[0], detector_z[1])]
    z_a.sort()
    z_a_1, z_a_2 = z_a

    z_b = [z_2(R, x, detector_phi[1], detector_z[0]),
           z_2(R, x, detector_phi[1], detector_z[1])]
    z_b.sort()
    z_b_1, z_b_2 = z_b

    if phi_b < phi_a:
        # If phi_b < phi_a we need two quadrilaterals on either end of [0, 2pi]
        dist_1, dist_2 = 2 * np.pi - phi_a, phi_b

        del_1 = (z_b_1 - z_a_1) / (phi_b - phi_a)  # Interpolate bottom
        del_2 = (z_b_2 - z_a_2) / (phi_b - phi_a)  # Interpolate top

        return MultiQuadrilateral([Quadrilateral([phi_a, z_a_1],
                                                 [phi_a, z_a_2],
                                                 [2 * np.pi, z_a_2 + del_2 * dist_1],
                                                 [2 * np.pi, z_a_1 + del_1 * dist_1]),
                                   Quadrilateral([0, z_a_1 + del_1 * dist_1],
                                                 [0, z_a_2 + del_2 * dist_1],
                                                 [phi_b, z_b_2],
                                                 [phi_b, z_b_1])])
    else:
        # Only need one quadrilateral, doesn't cross phi=2*pi
        return Quadrilateral([phi_a, z_a_1],
                             [phi_a, z_a_2],
                             [phi_b, z_b_2],
                             [phi_b, z_b_1])


def joint_probability(d: CylinderDetector, x: np.ndarray, i: int, j: int, n_samples: int = 1000):
    # Step 1: Compute Integral Region
    i_region = d.detector_cell_from_index(i)
    j_region = d.detector_cell_from_index(j)

    j_proj_region = projection_region(R=d.dim_radius_cm, x=x,
                                      detector_phi=j_region.x_range(),
                                      detector_z=j_region.y_range())

    # Step 2: Check if there is no intersection between (i) and back-project(j)
    if not j_proj_region.intersects(i_region):
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

    return monte_carlo_2d_integral(x_sample_range=i_region.x_range(),
                                   y_sample_range=i_region.y_range(),
                                   integrand=integrand,
                                   rejection_func=rejection_func,
                                   n_samples=n_samples)


def marginal_probability(d: CylinderDetector, x: np.ndarray, i: int, n_samples: int = 1000):
    # Step 1: Compute Integral Region
    i_region = d.detector_cell_from_index(i)

    i_region_proj = projection_region(R=d.dim_radius_cm, x=x,
                                      detector_phi=i_region.x_range(),
                                      detector_z=i_region.y_range())

    detector_surface = Quadrilateral([0, 0],
                                     [0, d.dim_height_cm],
                                     [2 * np.pi, d.dim_height_cm],
                                     [2 * np.pi, 0])

    if not i_region_proj.intersects(detector_surface):
        return 0.0

    summation_cells = []
    if isinstance(i_region_proj, MultiQuadrilateral):
        for q in i_region_proj.quads:
            summation_cells += d.detector_cells_from_region(q.x_range(), q.y_range())
    else:
        summation_cells = d.detector_cells_from_region(i_region_proj.x_range(), i_region_proj.y_range())

    # Debug Integration region
    # fig, ax = detector_plot(d.dim_height_cm)
    # i_region.plot(ax, 'b')
    # for c in summation_cells:
    #     d.detector_cell_from_index(c).plot(ax, 'g')
    # i_region_proj.plot(ax, 'r')
    # plt.show()

    # print('Computing marginal probability for', i)
    # print('Projection Region', i_region_proj)
    # print('Cells to include from that', summation_cells)

    # Step 2: Compute both terms (integration)
    R = d.dim_radius_cm

    def integrand(phi, z):
        frac = np.square(z - x[2])
        frac /= np.square(x[0] - R * np.cos(phi)) + np.square(x[1] - R * np.sin(phi)) + np.square(z - x[2])
        return 1 / (2 * np.pi) * np.sqrt(1 - frac)

    rejection_funcs = []

    def create_rejection_func(region):
        return lambda phi, z: not (i_region.inside([phi, z]) and region.inside([phi, z]))

    for j_idx in summation_cells:  # Over every possible detector cell
        j_region = d.detector_cell_from_index(j_idx)

        j_proj_region = projection_region(R=d.dim_radius_cm, x=x,
                                          detector_phi=j_region.x_range(),
                                          detector_z=j_region.y_range())
        print('--> Summation cell ', j_idx)
        print('----> Projection region', j_proj_region)

        fig, ax = detector_plot(d.dim_height_cm)
        j_region.plot(ax, 'b')
        j_proj_region.plot(ax, 'r')
        i_region.plot(ax, 'g')
        plt.title('Individual projection regions')
        plt.show()

        rejection_funcs += [create_rejection_func(j_proj_region)]

    # print('Executing integration...')
    second_term = joint_probability(d, x, i, i, n_samples)
    first_term = monte_carlo_2d_integral_multi_region(x_sample_range=i_region.x_range(),
                                                      y_sample_range=i_region.y_range(),
                                                      integrand=integrand,
                                                      rejection_funcs=rejection_funcs,
                                                      n_samples=n_samples)

    return np.sum(first_term) - second_term


if __name__ == '__main__':
    d = CylinderDetector()
    d.detectors_width = 0.01  # 0.05
    d.detectors_height = 0.01  # 0.05

    p = StaticParticle()
    R = d.dim_radius_cm
    p.set_position_cylindrical(r=0.96 * d.dim_radius_cm, theta=np.pi, z=d.dim_height_cm / 2)

    n_x, n_y = d.n_detector_cells()

    # phi_a, phi_b = (np.pi/8, np.pi/8 + 0.1)#(0.0, 0.0400202885807617)
    #
    # proj_phi_a, proj_phi_b = phi_2_working(d.dim_radius_cm, p.get_position_cartesian(), phi_a), \
    #                          phi_2_working(d.dim_radius_cm, p.get_position_cartesian(), phi_b)
    #
    # fig, ax = plt.subplots()
    # ax.add_patch(plt.Circle((0, 0), radius=R, fill=False))
    #
    # def draw_line(p1, p2):
    #     x_values = [p1[0], p2[0]]
    #     y_values = [p1[1], p2[1]]
    #     plt.plot(x_values, y_values, 'bo', linestyle="--")
    #
    # def point(phi):
    #     return [R*np.cos(phi), R*np.sin(phi)]
    #
    # draw_line(point(phi_a), point(proj_phi_a))
    # draw_line(point(phi_b), point(proj_phi_b))
    #
    # i, j, _ = p.get_position_cartesian()
    # plt.text(i, j, "x")
    # plt.scatter([i], [j])
    #
    # plt.show()

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
    for i in tqdm(range(n_x)):
        for j in range(n_y):
            x, y = xv[i, j], yv[i, j]
            idx = x + y * n_x
            integral_values[i, j] = marginal_probability(d, p.get_position_cartesian(), idx, 10)

    integral_values = integral_values / np.sum(integral_values)
    plt.imshow(integral_values.transpose(), origin='lower')
    plt.title(fr'MC Estimate of Marginal Probability for particle at {p.to_str_cylindrical(latex=True)}')
    plt.xlabel('Horizontal')
    plt.ylabel('Vertical')
    # plt.colorbar()
    plt.savefig('figures/marginal_mc_estimate.eps', format='eps', bbox_inches='tight')
    plt.show()

    print(np.sum(integral_values))
