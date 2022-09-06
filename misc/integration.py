from typing import Callable, List

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from geometry import Quadrilateral, MultiQuadrilateral, phi_proj, z_proj, rect_quad_intersection_area, \
    RectangleQuadrilateral
from model import CylinderDetector, StaticParticle
from plot import detector_plot

# Integration code that was discarded in the end

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


def projection_region(R: float, x: np.ndarray, detector_phi: tuple, detector_z: tuple):
    # Projects a detector cell through x, returning one or two regions mod 2pi

    phi_a, phi_b = [phi_proj(R, x, detector_phi[0]), phi_proj(R, x, detector_phi[1])]

    z_a_1, z_a_2 = z_proj(R, x, detector_phi[0], detector_z[1]), z_proj(R, x, detector_phi[0], detector_z[0])
    z_b_1, z_b_2 = z_proj(R, x, detector_phi[1], detector_z[1]), z_proj(R, x, detector_phi[1], detector_z[0])

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


# def joint_probability(d: CylinderDetector, x: np.ndarray, i: int, j: int, n_samples: int = 1000):
#     # Step 1: Compute Integral Region
#     i_region = d.detector_cell_from_index(i)
#     j_region = d.detector_cell_from_index(j)
#
#     j_proj_region = projection_region(R=d.dim_radius_cm, x=x,
#                                       detector_phi=j_region.x_range(),
#                                       detector_z=j_region.y_range())
#
#     # Step 2: Check if there is no intersection between (i) and back-project(j)
#     if not j_proj_region.intersects(i_region):
#         return 0.0
#
#     # Debug Integration region
#     # fig, ax = detector_plot(d.dim_height_cm)
#     # i_region.plot(ax, 'r')
#     # j_proj_region.plot(ax, 'b')
#     # ax.set_title('Integration regions')
#     # plt.show()
#
#     # Step 3: Perform integration
#     R = d.dim_radius_cm
#
#     def integrand(phi, z):
#         frac = np.square(z - x[2])
#         frac /= np.square(x[0] - R * np.cos(phi)) + np.square(x[1] - R * np.sin(phi)) + np.square(z - x[2])
#         return 1 / (4 * np.pi) * np.sqrt(1 - frac)
#
#     def rejection_func(phi, z):
#         return not (i_region.inside([phi, z]) and j_proj_region.inside([phi, z]))
#
#     return monte_carlo_2d_integral(x_sample_range=i_region.x_range(),
#                                    y_sample_range=i_region.y_range(),
#                                    integrand=integrand,
#                                    rejection_func=rejection_func,
#                                    n_samples=n_samples)

def joint_probability(d: CylinderDetector, x: np.ndarray, i: int, j: int):
    # Step 1: Compute Integral Region
    i_region = d.detector_cell_from_index(i)
    j_region = d.detector_cell_from_index(j)

    j_proj_region = projection_region(R=d.dim_radius_cm, x=x,
                                      detector_phi=j_region.x_range(),
                                      detector_z=j_region.y_range())

    # Step 2: Check if there is no intersection between (i) and back-project(j)
    if not j_proj_region.intersects(i_region):
        return 0.0

    # Step 3: Compute intersection area
    area_intersect = rect_quad_intersection_area(rect=i_region, quad=j_proj_region)

    # Step 4: Sample integrand at centroid
    centroid = np.mean(i_region.vertices(), axis=0)
    R = d.dim_radius_cm
    frac = np.square(centroid[1] - x[2])
    frac /= np.square(x[0] - R * np.cos(centroid[0])) + np.square(x[1] - R * np.sin(centroid[0])) \
            + np.square(centroid[1] - x[2])
    integrand_value = 1 / (4 * np.pi) * np.sqrt(1 - frac)

    # Step 5: Return estimate
    return area_intersect * integrand_value


# def marginal_probability(d: CylinderDetector, x: np.ndarray, i: int, n_samples: int = 1000):
#     # Step 1: Compute Integral Region
#     i_region = d.detector_cell_from_index(i)
#
#     i_region_proj = projection_region(R=d.dim_radius_cm, x=x,
#                                       detector_phi=i_region.x_range(),
#                                       detector_z=i_region.y_range())
#
#     detector_surface = Quadrilateral([0, 0],
#                                      [0, d.dim_height_cm],
#                                      [2 * np.pi, d.dim_height_cm],
#                                      [2 * np.pi, 0])
#
#     if not i_region_proj.intersects(detector_surface):
#         return 0.0
#
#     summation_cells = []
#     if isinstance(i_region_proj, MultiQuadrilateral):
#         for q in i_region_proj.quads:
#             summation_cells += d.detector_cells_from_region(q.x_range(), q.y_range())
#     else:
#         summation_cells = d.detector_cells_from_region(i_region_proj.x_range(), i_region_proj.y_range())
#
#     # Debug Integration region
#     # fig, ax = detector_plot(d.dim_height_cm)
#     # i_region.plot(ax, 'b')
#     # for c in summation_cells:
#     #     d.detector_cell_from_index(c).plot(ax, 'g')
#     # i_region_proj.plot(ax, 'r')
#     # plt.show()
#
#     # print('Computing marginal probability for', i)
#     # print('Projection Region', i_region_proj)
#     # print('Cells to include from that', summation_cells)
#
#     # Step 2: Compute both terms (integration)
#     R = d.dim_radius_cm
#
#     def integrand(phi, z):
#         frac = np.square(z - x[2])
#         frac /= np.square(x[0] - R * np.cos(phi)) + np.square(x[1] - R * np.sin(phi)) + np.square(z - x[2])
#         return 1 / (2 * np.pi) * np.sqrt(1 - frac)
#
#     rejection_funcs = []
#
#     def create_rejection_func(region):
#         return lambda phi, z: not (i_region.inside([phi, z]) and region.inside([phi, z]))
#
#     for j_idx in summation_cells:  # Over every possible detector cell
#         j_region = d.detector_cell_from_index(j_idx)
#
#         j_proj_region = projection_region(R=d.dim_radius_cm, x=x,
#                                           detector_phi=j_region.x_range(),
#                                           detector_z=j_region.y_range())
#         print('--> Summation cell ', j_idx)
#         print('----> Projection region', j_proj_region)
#
#         fig, ax = detector_plot(d.dim_height_cm)
#         j_region.plot(ax, 'b')
#         j_proj_region.plot(ax, 'r')
#         i_region.plot(ax, 'g')
#         plt.title('Individual projection regions')
#         plt.show()
#
#         rejection_funcs += [create_rejection_func(j_proj_region)]
#
#     # print('Executing integration...')
#     second_term = joint_probability(d, x, i, i, n_samples)
#     first_term = monte_carlo_2d_integral_multi_region(x_sample_range=i_region.x_range(),
#                                                       y_sample_range=i_region.y_range(),
#                                                       integrand=integrand,
#                                                       rejection_funcs=rejection_funcs,
#                                                       n_samples=n_samples)
#
#     return np.sum(first_term) - second_term

def marginal_probability(d: CylinderDetector, x: np.ndarray, i: int, n_samples: int = 1000):
    # Step 1: Compute Integral Region
    i_region = d.detector_cell_from_index(i)

    detector_surface = RectangleQuadrilateral([0, 0], [2 * np.pi, d.dim_height_cm])
    detector_surface_proj = projection_region(R=d.dim_radius_cm, x=x,
                                              detector_phi=detector_surface.x_range(),
                                              detector_z=detector_surface.y_range())
    print(detector_surface_proj)
    # # Step 2: Return zero quickly if at an impossible angle for a LOR to be detected
    # if not i_region_proj.intersects(detector_surface):
    #     return 0.0

    # Step 3: Compute detectors that could project onto i
    summation_cells = []
    if isinstance(i_region_proj, MultiQuadrilateral):
        for q in i_region_proj.quads:
            summation_cells += d.detector_cells_from_region(q.x_range(), q.y_range())
    else:
        summation_cells = d.detector_cells_from_region(i_region_proj.x_range(), i_region_proj.y_range())

    # # Step 4: Perform summation
    # first_term = 0.5 * sum([joint_probability(d, x, i, j) for j in summation_cells])

    # Project detector surface projection and compute first term
    if not detector_surface_proj.intersects(i_region):
        return 0.0

    fig, ax = detector_plot(d.dim_height_cm)
    i_region.plot(ax, 'g')
    detector_surface_proj.plot(ax, 'r')
    plt.title('i, surface_proj')
    plt.show()

    # Step 3: Compute intersection area
    area_intersect = rect_quad_intersection_area(rect=i_region, quad=detector_surface_proj)
    print(area_intersect)

    # Step 4: Sample integrand at centroid
    centroid = np.mean(i_region.vertices(), axis=0)
    R = d.dim_radius_cm
    frac = np.square(centroid[1] - x[2])
    frac /= np.square(x[0] - R * np.cos(centroid[0])) + np.square(x[1] - R * np.sin(centroid[0])) \
            + np.square(centroid[1] - x[2])
    integrand_value = 1 / (4 * np.pi) * np.sqrt(1 - frac)

    # Step 5: Return estimate
    first_term = 0.5 * area_intersect * integrand_value

    # Step 5: Compute D_i,i term
    second_term = joint_probability(d, x, i, i)

    return first_term - second_term

if __name__ == '__main__':
    d = CylinderDetector()
    d.detectors_width = 0.05  # 0.05
    d.detectors_height = 0.05  # 0.05

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
    # plt.savefig('figures/marginal_mc_estimate.eps', format='eps', bbox_inches='tight')
    plt.show()

    print(np.sum(integral_values))
