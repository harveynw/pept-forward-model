import abc
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from plot import point_3d, line_3d, arrow_3d
from geometry import azimuth_of_point, Point


class Detector:
    @abc.abstractmethod
    def impact(self, lor_normal: np.ndarray, lor_annihilation: np.ndarray) -> (bool, float, float):
        pass

    @abc.abstractmethod
    def impact_forward_only(self, lor_normal: np.ndarray, lor_annihilation: np.ndarray) -> (bool, float):
        pass

    @abc.abstractmethod
    def detector_cell_from_impact(self, impact: np.ndarray) -> (int, int):
        pass


@dataclass
class CylinderDetector(Detector):
    dim_radius_cm: float = 0.25
    dim_height_cm: float = 0.50

    detectors_height: float = 0.005
    detectors_width: float = 0.005

    def impact(self, lor_normal: np.ndarray, lor_annihilation: np.ndarray):
        p_x, p_y, p_z = lor_annihilation
        n_1, n_2, n_3 = lor_normal

        # Find intersection
        a = np.square(n_1) + np.square(n_2)
        b = 2 * n_1 * p_x + 2 * p_y * n_2
        c = np.square(p_x) + np.square(p_y) - np.square(self.dim_radius_cm)

        lambda_1, lambda_2 = np.roots([a, b, c])
        if lambda_2 < lambda_1:
            lambda_1, lambda_2 = lambda_2, lambda_1

        impact_1, impact_2 = lor_annihilation + lambda_1*lor_normal, lor_annihilation + lambda_2*lor_normal

        did_impact = (0 < impact_1[2] < self.dim_height_cm) and (0 < impact_2[2] < self.dim_height_cm)

        return did_impact, lambda_1, lambda_2

    def impact_forward_only(self, lor_normal: np.ndarray, lor_annihilation: np.ndarray) -> (bool, float):
        _, _, l = self.impact(lor_normal=lor_normal, lor_annihilation=lor_annihilation)

        impact = lor_annihilation + l*lor_normal
        return 0 < impact[2] < self.dim_height_cm, l

    def detector_cell_from_impact(self, impact: np.ndarray) -> (int, int):
        # Impact MUST lie on detector, otherwise this function is undefined
        x, y, z = impact

        phi = azimuth_of_point(x, y)

    def debug_plot(self, ax: plt.axis):
        # Plots the cylinder detector

        diameter = 2 * np.pi * self.dim_radius_cm
        n_detectors_horizontal = int(diameter / self.detectors_width)
        n_detectors_vertical = int(self.dim_height_cm / self.detectors_height)
        print('Detectors:', n_detectors_horizontal, n_detectors_vertical, n_detectors_horizontal * n_detectors_vertical)

        # Rough approximation:
        # n_detectors_horizontal, n_detectors_vertical = 10, 10

        theta_grid, z_grid = np.meshgrid(np.linspace(0, 2 * np.pi, n_detectors_horizontal),
                                         np.linspace(0, self.dim_height_cm, n_detectors_vertical))
        x_grid = self.dim_radius_cm * np.cos(theta_grid)  # + center_x
        y_grid = self.dim_radius_cm * np.sin(theta_grid)  # + center_y

        ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5)


@dataclass
class StaticParticle(Point):

    # Compton Scattering Rate
    scatter_rate: float = 2.0

    @staticmethod
    def _generate_scatter_rotation() -> np.ndarray:
        # Samples a change in trajectory of a particle due to Compton scattering, returning a 3D rotation of the z_axis
        phi = np.random.uniform(low=0, high=2*np.pi)
        theta = np.random.vonmises(mu=0, kappa=1)

        rot_theta = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])

        rot_phi = np.array([
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi), np.cos(phi), 0],
            [0, 0, 1]
        ])

        return np.matmul(rot_phi, rot_theta)

    def simulate_emissions(self, detector: Detector, n_lor=0.05 * (10 ** 4), debug_ax=None):
        if debug_ax is not None:
            point_3d(debug_ax, self.get_position_cartesian(), color='r', label='Particle Position')

        impacts = []
        n_scatters = 0

        for _ in range(n_lor):  # For each requested LOR
            plane_phi = np.random.uniform(0, 2 * np.pi)
            plane_theta = np.arccos(1-2*np.random.uniform(0, 1))  # Inverse Transform Sampling

            # Normal vector to plane, defining the LOR direction
            e_phi = np.array([
                np.cos(plane_phi),
                np.sin(plane_phi),
                0.0
            ])
            e_theta = np.array([
                -np.sin(plane_theta) * np.sin(plane_phi),
                np.sin(plane_theta * np.cos(plane_phi)),
                np.square(np.cos(plane_theta))
            ])
            n = np.cross(e_phi, e_theta)
            n = n / np.linalg.norm(n)

            # Compute collisions with detector:
            did_impact, lambda_1, lambda_2 = detector.impact(lor_normal=n,
                                                             lor_annihilation=self.get_position_cartesian())

            if not did_impact:
                continue

            # Note lambda_1 < 0 < lambda_2 and they represent distance in cm as n is a unit vector

            if debug_ax is not None and did_impact:
                line_3d(debug_ax, self.get_position_cartesian(), n, lambda_1, lambda_2, 50,
                        color='grey', label='Initial LOR')

                impact_1, impact_2 = self.get_position_cartesian() + lambda_1 * n, \
                                     self.get_position_cartesian() + lambda_2 * n

                point_3d(debug_ax, impact_1, color='grey')
                point_3d(debug_ax, impact_2, color='grey')

            # Compton Scattering
            final_impacts = []
            was_detected = True

            for l in [lambda_1, lambda_2]:
                distance = np.abs(l)

                # Model parameter here! Exponential distribution representing chance of collision before detector impact
                first_scatter = np.random.exponential(scale=1.0/self.scatter_rate)

                if first_scatter < distance:  # Compton scatter occurred
                    n_scatters += 1

                    scatter_point = self.get_position_cartesian() + (np.sign(l)*first_scatter)*n

                    # Compute new normal (with a random rotation)
                    change_of_basis = np.array([e_phi, e_theta, np.sign(l)*n]).transpose()
                    new_n = change_of_basis.dot(self._generate_scatter_rotation().dot([0, 0, 1]))

                    if debug_ax is not None:
                        point_3d(debug_ax, scatter_point, color='darkred', label='Compton Scatter')
                        arrow_3d(debug_ax, scatter_point, new_n, length=0.05, color='orange')

                    did_impact_scattered, scatter_lambda = detector.impact_forward_only(lor_normal=new_n,
                                                                                        lor_annihilation=scatter_point)

                    if not did_impact_scattered:
                        # Entire LOR is discarded
                        was_detected = False

                    if debug_ax is not None:
                        if did_impact_scattered:
                            line_3d(debug_ax, scatter_point, new_n, 0, scatter_lambda, 10, color='green')
                            point_3d(debug_ax, scatter_point + scatter_lambda * new_n, color='green',
                                     s=5, label='New impact')
                        else:
                            line_3d(debug_ax, scatter_point, new_n, 0, 1, 10, color='red')

                    # Final impact after scattering
                    final_impacts.append(scatter_point + scatter_lambda*new_n)
                else:
                    # Original trajectory
                    final_impacts.append(self.get_position_cartesian() + l * n)

            if was_detected:
                impacts += [final_impacts]

                if debug_ax is not None:
                    line_3d(debug_ax, final_impacts[0], -final_impacts[0]+final_impacts[1], 0, 1, 10, color='black',
                            label='LOR')

        return impacts, n_scatters
