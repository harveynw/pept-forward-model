import abc
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass


def debug_line_3d(ax: plt.axis, start_point: np.ndarray, normal: np.ndarray, lambda_start: float, lambda_end: float,
                  n_steps: int, **kwargs):
    trace_param = np.linspace(start=lambda_start, stop=lambda_end, num=n_steps)
    trace = np.array([start_point + l * normal for l in trace_param])
    ax.plot(xs=trace[:, 0], ys=trace[:, 1], zs=trace[:, 2], **kwargs)


def debug_point_3d(ax: plt.axis, point: np.ndarray, **kwargs):
    ax.scatter([point[0]], [point[1]], [point[2]], **kwargs)


def debug_arrow_3d(ax: plt.axis, origin: np.ndarray, dir: np.ndarray, **kwargs):
    u, v, w = [dir[0]], [dir[1]], [dir[2]]
    ax.quiver([origin[0]], [origin[1]], [origin[2]], u, v, w, **kwargs)


class Detector:
    @abc.abstractmethod
    def impact(self, lor_normal: np.ndarray, lor_annihilation: np.ndarray):
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
class StaticParticle:
    # Store position in spherical coordinates (r, θ, φ)
    r: float = 0.0
    theta: float = 0.0
    phi: float = 0.0

    # Compton Scattering Rate
    scatter_rate: float = 2.0

    def set_position_cartesian(self, x: float, y: float, z: float):
        # Sets the position of the particle according to the cartesian coordinate system
        self.r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
        self.theta = np.arccos(z / self.r)

        if np.isclose(x, 0.0):
            self.phi = np.pi / 2.0 if y > 0.0 else -np.pi / 2.0
        elif x > 0.0:
            self.phi = np.arctan(y / x)
        else:
            self.phi = np.arctan(y / x) + np.pi if y >= 0.0 else -np.pi

    def get_position_cartesian(self):
        x = self.r * np.cos(self.phi) * np.sin(self.theta)
        y = self.r * np.sin(self.phi) * np.sin(self.theta)
        z = self.r * np.cos(self.theta)
        return np.array([x, y, z])

    @staticmethod
    def _generate_scatter_rotation() -> np.ndarray:
        # Samples a change in trajectory of a particle due to Compton scattering, returning a 3D rotation
        delta_phi = np.random.vonmises(mu=0, kappa=1)
        delta_theta = np.random.uniform(low=-np.pi, high=np.pi)

        rot_theta = np.array([
            [1, 0, 0],
            [0, np.cos(delta_theta), -np.sin(delta_theta)],
            [0, np.sin(delta_theta), np.cos(delta_theta)]
        ])

        rot_phi = np.array([
            [np.cos(delta_phi), -np.sin(delta_phi), 0],
            [np.sin(delta_phi), np.cos(delta_phi), 0],
            [0, 0, 1]
        ])

        return np.matmul(rot_phi, rot_theta)

    def simulate_emissions(self, detector: Detector, n_lor=0.05 * (10 ** 4), debug_ax=None):
        if debug_ax is not None:
            debug_point_3d(debug_ax, self.get_position_cartesian(), color='r', label='Particle Position')

        impacts = []
        n_scatters = 0

        for _ in range(n_lor):  # For each requested LOR
            plane_phi = np.random.uniform(0, 2 * np.pi)
            plane_theta = np.random.uniform(0, np.pi)

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

            ### DEBUG PLOTTING ###
            # ax = plt.figure().add_subplot(projection='3d')
            # debug_arrow_3d(ax, np.zeros(3), e_phi, label=r'$e_{\phi}$', color='r', length=0.1)
            # debug_arrow_3d(ax, np.zeros(3), e_theta, label=r'$e_{\theta}$', color='r', length=0.1)
            # debug_arrow_3d(ax, np.zeros(3), n, label=r'$n$', color='b', length=0.1)
            # plt.title('Random plane basis and normal')
            # plt.legend()
            # plt.show()
            # exit()
            ###

            ### DEBUG Test angle sample ###
            # ax = plt.figure().add_subplot(projection='3d')
            # z_axis = np.array([0.0, 0.0, 1.0])
            # for _ in range(100):
            #     debug_arrow_3d(ax, np.zeros(3), self._generate_scatter_rotation().dot(z_axis), length=0.1)
            # plt.title('Random rotations of unit vector in z axis')
            # plt.show()
            # exit()
            ###

            # Compute collisions with detector:
            did_impact, lambda_1, lambda_2 = detector.impact(lor_normal=n,
                                                             lor_annihilation=self.get_position_cartesian())

            if not did_impact:
                continue

            # Note lambda_1 < 0 < lambda_2 and they represent distance in cm as n is a unit vector

            ### DEBUG Plot LOR (without scattering)
            if debug_ax is not None and did_impact:
                debug_line_3d(debug_ax, self.get_position_cartesian(), n, lambda_1, lambda_2, 50, color='grey')

                impact_1, impact_2 = self.get_position_cartesian() + lambda_1 * n, \
                                     self.get_position_cartesian() + lambda_2 * n

                debug_point_3d(debug_ax, impact_1, color='grey', label='Initial Trajectory')
                debug_point_3d(debug_ax, impact_2, color='grey')
            ###

            # Compton Scattering
            final_impacts = []
            for l in [lambda_1, lambda_2]:
                distance = np.abs(l)

                # Model parameter here! Exponential distribution representing chance of collision before detector impact
                first_scatter = np.random.exponential(scale=1.0/self.scatter_rate)

                if first_scatter < distance:  # Compton scatter occurred
                    n_scatters += 1

                    scatter_point = self.get_position_cartesian() + (np.sign(l)*first_scatter)*n
                    print('Scatter at', scatter_point)

                    new_n = self._generate_scatter_rotation().dot(n)

                    if debug_ax is not None:
                        debug_point_3d(debug_ax, scatter_point, color='darkred', label='Compton Scatter')
                        debug_arrow_3d(debug_ax, scatter_point, new_n, length=0.05, color='orange')

                    # Debug line
                    did_impact_scattered, _, scatter_lambda = detector.impact(lor_normal=new_n,
                                                                              lor_annihilation=scatter_point)

                    if did_impact_scattered:
                        print('--> Scattering DID impact')
                        if debug_ax is not None:
                            debug_line_3d(debug_ax, scatter_point, new_n, 0, scatter_lambda, 10, color='green')
                            debug_point_3d(debug_ax, scatter_point + scatter_lambda*new_n, color='green',
                                           s=5, label='New impact')

                        final_impacts.append(scatter_point + scatter_lambda*new_n)
                        continue
                    else:
                        print('--> Scattering did NOT impact')
                        if debug_ax is not None:
                            debug_line_3d(debug_ax, scatter_point, new_n, 0, 1, 10, color='red')

                # No compton scattering
                final_impacts.append(self.get_position_cartesian() + l * n)

            if debug_ax is not None:
                debug_line_3d(debug_ax, final_impacts[0], -final_impacts[0]+final_impacts[1], 0, 1, 10, color='black',
                              label='LOR')

            impacts += [final_impacts]

        return impacts, n_scatters


if __name__ == '__main__':
    particle = StaticParticle()
    particle.set_position_cartesian(0, 0, 0.2)

    detector = CylinderDetector()

    while True:
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111, projection='3d')
        detector.debug_plot(ax=ax)

        particle.simulate_emissions(detector=detector, debug_ax=ax)

        ax.axes.set_xlim3d(left=-detector.dim_radius_cm, right=detector.dim_radius_cm)
        ax.axes.set_ylim3d(bottom=-detector.dim_radius_cm, top=detector.dim_radius_cm)
        ax.axes.set_zlim3d(bottom=-0.1, top=detector.dim_height_cm+0.1)

        plt.title('Static Radioactive Particle in Cylinder Detector')
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    lors, n_impacts = particle.simulate_emissions(detector=detector, n_lor=1000)

    print(len(lors), n_impacts)
