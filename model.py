import abc
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from plot import point_3d, line_3d, arrow_3d
from geometry import atan2, Point, RectangleQuadrilateral


class Detector:
    """Abstract detector class

    This was done in case a detector of shape different to a Cylinder was ever implemented.
    """

    @abc.abstractmethod
    def impact(self, lor_normal: np.ndarray, lor_annihilation: np.ndarray) -> (bool, float, float):
        pass

    @abc.abstractmethod
    def impact_forward_only(self, lor_normal: np.ndarray, lor_annihilation: np.ndarray) -> (bool, float):
        pass

    @abc.abstractmethod
    def detector_index_from_impact(self, impact: np.ndarray) -> (int, int):
        pass


@dataclass
class CylinderDetector(Detector):
    """ CylinderDetector

    Object for storing the detector dimensions and individual cell information.
    Contains methods for mapping to the cell indices and for plotting.
    """

    dim_radius_cm: float = 0.25
    dim_height_cm: float = 0.50

    detectors_height: float = 0.005
    detectors_width: float = 0.005

    def _is_z_coordinate_in_detector(self, z):
        return -self.dim_height_cm / 2.0 <= z <= self.dim_height_cm / 2.0

    def impact(self, lor_normal: np.ndarray, lor_annihilation: np.ndarray):
        """ Impact detection

        Determines whether a line collides with the detector twice

        Args:
            lor_normal: The normal specifying the line
            lor_annihilation: A point on the line
        Returns:
            bool: True/false line collided twice
            float: Line parameter of first collision
            float: Line parameter of second collision
        """
        p_x, p_y, p_z = lor_annihilation
        n_1, n_2, n_3 = lor_normal

        # Find intersection
        a = np.square(n_1) + np.square(n_2)
        b = 2 * n_1 * p_x + 2 * p_y * n_2
        c = np.square(p_x) + np.square(p_y) - np.square(self.dim_radius_cm)

        lambda_1, lambda_2 = np.roots([a, b, c])
        if lambda_2 < lambda_1:
            lambda_1, lambda_2 = lambda_2, lambda_1

        impact_1, impact_2 = lor_annihilation + lambda_1 * lor_normal, lor_annihilation + lambda_2 * lor_normal

        did_impact = self._is_z_coordinate_in_detector(impact_1[2]) and self._is_z_coordinate_in_detector(impact_2[2])

        return did_impact, lambda_1, lambda_2

    def impact_forward_only(self, lor_normal: np.ndarray, lor_annihilation: np.ndarray) -> (bool, float):
        """ Impact detection (single case)

        Determines whether a line collides with the detector once,
         travelling in the direction of the normal

        Args:
            lor_normal: The normal specifying the direction of the line
            lor_annihilation: A point on the line
        Returns:
            bool: True/false line collided with detector once
            float: Line parameter of collision
        """
        _, _, l = self.impact(lor_normal=lor_normal, lor_annihilation=lor_annihilation)

        impact = lor_annihilation + l * lor_normal
        return self._is_z_coordinate_in_detector(impact[2]), l

    def n_detector_cells(self) -> (int, int):
        """ Number of detector cells

        Returns how the detector surface is discretised.

        Returns:
            n_x: Number of detector cells horizontally in grid
            n_y: Number of detector cells vertically in grid
        """
        n_horizontal = np.rint(2 * np.pi * self.dim_radius_cm / self.detectors_width)
        n_vertical = np.rint(self.dim_height_cm / self.detectors_height)
        return int(n_horizontal), int(n_vertical)

    def del_detector_cells(self) -> (float, float):
        """ Dimensions of detector cells

        Returns the size of each detector cell in cylindrical coordinate space.

        Returns:
            d_phi: azimuthal angle across each cell in radians from the origin
            d_z: z-coordinate height of each cell
        """
        n_x, n_y = self.n_detector_cells()
        return 2 * np.pi / n_x, self.dim_height_cm / n_y

    def detector_index_from_impact(self, impact: np.ndarray) -> (int, int):
        """ Detector cell index from collision

        Returns the horizontal, vertical index of the detector cell in which
         a cartesian collision point lies.

        Returns:
            phi_index: Horizontal index over detector surface 0 <= phi_index < n_x
            z_index: Vertical index over detector surface 0 <= z_index < n_y
        """
        x, y, z = impact
        phi = atan2(y, x)

        # Ensure impact is on detector
        assert np.isclose(self.dim_radius_cm, np.sqrt(x ** 2 + y ** 2)), 'Impact not on detector'
        assert -self.dim_height_cm / 2.0 <= z <= self.dim_height_cm / 2.0

        n_phi, _ = self.n_detector_cells()
        d_phi, d_z = self.del_detector_cells()

        phi_index, z_index = int(phi // d_phi), int((z + self.dim_height_cm / 2.0) // d_z)

        return phi_index + n_phi * z_index

    def detector_cell_from_index(self, i: int):
        """ Detector cell in space from its index

        Takes a detector cell index and returns its surface in cylindrical coordinate space.

        Args:
            i: The cell index 0 <= i < n_x * n_y
        Returns:
            RectangleQuadrilateral: The surface specified in (phi, z) coordinates on
                the entire detector surface [0, 2π] x [-H/2, H/2]
        """
        n_x, n_y = self.n_detector_cells()
        assert 0 <= i < n_x * n_y

        y, x = divmod(i, n_x)
        d_x, d_y = 2 * np.pi / n_x, self.dim_height_cm / n_y

        return RectangleQuadrilateral(
            [x * d_x, -self.dim_height_cm / 2.0 + y * d_y],
            [(x + 1) * d_x, -self.dim_height_cm / 2.0 + (y + 1) * d_y]
        )

    def detector_cells_from_region(self, phi_range: tuple, z_range: tuple):
        """ Detector cells from a rectangular region

        Takes a 2D rectangle specified on the detector surface and returns all
        the cell indices that touch it.

        Args:
            phi_range: Tuple of the min, max phi values of the rect in [0, 2π]
            z_range: Tuple of the min, max z values of the rect in [-H/2, H/2]
        Returns:
            cells: List of cell indices
        """
        phi_min, phi_max = phi_range
        z_min, z_max = z_range

        n_x, n_y = self.n_detector_cells()
        d_phi, d_z = self.del_detector_cells()

        # Range of values by index along dimension
        phi_coords = (np.floor(phi_min / d_phi).astype(int), np.ceil(phi_max / d_phi).astype(int))
        z_coords = (np.floor(z_min / d_z).astype(int), np.ceil(z_max / d_z).astype(int))

        # Ensure on detector
        phi_coords = np.clip(np.array(phi_coords), 0, n_x - 1)
        z_coords = np.clip(np.array(z_coords), 0, n_y - 1)

        cells = []
        for i in range(phi_coords[0], phi_coords[1] + 1):
            for j in range(z_coords[0], z_coords[1] + 1):
                cells += [i + n_x * j]

        return cells

    def debug_plot(self, ax: plt.axis):
        """ Plot cylinder

        Takes a matplotlib axis and plots the cylindrical detector surface

        Args:
            ax: plt.Axis
        """
        diameter = 2 * np.pi * self.dim_radius_cm
        n_detectors_horizontal = int(diameter / self.detectors_width)
        n_detectors_vertical = int(self.dim_height_cm / self.detectors_height)

        theta_grid, z_grid = np.meshgrid(np.linspace(0, 2 * np.pi, n_detectors_horizontal),
                                         np.linspace(-self.dim_height_cm/2.0, self.dim_height_cm/2.0,
                                                     n_detectors_vertical))
        x_grid = self.dim_radius_cm * np.cos(theta_grid)  # + center_x
        y_grid = self.dim_radius_cm * np.sin(theta_grid)  # + center_y

        ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.2)


@dataclass
class StaticParticle(Point):
    # Compton Scattering Rate
    scatter_rate: float = 2.0

    # Scattering angle distribution
    kappa: float = 5.0

    def generate_scatter_rotation(self) -> np.ndarray:
        """ Sample a scatter transformation

        This simulates a change in trajectory of a photon resulting from
        Compton Scattering (Section 2.2.1), kappa is hardcoded here.

        Returns:
            np.ndarray: 3D Rotation Matrix
        """
        phi = np.random.uniform(low=0, high=2 * np.pi)
        theta = np.random.vonmises(mu=0, kappa=self.kappa)

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

    def simulate_emissions(self, detector: Detector, n_emissions=0.05 * (10 ** 4), debug_ax=None):
        """ Simulates emissions given a detector

        Simulates a number of emissions, this is Algorithm 1
         in the report repeated n_emissions times.

        Args:
            detector: The (cylindrical) detector object
            n_emissions: Number of emissions to simulate
            debug_ax (optional): A matplotlib axis to plot each emission

        Returns:
            impacts: Returns a list of LoRs, List[(detector cell index i, detector cell index j)]
            n_scatters: Count of the number of photons scattered over all the detected LoRs.
        """
        if debug_ax is not None:
            point_3d(debug_ax, self.get_position_cartesian(), color='r', label='Particle Position')

        impacts = []
        n_scatters = 0

        for _ in range(n_emissions):  # For each requested LOR
            varphi = np.random.uniform(0, 2 * np.pi)
            theta = np.arccos(1 - np.random.uniform(0, 1))  # Inverse Transform Sampling

            n = np.array([
                np.sin(theta)*np.cos(varphi),
                np.sin(theta)*np.sin(varphi),
                np.cos(theta)
            ])
            n_varphi = np.array([
                -np.sin(varphi),
                np.cos(varphi),
                0.0
            ])
            n_theta = np.array([
                -np.cos(theta)*np.cos(varphi),
                -np.cos(theta)*np.sin(varphi),
                np.sin(theta)
            ])

            # Compute collisions with detector:
            did_impact, lambda_1, lambda_2 = detector.impact(lor_normal=n,
                                                             lor_annihilation=self.get_position_cartesian())

            if not did_impact:
                continue

            if debug_ax is not None and did_impact:
                line_3d(debug_ax, self.get_position_cartesian(), n, lambda_1, lambda_2, 50,
                        color='grey', label='Initial LOR')

                impact_1, impact_2 = self.get_position_cartesian() + lambda_1 * n, \
                                     self.get_position_cartesian() + lambda_2 * n

                point_3d(debug_ax, impact_1, color='grey')
                point_3d(debug_ax, impact_2, color='grey')

            # Compton Scattering
            final_impacts = []
            final_scatter_count = 0
            was_detected = True

            for l in [lambda_1, lambda_2]:
                distance = np.abs(l)

                # Model parameter here! Exponential distribution representing chance of collision before detector impact
                first_scatter = np.random.exponential(scale=1.0 / self.scatter_rate)

                if first_scatter < distance:  # Compton scatter occurred
                    final_scatter_count += 1

                    scatter_point = self.get_position_cartesian() + (np.sign(l) * first_scatter) * n

                    # Compute new normal (with a random rotation)
                    change_of_basis = np.array([n_varphi, n_theta, np.sign(l) * n]).transpose()
                    new_n = change_of_basis.dot(self.generate_scatter_rotation().dot([0, 0, 1]))

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
                            line_3d(debug_ax, scatter_point, new_n, 0, scatter_lambda, 10, color='orange')
                            point_3d(debug_ax, scatter_point + scatter_lambda * new_n, color='green',
                                     s=5, label='New impact')
                        else:
                            line_3d(debug_ax, scatter_point, new_n, 0, 1, 10, color='red')

                    # Final impact after scattering
                    final_impacts.append(scatter_point + scatter_lambda * new_n)
                else:
                    # Original trajectory
                    final_impacts.append(self.get_position_cartesian() + l * n)

            if was_detected:
                detector_i = detector.detector_index_from_impact(final_impacts[0])
                detector_j = detector.detector_index_from_impact(final_impacts[1])

                # detector_i is has higher z range by convention
                if final_impacts[1][2] > final_impacts[0][2]:
                    detector_i, detector_j = detector_j, detector_i

                # Return detector indices
                impacts += [(detector_i, detector_j)]

                # Return cartesian coordinate impacts
                # impacts += [final_impacts]

                n_scatters += final_scatter_count

                if debug_ax is not None:
                    line_3d(debug_ax, final_impacts[0], -final_impacts[0] + final_impacts[1], 0, 1, 10, color='black',
                            label='LOR')

        return impacts, n_scatters
