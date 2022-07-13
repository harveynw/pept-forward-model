import numpy as np


# Base class for representing a point in space, supports some useful representations
class Point:
    # Store position in cartesian coordinates
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def set_position_cylindrical(self, r: float, theta: float, z: float):
        # Sets the position of the particle according to the cylindrical coordinate system
        self.x = r*np.cos(theta)
        self.y = r*np.sin(theta)
        self.z = z

    def set_position_cartesian(self, x: float, y: float, z: float):
        # Sets the position of the particle according to the cartesian coordinate system
        self.x, self.y, self.z = x, y, z

    def set_position_spherical(self, r: float, theta: float, phi: float):
        # Sets the position of the particle according to the spherical coordinate system (r, θ, φ)
        self.x = r*np.cos(phi)*np.sin(theta)
        self.y = r*np.sin(phi)*np.sin(theta)
        self.z = r*np.cos(theta)

    def get_position_cartesian(self):
        return np.array([self.x, self.y, self.z])

    def __str__(self):
        return f'Point({self.x}, {self.y}, {self.z})'


def azimuth_of_point(x: float, y: float):
    # Returns the azimuthal angle of a 2D cartesian point in the range [0, 2π]
    if np.isclose(x, 0.0):
        phi = np.pi / 2.0 if y > 0.0 else -np.pi / 2.0
    elif x > 0.0:
        phi = np.arctan(y / x)
    else:
        phi = np.arctan(y / x) + (np.pi if y >= 0.0 else -np.pi)
    return np.mod(phi, 2 * np.pi)
