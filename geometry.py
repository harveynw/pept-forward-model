import numpy as np


# Base class for representing a point in space, supports some useful representations
class Point:
    # Store position in spherical coordinates (r, θ, φ)
    r: float = 0.0
    theta: float = 0.0
    phi: float = 0.0

    def set_position_cylindrical(self, r: float, theta: float, z: float):
        # Sets the position of the particle according to the cylindrical coordinate system
        self.r = np.sqrt(np.square(r) + np.square(z))
        self.phi = theta
        if np.sign(r) == -1:
            self.phi += np.pi
            r *= -1
        self.phi = np.mod(self.phi, 2*np.pi)

        if np.isclose(z, 0.0):
            self.theta = np.pi/2
        elif np.isclose(r, 0.0):
            self.theta = np.pi if z < 0.0 else 0.0
        else:
            self.theta = np.arctan(r/z)
            if z < 0.0:
                self.theta += np.pi

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


def azimuth_of_point(x: float, y: float):
    # Returns the azimuthal angle of a 2D cartesian point in the range [0, 2π]
    if np.isclose(x, 0.0):
        phi = np.pi / 2.0 if y > 0.0 else -np.pi / 2.0
    elif x > 0.0:
        phi = np.arctan(y / x)
    else:
        phi = np.arctan(y / x) + (np.pi if y >= 0.0 else -np.pi)
    return np.mod(phi, 2 * np.pi)
