import numpy as np
import shapely.geometry as sg

from dataclasses import dataclass
from typing import Union, List
from matplotlib.patches import Polygon

Coordinate = Union[np.ndarray, tuple, list]


# Base class for representing a point in space, supports some useful representations
class Point:
    # Store position in cartesian coordinates
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def set_position_cylindrical(self, r: float, theta: float, z: float):
        # Sets the position of the particle according to the cylindrical coordinate system
        self.x = r * np.cos(theta)
        self.y = r * np.sin(theta)
        self.z = z

    def set_position_cartesian(self, x: float, y: float, z: float):
        # Sets the position of the particle according to the cartesian coordinate system
        self.x, self.y, self.z = x, y, z

    def set_position_spherical(self, r: float, theta: float, phi: float):
        # Sets the position of the particle according to the spherical coordinate system (r, θ, φ)
        self.x = r * np.cos(phi) * np.sin(theta)
        self.y = r * np.sin(phi) * np.sin(theta)
        self.z = r * np.cos(theta)

    def get_position_cartesian(self):
        return np.array([self.x, self.y, self.z])

    def __str__(self):
        return self.to_str_cartesian()

    def to_str_cartesian(self):
        return f'Point({self.x}, {self.y}, {self.z})'

    def to_str_cylindrical(self, latex=False):
        rho = np.linalg.norm([self.x, self.y])
        theta = azimuth_of_point(self.x, self.y)/np.pi
        if latex:
            return fr'Point($\rho={rho:.2f}$, $\theta={theta:.2f}\pi$, $z={self.z:.2f}$)'
        else:
            return f'Point(r={rho:.2f}, theta={theta:.2f}pi, z={self.z:.2f})'


@dataclass
class MultiQuadrilateral:
    quads: List

    def plot(self, ax, colour='r'):
        for q in self.quads:
            v = q.vertices()
            poly = Polygon(v + [v[0]], facecolor=colour)
            ax.add_patch(poly)

    def inside(self, p: Coordinate) -> bool:
        for q in self.quads:
            if not inside_quad(p, q.vertices()):
                return False
        return True

    def intersects(self, quad) -> bool:
        for q in self.quads:
            p1 = sg.Polygon(q.vertices())
            p2 = sg.Polygon(quad.vertices())

            if not p2.intersects(p1):
                return False

        return True


@dataclass
class Quadrilateral:
    v1: Coordinate
    v2: Coordinate
    v3: Coordinate
    v4: Coordinate

    @classmethod
    def from_range(cls, x_range: Coordinate, y_range: Coordinate):
        return cls([x_range[0], y_range[0]],
                   [x_range[0], y_range[1]],
                   [x_range[1], y_range[1]],
                   [x_range[1], y_range[0]])

    def plot(self, ax, colour='r'):
        v = self.vertices()
        poly = Polygon(v + [v[0]], facecolor=colour)
        ax.add_patch(poly)

    def inside(self, p: Coordinate) -> bool:
        return inside_quad(p, self.vertices())

    def intersects(self, quad) -> bool:
        p1 = sg.Polygon(self.vertices())
        p2 = sg.Polygon(quad.vertices())

        return p2.intersects(p1)

    def max(self):
        # Max X, Max Y
        v = np.array(self.vertices())
        return np.max(v[:, 0]), np.max(v[:, 1])

    def min(self):
        # Min X, Max Y
        v = np.array(self.vertices())
        return np.min(v[:, 0]), np.min(v[:, 1])

    def x_range(self):
        v = np.array(self.vertices())
        return np.min(v[:, 0]), np.max(v[:, 0])

    def y_range(self):
        v = np.array(self.vertices())
        return np.min(v[:, 1]), np.max(v[:, 1])

    def vertices(self):
        return [self.v1, self.v2, self.v3, self.v4]


class RectangleQuadrilateral(Quadrilateral):
    def __init__(self, min_point: Coordinate, max_point: Coordinate):
        super().__init__(
            [min_point[0], min_point[1]],
            [min_point[0], max_point[1]],
            [max_point[0], max_point[1]],
            [max_point[0], min_point[1]]
        )


def azimuth_of_point(x: float, y: float):
    # Returns the azimuthal angle of a 2D cartesian point in the range [0, 2π]
    angle = np.arctan2(y, x)
    return angle if angle > 0.0 else 2 * np.pi + angle


def barycentric_coords_from_triangle(p: np.ndarray, verts: list):
    p1, p2, p3 = verts

    T = np.array([[p1[0]-p3[0], p2[0]-p3[0]],
                  [p1[1]-p3[1], p2[1]-p3[1]]])

    l_1, l_2 = np.linalg.inv(T).dot(p-p3)
    l_3 = 1 - l_1 - l_2

    return np.array([l_1, l_2, l_3])


def inside_quad(x: Coordinate, quad_points: list) -> bool:
    # x inside quadrilateral test,
    # quad_points must be clockwise or counter-clockwise
    p1, p2, p3, p4 = quad_points

    b_1 = barycentric_coords_from_triangle(p=np.array(x), verts=[p1, p2, p3])
    b_2 = barycentric_coords_from_triangle(p=np.array(x), verts=[p3, p4, p1])

    return np.all(((b_1 >= 0) & (b_1 <= 1)) | ((b_2 >= 0) & (b_2 <= 1)))


if __name__ == '__main__':
    p1, p2, p3, p4 = np.array([0.0, 0.0]), np.array([1.0, 0.0]), \
        np.array([1.0, 1.0]), np.array([0.0, 1.0])

    print(inside_quad(np.array([0.25, 0.25]), [p1, p2, p3, p4]))
    print(inside_quad(np.array([0.5, 1.001]), [p1, p2, p3, p4]))
