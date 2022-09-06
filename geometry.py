import numpy as np
import shapely.geometry as sg

from dataclasses import dataclass
from typing import Union, List
from matplotlib.patches import Polygon
from numpy.linalg import LinAlgError

# In this file, defines the Point base class and some miscellaneous geometry/computer graphics algorithms that
# were not used in the end for the inversion algorithm.


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
        theta = atan2(self.x, self.y) / np.pi
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
        v = self.vertices().tolist()
        poly = Polygon(v + [v[0]], facecolor=colour)
        ax.add_patch(poly)

    def inside(self, p: Coordinate) -> bool:
        return inside_quad(p, self.vertices())

    def intersects(self, quad) -> bool:
        p1 = sg.Polygon(self.vertices())
        p2 = sg.Polygon(quad.vertices())

        return p2.intersects(p1)

    def edges(self):
        v = self.vertices()
        return [(v[i], v[(i+1) % len(v)]) for i in range(len(v))]

    def max(self):
        # Max X, Max Y
        v = self.vertices()
        return np.max(v[:, 0]), np.max(v[:, 1])

    def min(self):
        # Min X, Max Y
        v = self.vertices()
        return np.min(v[:, 0]), np.min(v[:, 1])

    def x_range(self):
        v = self.vertices()
        return np.min(v[:, 0]), np.max(v[:, 0])

    def y_range(self):
        v = self.vertices()
        return np.min(v[:, 1]), np.max(v[:, 1])

    def vertices(self) -> np.ndarray:
        return np.array([np.array(self.v1), np.array(self.v2),
                         np.array(self.v3), np.array(self.v4)])


class RectangleQuadrilateral(Quadrilateral):
    def __init__(self, min_point: Coordinate, max_point: Coordinate):
        super().__init__(
            [min_point[0], min_point[1]],
            [min_point[0], max_point[1]],
            [max_point[0], max_point[1]],
            [max_point[0], min_point[1]]
        )


def atan2(x: float, y: float):
    """ Numpy ARCTAN2

    https://en.wikipedia.org/wiki/Atan2 to range [0, 2π]

    Args:
        x: point x coordinate
        y: point y coordinate

    Returns:
        Azimuthal angle in [0, 2π]
    """
    angle = np.arctan2(y, x)
    return angle if angle > 0.0 else 2 * np.pi + angle


def barycentric_coords_from_triangle(p: np.ndarray, verts: list):
    """ Barycentric coordinates of point

    Finds the barycentric coordinates of a point relative to a triangle.

    Args:
        p: The point
        verts: List of numpy 2D vertices of the triangle

    Returns:
        2D barycentric coordinates.
    """
    p1, p2, p3 = verts

    T = np.array([[p1[0]-p3[0], p2[0]-p3[0]],
                  [p1[1]-p3[1], p2[1]-p3[1]]])

    l_1, l_2 = np.linalg.inv(T).dot(p-p3)
    l_3 = 1 - l_1 - l_2

    return np.array([l_1, l_2, l_3])


def line_segments_intersection(p1: Coordinate, n1: Coordinate, p2: Coordinate, n2: Coordinate):
    """ Intersection point of 2D line segments

    Finds the intersection point of the two line segments p1 + λ*n1, p2 + µ*n2
     assuming λ,µ ∈ [0,1].

    Args:
        p1: Line segment 1 start
        n1: Line segment 1 direction
        p2: Line segment 2 start
        n2: Line segment 2 direction

    Returns:
        2D coordinate of intersection, or None if not found.
    """
    try:
        coeff = np.linalg.solve(np.array([n1, n2]).transpose(), np.array(p2) - np.array(p1))
    except LinAlgError:
        return None  # Parallel

    lam, mu = coeff[0], -coeff[1]

    if 0.0 <= lam <= 1.0 and 0.0 <= mu <= 1.0:
        return np.array(p1) * lam*np.array(n1)
    else:
        return None  # Not on both line segments


def rect_quad_intersection_area(rect: RectangleQuadrilateral, quad: Union[Quadrilateral, MultiQuadrilateral]) -> float:
    """Intersection area between a Rectangle and Quadrilateral(s)

    Finds the intersection polygon between rect and quad, and then computes its area

    Args:
        rect: The rectangle
        quad: The quadrilateral(s)

    Returns:
        area, float
    """
    if isinstance(quad, MultiQuadrilateral):
        return sum([rect_quad_intersection_area(rect, q) for q in quad.quads])

    interior_poly: List[Coordinate] = []

    # First test if any vertices of rect are interior points of quad
    for v in rect.vertices():
        if quad.inside(v):
            interior_poly.append(v)

    # And vice-versa
    for v in quad.vertices():
        if rect.inside(v):
            interior_poly.append(v)

    # Next find intersection points of quad edges with rect
    for rect_edge in rect.edges():
        a, b = rect_edge
        for quad_edge in quad.edges():
            c, d = quad_edge
            intersect = line_segments_intersection(a, b-a, c, d-c)
            if intersect is not None:
                interior_poly.append(intersect)

    # Form intersection polygon
    if len(interior_poly) == 0:
        return 0.0
    interior_poly = points_to_convex_polygon(interior_poly)

    # Compute area using triangles
    triangles = []
    for i in range(1, len(interior_poly)-1):
        triangles.append([interior_poly[0], interior_poly[i], interior_poly[i+1]])

    return sum([area_of_triangle(*tri) for tri in triangles])


def area_of_triangle(v1: Coordinate, v2: Coordinate, v3: Coordinate):
    """Area of a triangle

    Computes the area of a triangle from its vertices, using the shoelace formula
    """
    return 0.5 * ((v1[0]-v3[0])*(v2[1]-v1[1]) - (v1[0]-v2[0])*(v3[1]-v1[1]))


def points_to_convex_polygon(points: List[Coordinate]) -> List[Coordinate]:
    """Collection of points to an ordered list of vertices for a convex polygon

    Sorts a list of 2D points into an ordered list of convex polygon vertices.
    This works in principle by comparing the complex argument of each point.

    Args:
        points: List of 2D points

    Returns:
        list of points corresponding to a convex polygon
    """
    # Ensure numpy array
    points = np.array(points)

    # Centroid is always an interior point
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # Sort by argument
    sorted_idx = np.argsort([atan2(p[0], p[1]) for p in centered_points])

    return points[sorted_idx, :]


def inside_quad(x: Coordinate, quad_points: np.ndarray) -> bool:
    """Point inside quadrilateral test

    Tests if a 2D point lies within a quadrilateral

    Args:
        x: 2D point
        quad_points: numpy array containing vertices of the quad

    Returns:
        bool result of the test
    """
    p1, p2, p3, p4 = quad_points

    b_1 = barycentric_coords_from_triangle(p=np.array(x), verts=[p1, p2, p3])
    b_2 = barycentric_coords_from_triangle(p=np.array(x), verts=[p3, p4, p1])

    return np.all(((b_1 >= 0) & (b_1 <= 1)) | ((b_2 >= 0) & (b_2 <= 1)))


def random_point_within_cylinder(radius: float, height: float):
    """Random point within a cylinder

    Generates a random cartesian point (uniformly) in a cylinder, which is
    assumed to have its center at the origin.

    Args:
        radius: radius of cylinder
        height: height of cylinder

    Returns:
        shape (3,) numpy array of the sampled point
    """

    z = np.random.uniform(-height/2.0, height/2.0)
    while True:
        x, y = np.random.uniform(-radius, radius), np.random.uniform(-radius, radius)
        if np.sqrt(x**2 + y**2) < radius:
            return np.array([x, y, z])
