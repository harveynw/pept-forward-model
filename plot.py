import numpy as np
import matplotlib.pyplot as plt

from typing import Union


def line_3d(ax: plt.axis, start_point: Union[np.ndarray, list], normal: Union[np.ndarray, list],
            lambda_start: float, lambda_end: float,
            n_steps: int, **kwargs):
    """ Line 3D

    Plots a 3D line on ax
    """
    trace_param = np.linspace(start=lambda_start, stop=lambda_end, num=n_steps)
    trace = np.array([start_point + l * normal for l in trace_param])
    ax.plot(xs=trace[:, 0], ys=trace[:, 1], zs=trace[:, 2], **kwargs)


def point_3d(ax: plt.axis, point: Union[np.ndarray, list], **kwargs):
    """ Point 3D

    Plots a single point on ax
    """
    ax.scatter([point[0]], [point[1]], [point[2]], **kwargs)


def points_3d(ax: plt.axis, points: np.ndarray, **kwargs):
    """ Points 3D

    Plots 3D points on ax
    """
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], **kwargs)


def arrow_3d(ax: plt.axis, origin: Union[np.ndarray, list], dir: Union[np.ndarray, list], **kwargs):
    """ Arrow 3D

    Plots a 3D arrow on ax
    """
    u, v, w = [dir[0]], [dir[1]], [dir[2]]
    ax.quiver([origin[0]], [origin[1]], [origin[2]], u, v, w, **kwargs)


def detector_plot(detector_height: float) -> (plt.Figure, plt.axis):
    """ Detector surface plot
    
    Sets up a fig, axis for plotting across the detector surface [0, 2π] x [-H/2, H/2]

    Args:
        detector_height: H, the height of the cylindrical detector

    Returns:
        (fig, ax): Matplotlib figure and axis objects
    """
    fig, ax = plt.subplots()
    ax.set_xlim([0, 2*np.pi])
    ax.set_ylim([0, detector_height])

    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$z$")

    return fig, ax