import numpy as np
import matplotlib.pyplot as plt

from typing import Union


def line_3d(ax: plt.axis, start_point: Union[np.ndarray, list], normal: Union[np.ndarray, list],
            lambda_start: float, lambda_end: float,
            n_steps: int, **kwargs):
    trace_param = np.linspace(start=lambda_start, stop=lambda_end, num=n_steps)
    trace = np.array([start_point + l * normal for l in trace_param])
    ax.plot(xs=trace[:, 0], ys=trace[:, 1], zs=trace[:, 2], **kwargs)


def point_3d(ax: plt.axis, point: Union[np.ndarray, list], **kwargs):
    ax.scatter([point[0]], [point[1]], [point[2]], **kwargs)


def points_3d(ax: plt.axis, points: np.ndarray, **kwargs):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], **kwargs)


def arrow_3d(ax: plt.axis, origin: Union[np.ndarray, list], dir: Union[np.ndarray, list], **kwargs):
    u, v, w = [dir[0]], [dir[1]], [dir[2]]
    ax.quiver([origin[0]], [origin[1]], [origin[2]], u, v, w, **kwargs)


def detector_plot(detector_height: float) -> (plt.Figure, plt.axis):
    fig, ax = plt.subplots()
    ax.set_xlim([0, 2*np.pi])
    ax.set_ylim([0, detector_height])

    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$z$")

    return fig, ax