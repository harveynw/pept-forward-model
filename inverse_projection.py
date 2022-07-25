import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from geometry import MultiQuadrilateral
from integration import projection_region
from model import CylinderDetector, StaticParticle
from plot import detector_plot


def atan2(x1, x2):
    # Returns argument of the complex number x2+x1*i in the range [0, 2pi]
    angle = np.arctan2(x1, x2)
    return angle if angle > 0.0 else 2*np.pi + angle


def phi_2(R: float, X: np.ndarray, phi_1: float):
    a_1 = R * np.cos(phi_1)
    a_2 = R * np.sin(phi_1)

    b_1 = X[0] - a_1
    b_2 = X[1] - a_2

    sols = np.roots([
        b_1 ** 2 + b_2 ** 2,
        2 * (a_1 * b_1 + a_2 * b_2),
        a_1 ** 2 + a_2 ** 2 - R ** 2
    ])
    sols.sort()
    mu = sols[1]  # Has to be the positive root
    return atan2(a_2 + mu * b_2, a_1 + mu * b_1)


def z_2(R: float, X: np.ndarray, phi_1: float, z_1: float):
    a_1 = R * np.cos(phi_1)
    a_2 = R * np.sin(phi_1)

    b_1 = X[0] - a_1
    b_2 = X[1] - a_2

    sols = np.roots([
        b_1 ** 2 + b_2 ** 2,
        2 * (a_1 * b_1 + a_2 * b_2),
        a_1 ** 2 + a_2 ** 2 - R ** 2
    ])
    sols.sort()
    mu = sols[1]  # Has to be the positive root
    return z_1 + mu * (X[2] - z_1)


d = CylinderDetector()
R = d.dim_radius_cm


# detector_d_phi = 5*2*np.pi * 1/(2*np.pi*d.dim_radius_cm/d.detectors_width)
# detector_d_z = 5*d.detectors_height
# phi_1_range = (np.pi/6, np.pi/6+detector_d_phi)
# z_1_range = (detector_d_z * 3, detector_d_z * 4)

n_x, n_y = d.n_detector_cells()
region = d.detector_cell_from_index(int(n_x/2 + n_y/2 * n_x))

phi_1_range = region.x_range()
z_1_range = region.y_range()

particle = StaticParticle()
p_r = st.slider('R', min_value=0.0, max_value=d.dim_radius_cm, value=0.21)
p_theta = st.slider('theta', min_value=0.0, max_value=2*np.pi, value=0.32)
p_z = st.slider('z', min_value=0.0, max_value=d.dim_height_cm, value=0.11)
samples = st.number_input('samples', min_value=1, max_value=10**9, value=10000)

particle.set_position_cylindrical(r=p_r, theta=p_theta, z=p_z)
p = particle.get_position_cartesian()

# Monte Carlo Plot

hit_samples = []
proj_samples = []
for _ in range(samples):
    phi_1_sample = np.random.uniform(phi_1_range[0], phi_1_range[1])
    z_1_sample = np.random.uniform(z_1_range[0], z_1_range[1])

    hit_samples += [(phi_1_sample, z_1_sample)]

    phi_2_sample = phi_2(R, p, phi_1_sample)
    z_2_sample = z_2(R, p, phi_1_sample, z_1_sample)

    if 0.0 < z_2_sample < d.dim_height_cm:
        proj_samples += [(phi_2_sample, z_2_sample)]

x_plot, y_plot = zip(*(hit_samples+proj_samples))

fig, ax = plt.subplots()
ax.scatter(x_plot, y_plot, marker='o', s=(72./fig.dpi)**2)
ax.set_title('Monte Carlo Back-Projection')
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$z$")
ax.set_xlim((0, 2*np.pi))
ax.set_ylim((0, d.dim_height_cm))

st.pyplot(fig)

# Exact Plot

i_proj_region = projection_region(R=R, x=particle.get_position_cartesian(), detector_phi=phi_1_range, detector_z=z_1_range)

fig, ax = detector_plot(d.dim_height_cm)

ax.set_title("Quadrilateral Approximation")
region.plot(ax, 'g')
i_proj_region.plot(ax, 'r')

st.pyplot(fig)
