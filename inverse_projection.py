import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from model import CylinderDetector, StaticParticle


def atan2(x1, x2):
    # Returns argument of the complex number x2+x1*i in the range [0, 2pi]
    angle = np.arctan2(x1, x2)
    return angle if angle > 0.0 else 2*np.pi + angle


def phi_2(R: float, X: np.ndarray, phi_1: float):
    x, y, _ = X
    phi = atan2(R*np.sin(phi_1)-y, R*np.cos(phi_1)-x)

    mu_2_const = x*np.cos(phi) + y*np.sin(phi)
    mu_2 = -(mu_2_const)
    mu_2 -= np.sqrt(np.square((mu_2_const)) - (np.square(x) + np.square(y) - np.square(R)))

    return atan2(y+mu_2*np.cos(phi), x + mu_2*np.sin(phi))


def z_2(R: float, X: np.ndarray, phi_1: float, z_1: float):
    x, y, z = X
    phi = atan2(R*np.sin(phi_1)-y, R*np.cos(phi_1)-x)

    l_1 = np.linalg.norm(X - np.array([R*np.cos(phi_1), R*np.sin(phi_1), z_1]))
    theta = np.arccos((z_1-z)/l_1)

    l_solved = np.roots([
        np.square(np.sin(theta)),
        2*np.sin(theta)*(x*np.cos(phi)+y*np.sin(phi)),
        np.square(x)+np.square(y)-np.square(R)
    ])
    l_solved.sort()
    l_2, l_1_check = l_solved

    return z + l_2*np.cos(theta)


d = CylinderDetector()
R = d.dim_radius_cm

detector_d_phi = 5*2*np.pi * 1/(2*np.pi*d.dim_radius_cm/d.detectors_width)
detector_d_z = 5*d.detectors_height

phi_1_range = (np.pi/6, np.pi/6+detector_d_phi)
z_1_range = (detector_d_z * 3, detector_d_z * 4)

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

# plt.hist2d(x_plot, y_plot, (5*314, 5*100), cmap=plt.cm.jet,
#            range=np.array([(0, 2*np.pi), (0, d.dim_height_cm)]))
# plt.title('Detector Hit Count')
# plt.xlabel('Horizontal')
# plt.ylabel('Vertical')
# plt.xlim((0, 2*np.pi))
# plt.ylim((0, d.dim_height_cm))
# plt.show()

fig, ax = plt.subplots()
# ax.hist2d(x_plot, y_plot, (5*314, 5*100), cmap=plt.cm.jet,
#            range=np.array([(0, 2*np.pi), (0, d.dim_height_cm)]))
ax.scatter(x_plot, y_plot, marker='o', s=(72./fig.dpi)**2)
ax.set_title('Monte Carlo Back-Projection')
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$z$")
ax.set_xlim((0, 2*np.pi))
ax.set_ylim((0, d.dim_height_cm))

st.pyplot(fig)

# Exact Plot

points = ([
    [phi_1_range[0], z_1_range[0]],
    [phi_1_range[0], z_1_range[1]],
    [phi_1_range[1], z_1_range[1]],
    [phi_1_range[1], z_1_range[0]]
])
points += [points[0]]

proj_points = [[phi_2(R, p, po[0]), z_2(R, p, po[0], po[1])] for po in points]
proj_points += [proj_points[0]]

p1 = Polygon(points, facecolor='r')
p2 = Polygon(proj_points, facecolor='b')

fig, ax = plt.subplots()

ax.set_title("Quadrilateral Approximation")

ax.add_patch(p1)
ax.add_patch(p2)

ax.set_xlim([0, 2*np.pi])
ax.set_ylim([0, d.dim_height_cm])

ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$z$")

st.pyplot(fig)
