import jax
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from geometry import atan2
from inversion.equations import G_phi, G_z
from model import CylinderDetector, StaticParticle


def phi_proj(R: float, X: np.ndarray, phi: float) -> float:
    s, c = np.sin(phi), np.cos(phi)
    c_1, c_2 = X[0] - R*c, X[1] - R*s
    omega = -2*R*(c * c_1 + s * c_2)/(np.square(c_1) + np.square(c_2))

    return atan2(R*s + omega*c_2, R*c + omega*c_1)


def z_proj(R: float, X: np.ndarray, phi: float, z: float) -> float:
    s, c = np.sin(phi), np.cos(phi)
    c_1, c_2 = X[0] - R*c, X[1] - R*s
    omega = -2*R*(c * c_1 + s * c_2)/(np.square(c_1) + np.square(c_2))

    return z + omega*(X[2]-z)


d = CylinderDetector()
R, H = d.dim_radius_cm, d.dim_height_cm


st.title('Projecting detector cell through particle')

st.subheader('Detector cell size')

d.detectors_width = st.slider('Width', min_value=0.005, max_value=R, step=0.005, value=0.14)
d.detectors_height = st.slider('Height', min_value=0.005, max_value=H/4.0, step=0.005, value=0.07)

n_x, n_y = d.n_detector_cells()

st.subheader('Source detector cell')

idx_phi = st.slider('Horizontal Coordinate', min_value=0, max_value=n_x-1, value=6, step=1)
idx_z = st.slider('Vertical Coordinate', min_value=0, max_value=n_y-1, value=4, step=1)

region = d.detector_cell_from_index(int(idx_phi + idx_z * n_x))
phi_1_range = region.x_range()
z_1_range = region.y_range()

st.markdown("""---""")

st.subheader('Particle Position')

particle = StaticParticle()
p_r = st.slider('R', min_value=0.0, max_value=R, value=0.05)
p_theta = st.slider('theta', min_value=0.0, max_value=2*np.pi, value=1.57)
p_z = st.slider('z', min_value=-H/2, max_value=H/2, value=-0.05)
samples = st.number_input('samples', min_value=1, max_value=10**9, value=5000)

particle.set_position_cylindrical(r=p_r, theta=p_theta, z=p_z)
x = particle.get_position_cartesian()
X = jax.numpy.array(x)

# Monte Carlo Plot

hit_samples = []
proj_samples = []

g_phi, g_z = jax.vmap(G_phi, (None, 0, None), 0), jax.vmap(G_z, (None, 0, 0, None), 0)

phi_1_samples = np.random.uniform(phi_1_range[0], phi_1_range[1], size=(samples,))
z_1_samples = np.random.uniform(z_1_range[0], z_1_range[1], size=(samples,))

phi_2_samples = g_phi(R, phi_1_samples, X)
z_2_samples = g_z(R, phi_1_samples, z_1_samples, X)

fig, ax = plt.subplots()
ax.scatter(phi_1_samples, z_1_samples, marker='o', color='r', label=r'$\mathcal{D}_i$', s=(72./fig.dpi)**2)
ax.scatter(phi_2_samples, z_2_samples, marker='o', color='b', label=r'$\mathcal{D}_j$', s=(72./fig.dpi)**2)

ax.set_title(r'Monte Carlo Projection through $G_{\phi}, G_z$' + f' (n={samples}).')
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$z$")
ax.set_xlim((0, 2*np.pi))
ax.set_ylim((-H/2, H/2))
plt.legend()

st.pyplot(fig)
