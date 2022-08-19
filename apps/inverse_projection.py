import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from geometry import phi_proj, z_proj
from misc.integration import projection_region
from model import CylinderDetector, StaticParticle
from plot import detector_plot

d = CylinderDetector()
R = d.dim_radius_cm

n_x, n_y = d.n_detector_cells()
region = d.detector_cell_from_index(int(n_x/4 + n_y/2 * n_x))

phi_1_range = region.x_range()
z_1_range = region.y_range()

st.title('Projecting detector surface through particle')

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

    phi_2_sample = phi_proj(R, p, phi_1_sample)
    z_2_sample = z_proj(R, p, phi_1_sample, z_1_sample)

    if 0.0 < z_2_sample < d.dim_height_cm:
        proj_samples += [(phi_2_sample, z_2_sample)]

fig, ax = plt.subplots()

if hit_samples:
    x_plot, y_plot = zip(*hit_samples)
    ax.scatter(x_plot, y_plot, marker='o', color='g', s=(72./fig.dpi)**2)
if proj_samples:
    x_plot, y_plot = zip(*proj_samples)
    ax.scatter(x_plot, y_plot, marker='o', s=(72./fig.dpi)**2)

ax.set_title(f'Monte Carlo Projection (n={samples})')
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
