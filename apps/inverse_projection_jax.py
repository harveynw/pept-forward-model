import numpy as np
import streamlit as st
st.set_page_config(layout="wide")

from inversion.jax_implementation import plot_proj_area
from model import CylinderDetector, StaticParticle

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

particle.set_position_cylindrical(r=p_r, theta=p_theta, z=p_z)
x, y, z = particle.get_position_cartesian()

fig, ax = plot_proj_area(min_phi=np.pi/2-0.05,
                         max_phi=np.pi/2+0.05,
                         min_z=0.39, max_z=0.49,
                         x=x, y=y, z=z)

ax.set_title(ax.get_title() + r', particle at ' + particle.to_str_cylindrical(True))

st.pyplot(fig)
