import numpy as np
import streamlit as st
st.set_page_config(layout="wide")

from inversion.generate_plots import plot_proj_area
from model import CylinderDetector, StaticParticle

d = CylinderDetector()
R, H = d.dim_radius_cm, d.dim_height_cm

st.title('Characteristic function')
st.subheader('Detector cell size')

d.detectors_width = st.slider('Width', min_value=0.005, max_value=R, step=0.005, value=0.14)
d.detectors_height = st.slider('Height', min_value=0.005, max_value=H/4.0, step=0.005, value=0.07)

n_x, n_y = d.n_detector_cells()
d_x, d_y = d.del_detector_cells()

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

st.subheader('Sharpness')

gamma = st.slider('gamma', min_value=1, max_value=1000, value=50, step=1)

fig, ax = plot_proj_area(min_phi=phi_1_range[0],
                         min_z=z_1_range[0],
                         d_phi=d_x, d_z=d_y,
                         p=particle, d=d, gamma=gamma)

ax.set_title(ax.get_title() + r', particle at ' + particle.to_str_cylindrical(True))

st.text('(Warning: this is a potentially slow running plot)')
st.pyplot(fig)
