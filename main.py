import matplotlib.pyplot as plt
import numpy as np

from geometry import azimuth_of_point
from model import StaticParticle, CylinderDetector, Detector
from joblib import Parallel, delayed

particle = StaticParticle()
# particle.set_position_cartesian(0.1, 0, 0.1)
particle.set_position_cylindrical(r=0.1, theta=0, z=0.1)

detector = CylinderDetector()

# Plot 1: Find an emission with Compton Scattering
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(111, projection='3d')
# while True:
#     impacts, n_scatters = particle.simulate_emissions(detector=detector, n_lor=1, debug_ax=ax)
#
#     if n_scatters > 0:  # Make sure it did occur
#         ax.axes.set_xlim3d(left=-0.01-detector.dim_radius_cm, right=0.01+detector.dim_radius_cm)
#         ax.axes.set_ylim3d(bottom=-0.01-detector.dim_radius_cm, top=0.01+detector.dim_radius_cm)
#         ax.axes.set_zlim3d(bottom=-0.01, top=detector.dim_height_cm + 0.01)
#
#         plt.title('Static Radioactive Particle in Cylinder Detector')
#         # Put a legend to the right of the current axis
#         # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#         ax.view_init(azim=45, elev=30)
#         detector.debug_plot(ax=ax)
#         plt.show()
#         break
#     else:  # Otherwise try again
#         ax.clear()


# Plot 2: Analyse scattering rate, mainly for debugging
# def proportions(detector: Detector, scatter_rate=2.0, n_lor=10000):
#     p = StaticParticle(scatter_rate=scatter_rate)
#     p.set_position_cartesian(0, 0, 0.2)
#
#     lors, n_impacts = p.simulate_emissions(detector=detector, n_lor=n_lor)
#
#     # Detector hit rate overall from n_lor emissions, Scattering rate given detector hit
#     return len(lors)/n_lor, n_impacts/(2.0*len(lors))
#
#
# rates = np.linspace(start=0.001, stop=2.0, num=10)
# prop = np.array([proportions(detector=detector, scatter_rate=r) for r in rates])
# plt.figure()
# plt.plot(rates, prop[:, 0], label='Detector hit rate')
# plt.plot(rates, prop[:, 1], label='Scattering rate per emission')
# plt.xlabel(r'Scattering rate $\lambda$')
# plt.title(r'$P(\mathrm{Scattered})$ per photon emission and $P(\mathrm{LOR\:detected})$')
# plt.legend()
# plt.show()


# Plot 3: Detector hit rate as a function of particle position, takes a couples of minutes on an M1 Pro CPU
# def plot_3_sample_hit_rate(detector: Detector, r: float, z: float, n_lor=100):
#     p = StaticParticle()
#     p.set_position_cylindrical(r=r, theta=0.0, z=z)
#     lors, _ = p.simulate_emissions(detector=detector, n_lor=n_lor)
#     return len(lors)/n_lor
#
#
# N = 100
# r = np.linspace(start=-detector.dim_radius_cm+0.01, stop=detector.dim_radius_cm-0.01, num=N)
# z = np.linspace(start=-0.1, stop=detector.dim_height_cm+0.1, num=N)
#
# rr, zz = np.meshgrid(r, z, indexing='ij')
# hit_rate = Parallel(n_jobs=-1)(delayed(plot_3_sample_hit_rate)(detector, rr[i, j], zz[i, j]) for j in range(N) for i in range(N))
# hit_rate = np.array(hit_rate).reshape((N, N))
# h = plt.contourf(r, z, hit_rate)
#
# plt.title('LOR Detection Rate as a function of Particle Position')
# plt.xlabel('Radial position, r')
# plt.ylabel('z')
# plt.colorbar()
# plt.show()


# Plot 4: Debugging symmetry along z-axis
# def plot_4_sample_hit_rate(z=0.0, r=0.0):
#     p = StaticParticle()
#     p.set_position_cylindrical(r=r, theta=0.0, z=z)
#     d = CylinderDetector()
#
#     n_lors = 10000
#
#     lors, _ = p.simulate_emissions(detector=d, n_lor=n_lors)
#     return len(lors)/n_lors
#
#
# zz = np.linspace(-0.5, 1.0, 100)
# hit_rates = Parallel(n_jobs=-1)(delayed(plot_4_sample_hit_rate)(z) for z in zz)
# plt.vlines([0.0, 0.5], ymin=0.0, ymax=max(hit_rates))
# plt.plot(zz, hit_rates)
# plt.title('Hit rate against z coordinate')
# plt.xlabel('z')
# plt.show()

# Plot 5: Detector cell analysis
d5 = CylinderDetector()
cells_hit_counts = np.zeros(shape=d5.n_detector_cells(), dtype=int)
print(cells_hit_counts.shape)
p5 = StaticParticle()
p5.set_position_cylindrical(r=0.96*d5.dim_radius_cm, theta=np.pi, z=d5.dim_height_cm/2)
# p5.set_position_cylindrical(r=0.20, theta=np.pi/2, z=0.25)
p5.scatter_rate = 0.001
print('Mean scattering distance:', 1/p5.scatter_rate)

use_multicore = True
if use_multicore:
    n_batches = 8
    n_lor = 10000
    sims = Parallel(n_jobs=-1)(delayed(p5.simulate_emissions)(detector=d5, n_lor=n_lor) for _ in range(n_batches))
    lors, scatters = sum([sim[0] for sim in sims], []), sum([sim[1] for sim in sims])
else:
    lors, scatters = p5.simulate_emissions(detector=d5, n_lor=100000)

print('Statistics', scatters, len(lors), scatters/(2*len(lors)))
scatter_rate = scatters/(2*len(lors))

for lor in lors:
    for impact in lor:
        i, j = d5.detector_cell_from_impact(impact=impact)
        cells_hit_counts[i, j] += 1

plt.imshow(cells_hit_counts.transpose(), origin='lower')
plt.title(fr'Forward-Model Hit Count for particle at {p5.to_str_cylindrical(latex=True)}')
plt.suptitle(f'Scattering rate for individual photons is {scatter_rate:.00%}')
plt.xlabel('Horizontal')
plt.ylabel('Vertical')
plt.show()

# Plot 6: Detector cell analysis in another way
x_plot, y_plot = [], []
for lor in lors:
    for impact in lor:
        x, y, z = impact
        x_plot.append(azimuth_of_point(x, y))
        y_plot.append(z)

plt.hist2d(x_plot, y_plot, (314, 100), cmap=plt.cm.jet)
plt.title('Detector Hit Count')
plt.xlabel('Horizontal')
plt.ylabel('Vertical')
plt.show()
