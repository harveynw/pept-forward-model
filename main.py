import matplotlib.pyplot as plt
import numpy as np

from model import StaticParticle, CylinderDetector, Detector

particle = StaticParticle()
particle.set_position_cartesian(0, 0, 0.2)

detector = CylinderDetector()

# Plot 1: Find an emission with Compton Scattering
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
while True:
    impacts, n_scatters = particle.simulate_emissions(detector=detector, n_lor=1, debug_ax=ax)

    if n_scatters > 0:  # Make sure it did occur
        ax.axes.set_xlim3d(left=-0.1-detector.dim_radius_cm, right=0.1+detector.dim_radius_cm)
        ax.axes.set_ylim3d(bottom=-0.1-detector.dim_radius_cm, top=0.1+detector.dim_radius_cm)
        ax.axes.set_zlim3d(bottom=-0.1, top=detector.dim_height_cm + 0.1)

        plt.title('Static Radioactive Particle in Cylinder Detector')
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        detector.debug_plot(ax=ax)
        plt.show()
        break
    else:  # Otherwise try again
        ax.clear()


# Plot 2: Analyse scattering rate, mainly for debugging
def proportions(detector: Detector, scatter_rate=2.0, n_lor=10000):
    p = StaticParticle(scatter_rate=scatter_rate)
    p.set_position_cartesian(0, 0, 0.2)

    lors, n_impacts = p.simulate_emissions(detector=detector, n_lor=n_lor)

    # Detector hit rate overall from n_lor emissions, Scattering rate given detector hit
    return len(lors)/n_lor, n_impacts/(2.0*len(lors))


rates = np.linspace(start=0.001, stop=2.0, num=10)
prop = np.array([proportions(detector=detector, scatter_rate=r) for r in rates])
plt.figure()
plt.plot(rates, prop[:, 0], label='Detector hit rate')
plt.plot(rates, prop[:, 1], label='Scattering rate per emission')
plt.xlabel(r'Scattering rate $\lambda$')
plt.title(r'$P(\mathrm{Scattered})$ per photon emission and $P(\mathrm{LOR\:detected})$')
plt.legend()
plt.show()
