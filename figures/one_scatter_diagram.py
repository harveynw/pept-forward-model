import matplotlib.pyplot as plt

from model import StaticParticle, CylinderDetector


particle = StaticParticle()
particle.set_position_cylindrical(r=0.0, theta=0, z=0.25)

detector = CylinderDetector()

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
while True:
    impacts, n_scatters = particle.simulate_emissions(detector=detector, n_lor=1, debug_ax=ax)

    if n_scatters == 1:  # Make sure one did occur
        ax.axes.set_xlim3d(left=-0.01-detector.dim_radius_cm, right=0.01+detector.dim_radius_cm)
        ax.axes.set_ylim3d(bottom=-0.01-detector.dim_radius_cm, top=0.01+detector.dim_radius_cm)
        ax.axes.set_zlim3d(bottom=-0.01, top=detector.dim_height_cm + 0.01)

        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        #
        ax.set_xlabel("x", labelpad=20)
        ax.set_ylabel("y", labelpad=20)
        ax.set_zlabel("z", labelpad=20)

        ax.tick_params(labelleft=False, labelright=False, labelbottom=False)

        plt.title('Static Radioactive Particle in Cylinder Detector')
        ax.view_init(azim=45, elev=30)
        detector.debug_plot(ax=ax)
        plt.show()
        plt.savefig('figures/one_scatter_diagram.eps', format='eps')
        break
    else:  # Otherwise, try again
        ax.clear()