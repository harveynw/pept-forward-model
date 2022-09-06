import matplotlib.pyplot as plt

from model import StaticParticle, CylinderDetector


particle = StaticParticle()
particle.set_position_cylindrical(r=0.0, theta=0, z=0.0)
detector = CylinderDetector()

while True:
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')

    impacts, n_scatters = particle.simulate_emissions(detector=detector, n_emissions=1, debug_ax=ax)

    if n_scatters != 1:  # Make sure one did occur
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()
        continue

    R, H = detector.dim_radius_cm, detector.dim_height_cm
    ax.axes.set_xlim3d(left=-0.01-R, right=0.01+R)
    ax.axes.set_ylim3d(bottom=-0.01-R, top=0.01+R)
    ax.axes.set_zlim3d(bottom=-0.01-H/2, top=H/2 + 0.01)

    ax.set_xlabel("x", labelpad=0)
    ax.set_ylabel("y", labelpad=0)
    ax.set_zlabel("z", labelpad=0)

    ax.tick_params(labelleft=False, labelright=False, labelbottom=False)
    # plt.title('Static Radioactive Particle in Cylinder Detector')
    ax.view_init(azim=45, elev=30)
    detector.debug_plot(ax=ax)
    plt.legend()
    plt.savefig('figures/one_scatter_diagram.png', transparent=True)
    plt.show()
