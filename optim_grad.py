import numpy as onp
import jax
import jax.numpy as np
import matplotlib.pyplot as plt

from model import CylinderDetector, StaticParticle
from poisson_model import eval_single_dimensional_likelihood

d = CylinderDetector()
p = StaticParticle()
p.set_position_cartesian(x=0.12, y=-0.1, z=0.13)
p.scatter_rate = 2.0
T, activity = 1.0, 10 ** 4
lors, scatters = p.simulate_emissions(detector=d, n_lor=int(T * activity))
X_actual = np.array(p.get_position_cartesian())
print('Sim done')

likelihood = lambda x: eval_single_dimensional_likelihood(d=d,
                                                          activity=activity,
                                                          T=T,
                                                          lors=lors,
                                                          gamma=50.0,
                                                          X=x,
                                                          scattering=True)
gradient = lambda x: eval_single_dimensional_likelihood(d=d,
                                                        activity=activity,
                                                        T=T,
                                                        lors=lors,
                                                        gamma=50.0,
                                                        X=x,
                                                        scattering=True,
                                                        gradient=True)


def grad_fd(x):
    h = 0.00001

    del_x = np.array([1, 0, 0])
    dx = likelihood(x + h * del_x) - likelihood(x - h * del_x)

    del_y = np.array([0, 1, 0])
    dy = likelihood(x + h * del_y) - likelihood(x - h * del_y)

    del_z = np.array([0, 0, 1])
    dz = likelihood(x + h * del_z) - likelihood(x - h * del_z)

    return np.array([dx, dy, dz]) / (2.0 * h)

print('Compiled gradient')

print('Starting...')
X = np.array([0.0, 0.0, 0.0])
nu = 0.05

likelihood_history = []
n_iters = 20
for iter in range(n_iters):
    g = grad_fd(X)
    g = g / jax.numpy.linalg.norm(g)
    print('X', X, 'Grad', g)
    X = X + nu * (1 - iter/n_iters) * g
    likelihood_history.append(jax.numpy.linalg.norm(X_actual - X))

plt.plot(onp.arange(stop=n_iters), likelihood_history)
plt.show()


