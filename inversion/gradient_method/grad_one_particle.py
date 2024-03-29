import numpy as onp
import jax
import jax.numpy as np
import matplotlib.pyplot as plt

from model import CylinderDetector, StaticParticle
from inversion.inference import create_likelihood

d = CylinderDetector()
p = StaticParticle()
p.set_position_cartesian(x=0.12, y=-0.1, z=0.13)
p.scatter_rate = 2.0
T, activity = 1.0, 10 ** 4
lors, scatters = p.simulate_emissions(detector=d, n_emissions=int(T * activity))
X_actual = np.array(p.get_position_cartesian())
print('Sim done')

likelihood, gradient = create_likelihood(d, activity, T, lors, 50.0, p.scatter_rate, mc_samples=5, mapped=False)


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
n_iters = 100
for iter in range(n_iters):
    g = grad_fd(X)
    g = g / jax.numpy.linalg.norm(g)
    exact_g = gradient(X)
    exact_g = exact_g / jax.numpy.linalg.norm(exact_g)
    print('X', X, 'Grad', g, 'Exact gradient', exact_g)
    X = X + nu * (1 - iter/n_iters) * g
    likelihood_history.append(jax.numpy.linalg.norm(X_actual - X))


plt.plot(onp.arange(stop=n_iters), likelihood_history)
plt.title(rf'Gradient Ascent Error starting at origin, real position {p.to_str_cartesian()}')
plt.savefig('figures/likelihood/gradient_ascent.png', format='png')
plt.show()


