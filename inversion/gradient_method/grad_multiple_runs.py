import numpy as onp
import jax
import jax.numpy as np
import matplotlib.pyplot as plt

from geometry import random_point_within_cylinder
from model import CylinderDetector, StaticParticle
from inversion.inference import create_likelihood

d = CylinderDetector()
p = StaticParticle()
p.set_position_cartesian(x=0.12, y=-0.1, z=0.13)
p.scatter_rate = 2.0
T, activity = 1.0, 10 ** 4
lors, scatters = p.simulate_emissions(detector=d, n_lor=int(T * activity))
X_actual = np.array(p.get_position_cartesian())
print('Sim done')

likelihood, gradient = create_likelihood(d, activity, T, lors, 50.0, p.scatter_rate, mc_samples=5, mapped=True)

print('Starting...')
print('XLA compilation of gradient may take a while...')

# Random positions
n_experiments = 100
X = np.array([random_point_within_cylinder(d.dim_radius_cm, d.dim_height_cm) for _ in range(n_experiments)])
print('Initial points', X)
nu = 0.05

err_history = []
n_iters = 10
for iter in range(n_iters):
    g = gradient(X)
    g_norm = np.linalg.norm(g, axis=1)
    g = (g.transpose() / g_norm).transpose()
    X = X + nu * (1 - iter/n_iters) * g
    errs = jax.numpy.linalg.norm(X_actual - X, axis=1)
    err_history.append(errs)
    print(iter)

min_err, max_err, avg_err = [], [], []
for err in err_history:
    err = onp.array(err)
    err = err[~onp.isnan(err)]

    min_err.append(onp.min(err))
    max_err.append(onp.max(err))
    avg_err.append(onp.mean(err))

print(min_err)
print(max_err)
print(avg_err)

t = onp.arange(start=0, stop=len(err_history))

plt.plot(t, avg_err, label='Average Error')
plt.fill_between(x=t, y1=min_err, y2=max_err, alpha=0.5)
plt.plot(t, min_err, '-', c='grey', label='Min Error')
plt.plot(t, max_err, '--', c='grey', label='Max Error')
plt.title(f'Gradient Ascent Convergence for {p.to_str_cartesian()}, n={n_experiments} with random initial position')
plt.legend()
plt.show()


