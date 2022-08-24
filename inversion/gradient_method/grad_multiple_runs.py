import numpy as onp
import jax
import jax.numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
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

likelihood, gradient = create_likelihood(d, activity, T, lors, 50.0, mc_samples=5, mapped=True)

print('Starting...')

# Random positions
n_experiments = 10
X = np.array([random_point_within_cylinder(d.dim_radius_cm, d.dim_height_cm) for _ in range(n_experiments)])
nu = 0.05

err_history = []
n_iters = 50
for iter in tqdm(range(n_iters)):
    g = gradient(X)
    g_norm = np.linalg.norm(g, axis=1)
    g = (g.transpose() / g_norm).transpose()
    X = X + nu * (1 - iter/n_iters) * g
    err_history.append(jax.numpy.linalg.norm(X_actual - X, axis=1))


min_err, max_err, avg_err = [], [], []
for err in err_history:
    err = onp.array(err)
    min_err.append(onp.min(err))
    max_err.append(onp.max(err))
    avg_err.append(onp.mean(err))


plt.plot(avg_err, label='Average Error')
plt.fill_between(x=range(len(err_history)), y1=min_err, y2=max_err, alpha=0.5)

plt.plot(range(len(err_history)), min_err, '-', c='grey', label='Min Energy')
plt.plot(range(len(err_history)), max_err, '--', c='grey', label='Max Energy')

plt.title(f'Gradient Ascent Convergence for {p.to_str_cartesian()}, n={n_experiments} with random initial value')
plt.legend()
plt.show()


