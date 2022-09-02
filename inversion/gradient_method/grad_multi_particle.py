import itertools
import numpy as onp
import jax.numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from geometry import random_point_within_cylinder
from model import CylinderDetector, StaticParticle
from inversion.inference import create_multi_likelihood

d = CylinderDetector()
p = StaticParticle()
p.scatter_rate = 0.01

T = 1.0

activities = [500, 500, 500]
X_actual = [
    [0.12, -0.1, 0.13],
    [0.0, 0.0, 0.0],
    [-0.1, 0.1, -0.2]
]
lors = []

for activity, x in zip(activities, X_actual):
    p.set_position_cartesian(*x)
    lors_new, _ = p.simulate_emissions(detector=d, n_emissions=int(T * activity))
    lors += lors_new

print('Sim done')

likelihood, gradient = create_multi_likelihood(d, np.array(activities), T, lors, 50.0, p.scatter_rate, mc_samples=5)

print('Starting...')
print('XLA compilation of gradient may take a while...')

# Random positions
X = np.array([
    random_point_within_cylinder(d.dim_radius_cm, d.dim_height_cm) for _ in range(len(activities))
])
X_actual = np.array(X_actual)
print('Initial points', X)


def err(X_estimates):
    indices = range(len(activities))

    min_norm, err_norm = None, None
    for perm in list(itertools.permutations(indices)):
        val = np.linalg.norm(X_actual - X_estimates[perm, :], axis=1)
        if min_norm is None or np.sum(val) < min_norm:
            err_norm = val
            min_norm = np.sum(val)

    return err_norm


nu = 0.05
err_history, traj_history = [], []
n_iters = 500
for iter in tqdm(range(n_iters)):
    g = gradient(X)
    g_norm = np.linalg.norm(g, axis=1)
    g = (g.transpose() / g_norm).transpose()

    X = X + nu * (1 - iter/n_iters) * g

    # diff = X_actual - X
    # dot = np.tensordot(diff, diff, (1, 1))
    # err_history.append(np.sum(dot))
    # traj_history.append(X)
    err_history.append(err(X))
    traj_history.append(X)

print(np.around(onp.array(traj_history[-1]), 3))

min_err, max_err, avg_err = [], [], []
for err in err_history:
    min_err.append(onp.min(err))
    max_err.append(onp.max(err))
    avg_err.append(onp.mean(err))

t = onp.arange(start=0, stop=len(err_history))

plt.plot(t, avg_err, label='Average Error')
plt.fill_between(x=t, y1=min_err, y2=max_err, alpha=0.5)
plt.plot(t, min_err, '-', c='grey', label='Min Error')
plt.plot(t, max_err, '--', c='grey', label='Max Error')
plt.title(f'Gradient Ascent Convergence for n={len(activities)} particles initialised at random')
plt.xlabel('Iterations, n')
plt.ylabel(r'Error $\|x^{*} - x_n\|$')
plt.legend()
plt.show()


traj = onp.array(traj_history)
for traj_index in range(traj.shape[1]):
    t = traj[:, traj_index, :]

    plt.plot(t[:, 0], t[:, 1])
    plt.xlim((-d.dim_radius_cm, d.dim_radius_cm))
    plt.ylim((-d.dim_radius_cm, d.dim_radius_cm))

    # Real Maxima
    plt.scatter([X_actual[0]], [X_actual[1]])

    # Start, Finish
    plt.scatter([t[0, 0]], [t[0, 1]], c='green')
    plt.scatter([t[-1, 0]], [t[-1, 1]], c='red')

    plt.show()
