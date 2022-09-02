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
p.scatter_rate = 0.01
T, activity = 1.0, 10 ** 4
lors, scatters = p.simulate_emissions(detector=d, n_emissions=int(T * activity))
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

err_history, traj_history = [], []
n_iters = 50
for iter in tqdm(range(n_iters)):
    g = gradient(X)
    g_norm = np.linalg.norm(g, axis=1)
    g = (g.transpose() / g_norm).transpose()

    X = X + nu * (1 - iter/n_iters) * g

    errs = jax.numpy.linalg.norm(X_actual - X, axis=1)
    err_history.append(errs)
    traj_history.append(X)

min_err, max_err, avg_err = [], [], []
for err in err_history:
    min_err.append(onp.min(err))
    max_err.append(onp.max(err))
    avg_err.append(onp.mean(err))

t = onp.arange(start=0, stop=len(err_history))

plt.fill_between(x=t, y1=min_err, y2=max_err, color='#86b3cf')
plt.plot(t, avg_err, label='Average Error')
plt.plot(t, min_err, '-', c='grey', label='Min Error')
plt.plot(t, max_err, '--', c='grey', label='Max Error')
plt.title(f'Gradient Ascent of Likelihood recovering {p.to_str_cartesian()} over n={n_experiments} runs')
plt.xlabel('Iterations, n')
plt.ylabel(r'Error $\|x^{*} - x_n\|$')
plt.legend()
plt.savefig('figures/grad_ascent_multiple.png', format='png')
plt.savefig('figures/grad_ascent_multiple.eps', format='eps', bbox_inches='tight')

exit()

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
