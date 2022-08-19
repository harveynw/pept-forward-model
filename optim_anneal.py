import numpy as onp
import jax.numpy as np
import random
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Callable, Any, List
from dataclasses import dataclass

# State type can be any object/number
from model import CylinderDetector, StaticParticle
from poisson_model import eval_single_dimensional_likelihood

State = Any


def default_temperature(k: int, k_max: int) -> float:
    return 1.0-k/k_max


def default_acceptance(e, e_dash, t: float) -> float:
    return 1.0 if e_dash < e else onp.exp(-(e_dash - e)/t)


@dataclass
class Anneal:
    """
    Class for performing Simulated Annealing
    https://en.wikipedia.org/wiki/Simulated_annealing#Pseudocode
    """

    # Initial state
    s_0: Any
    # Max number of iterations
    k_max: int
    # Function for picking a random neighbour
    neighbour_func: Callable[[State], Any]
    # Energy function
    energy_func: Callable[[State], float]
    # Temperature function
    temperature_func: Callable[[int, int], float] = default_temperature
    # Acceptance function
    acceptance_func: Callable[[State, State, float], float] = default_acceptance

    def simulate(self) -> (State, List[Any]):
        print(f'Begin: Anneal(k_max={self.k_max})')

        s = self.s_0
        history = []

        for k in tqdm(range(self.k_max)):
            history.append(s)

            t = self.temperature_func(k, self.k_max)
            s_new = self.neighbour_func(s)

            p = self.acceptance_func(self.energy_func(s),
                                     self.energy_func(s_new),
                                     t)

            if p > random.uniform(0, 1):
                s = s_new

        return s, history


if __name__ == '__main__':
    X_0 = np.array([0.0, 0.0, 0.0])

    d = CylinderDetector()
    p = StaticParticle()
    p.set_position_cartesian(x=0.12, y=0.1, z=0.03)
    p.scatter_rate = 2.0
    T, activity = 1.0, 10 ** 4
    lors, scatters = p.simulate_emissions(detector=d, n_lor=int(T * activity))

    def neighbour(X):
        while True:
            phi = onp.random.uniform(0, 2*onp.pi)
            dist = onp.random.uniform(0, 0.5*d.dim_radius_cm)

            X_cand = X + np.array([dist*onp.cos(phi),
                                   dist*onp.sin(phi),
                                   onp.random.uniform(-d.dim_height_cm/4, d.dim_height_cm/4)])

            if -d.dim_height_cm/2 < X_cand[2] < d.dim_height_cm/2:
                if np.sqrt(X_cand[0]**2 + X_cand[1]**2) < d.dim_radius_cm:
                    return X_cand

    def energy(X):
        return eval_single_dimensional_likelihood(d=d,
                                                  activity=activity,
                                                  T=T,
                                                  lors=lors,
                                                  gamma=50.0,
                                                  X=X,
                                                  scattering=True)

    anneal = Anneal(s_0=X_0, k_max=1000, energy_func=energy, neighbour_func=neighbour)
    x_sol, history = anneal.simulate()

    print(history)

    dels = []
    real_sol = p.get_position_cartesian()
    for vec in history:
        dels.append(onp.linalg.norm(real_sol - onp.array(vec)))

    plt.plot(onp.array(range(len(history))), onp.array(dels))
    plt.show()


