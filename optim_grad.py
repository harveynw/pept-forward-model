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


if __name__ == '__main__':
    X_0 = np.array([0.0, 0.0, 0.0])

    d = CylinderDetector()
    p = StaticParticle()
    p.set_position_cartesian(x=0.12, y=0.1, z=0.03)
    p.scatter_rate = 2.0
    T, activity = 1.0, 10 ** 4
    lors, scatters = p.simulate_emissions(detector=d, n_lor=int(T * activity))


    dels = []
    real_sol = p.get_position_cartesian()
    for vec in history:
        dels.append(onp.linalg.norm(real_sol - onp.array(vec)))

    plt.plot(onp.array(range(len(history))), onp.array(dels))
    plt.show()


