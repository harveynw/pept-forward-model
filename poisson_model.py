import jax.numpy as np


@jit
def single_particle_likelihood(R, i_n: np.array, j_n: np.array, X: np.array):
