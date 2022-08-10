import jax.numpy as np

from jax import vmap, jit


@jit
def f(a, b, c):
    return a + b + c


f_map = vmap(f, in_axes=(0, None, 0), out_axes=0)

print(f_map(np.array([1, 2, 3]), 1, np.array([10, 20, 30])))
