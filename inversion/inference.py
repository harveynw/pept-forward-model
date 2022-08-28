import jax.numpy as np

from jax import jit, vmap, random, grad
from inversion.integrals import scattering_density
from inversion.poisson_likelihood import single_particle_scattered_likelihood, single_particle_likelihood, \
    single_particle_scattered_likelihood_v2
from model import CylinderDetector


def encode_lors(d: CylinderDetector, lors: list):
    # Converts list of LoRs to a JAX array
    d_phi, d_z = d.del_detector_cells()

    detections_i = np.array([
        list(quad.min()) + [d_phi, d_z] for quad in [d.detector_cell_from_index(lor[0]) for lor in lors]
    ])
    detections_j = np.array([
        list(quad.min()) + [d_phi, d_z] for quad in [d.detector_cell_from_index(lor[1]) for lor in lors]
    ])

    return detections_i, detections_j


def create_likelihood(d, activity, T, lors, gamma, mu, mc_samples=5, mapped=True):
    # Returns a closure that evaluates the likelihood (and the gradient) for a given X (or X's)

    # Uniform samples on [0,1] x [0,1] for estimating integrals
    key = random.PRNGKey(0)
    unifs = random.uniform(key=key, shape=(mc_samples, 2))

    # Collecting LoRs into JAX Arrays
    detections_i, detections_j = encode_lors(d, lors)
    R, H = d.dim_radius_cm, d.dim_height_cm
    n_cells = d.n_detector_cells()[0] * d.n_detector_cells()[1]

    # Construct likelihood closure
    scat_dens_f = scattering_density
    likelihood_f = single_particle_scattered_likelihood_v2
    gradient_f = jit(grad(single_particle_scattered_likelihood_v2, 7))
    if mapped:
        scat_dens_f = jit(vmap(scat_dens_f, (None, None, 0), 0))
        likelihood_f = jit(vmap(likelihood_f, [None] * 7 + [0, 0] + [None] * 2, 0))
        gradient_f = jit(vmap(gradient_f, [None] * 7 + [0, 0] + [None] * 2, 0))

    def likelihood(X):
        scat_dens = scat_dens_f(R, mu, X)
        return likelihood_f(R, H, n_cells, activity, T, detections_i, detections_j, X, scat_dens, gamma, unifs)

    def gradient(X):
        scat_dens = scat_dens_f(R, mu, X)
        return gradient_f(R, H, n_cells, activity, T, detections_i, detections_j, X, scat_dens, gamma, unifs)

    return likelihood, gradient
