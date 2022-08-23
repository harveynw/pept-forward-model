import jax.numpy as np

from jax import jit, vmap, random, grad
from inversion.integrals import scattering_density
from inversion.poisson_likelihood import single_dimensional_scattered_likelihood, single_dimensional_likelihood
from model import CylinderDetector

@jit
def single_dimensional_scattered_likelihood_grad(R, H, n_cells, rate, T, detections_i, detections_j, X, scattering_dens,
                                                 gamma, unifs):
    return grad(single_dimensional_scattered_likelihood, 7)(R, H, n_cells, rate, T, detections_i, detections_j, X, scattering_dens,
                                                            gamma, unifs)


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


def create_likelihood(d, activity, T, lors, gamma, mc_samples=5, mapped=True):
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
    likelihood_f = single_dimensional_scattered_likelihood
    gradient_f = jit(grad(single_dimensional_scattered_likelihood, 7))
    if mapped:
        scat_dens_f = jit(vmap(scat_dens_f, (None, 0), 0))
        likelihood_f = jit(vmap(likelihood_f, [None] * 7 + [0, 0] + [None] * 2, 0))
        gradient_f = lambda *args: None  # Not needed yet

    def likelihood(X):
        scat_dens = scat_dens_f(R, X)
        return likelihood_f(R, H, n_cells, activity, T, detections_i, detections_j, X, scat_dens, gamma, unifs)

    def gradient(X):
        scat_dens = scat_dens_f(R, X)
        return gradient_f(R, H, n_cells, activity, T, detections_i, detections_j, X, scat_dens, gamma, unifs)

    return likelihood, gradient


def eval_single_dimensional_likelihood(d: CylinderDetector, activity: float, T: float, lors: list, gamma: float,
                                       X: np.array, scattering=False, gradient=False):
    # Uniform samples on [0,1] x [0,1] for estimating integral
    key = random.PRNGKey(0)
    unifs = random.uniform(key=key, shape=(5, 2))

    # Collecting LoRs into JAX Arrays
    detections_i, detections_j = encode_lors(d, lors)
    R, H = d.dim_radius_cm, d.dim_height_cm
    n_cells = d.n_detector_cells()[0] * d.n_detector_cells()[1]

    if np.ndim(X) == 1:
        if scattering:
            scat_dens = scattering_density(R, X)
            if gradient:
                return single_dimensional_scattered_likelihood_grad(R=R, H=H, n_cells=n_cells, rate=activity, T=T,
                                                               detections_i=detections_i, detections_j=detections_j, X=X,
                                                               scattering_dens=scat_dens, gamma=gamma, unifs=unifs)
            else:
                return single_dimensional_scattered_likelihood(R=R, H=H, n_cells=n_cells, rate=activity, T=T,
                                                           detections_i=detections_i, detections_j=detections_j, X=X,
                                                           scattering_dens=scat_dens, gamma=gamma, unifs=unifs)
        else:
            return single_dimensional_likelihood(R=R, H=H, rate=activity, T=T,
                                                 detections_i=detections_i, detections_j=detections_j, X=X,
                                                 gamma=gamma, unifs=unifs)
    else:
        if scattering:
            scattering_mapped = jit(vmap(scattering_density, (None, 0), 0))
            scat_dens_eval = scattering_mapped(R, X)
            likelihood_mapped = jit(vmap(single_dimensional_scattered_likelihood,
                                         (None, None, None, None, None, None, None, 0, 0, None, None), 0))
            return likelihood_mapped(R, H, n_cells, activity, T, detections_i, detections_j, X, scat_dens_eval,
                                     gamma, unifs)
        else:
            # Vmap case
            likelihood_mapped = jit(vmap(single_dimensional_likelihood,
                                         (None, None, None, None, None, None, 0, None, None), 0))
            return likelihood_mapped(R, H, activity, T, detections_i, detections_j, X, gamma, unifs)