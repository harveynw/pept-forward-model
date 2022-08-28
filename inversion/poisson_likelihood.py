import jax.numpy as np

from jax import jit, vmap
from inversion.integrals import G_integral, H_integral
from inversion.jax_implementation import compute_joint_probability, compute_j_marginal_probability, \
    compute_i_marginal_probability


@jit
def single_particle_likelihood(R, H, rate, T, detections_i, detections_j, X, gamma, unifs):
    joint_vmapped = vmap(compute_joint_probability, (None, 0, 0, None, None, None), 0)
    joint_evaluations = joint_vmapped(R, detections_i, detections_j, X, gamma, unifs)

    term_1 = np.sum(np.log(rate * T * joint_evaluations))
    term_2 = - rate * T * G_integral(R, H, X)
    return term_1 + term_2


@jit
def _compute_scattering_joint_density(R, detector_i: tuple, detector_j: tuple, X: np.array, H_x: np.array,
                                      n_cells: float, scattering_dens: np.array, gamma, unifs: np.array):
    # Returns P(i, j | X) = Σ P(i, j | S, X)P(S | X)
    P = np.array([
        (H_x/n_cells) ** 2,
        H_x/n_cells * compute_j_marginal_probability(R, detector_j, X),
        compute_i_marginal_probability(R, detector_i, X) * H_x/n_cells,
        compute_joint_probability(R, detector_i, detector_j, X, gamma, unifs)
    ])
    return np.dot(P, scattering_dens)


@jit
def single_particle_scattered_likelihood(R, H, n_cells, rate, T, detections_i, detections_j, X, scattering_dens,
                                         gamma, unifs):
    # TODO: A hotfix
    x_comp, y_comp, z_comp = X
    X = np.array([y_comp, x_comp, z_comp])

    G_x = G_integral(R, H, X)
    H_x = H_integral(R, H, X)

    # P(T,T) P(T, F) P(F, T) P(F, F)
    prob_s_1, prob_s_2, prob_s_3, prob_s_4 = scattering_dens

    # Case 1: Both Scattered
    term_1 = prob_s_1 * (H_x / n_cells) ** 2

    # Case 2: T, F
    marginal_j_vmapped = vmap(compute_j_marginal_probability, (None, 0, None), 0)
    marginal_j_evaluations = marginal_j_vmapped(R, detections_j, X)
    term_1 = term_1 + prob_s_2 * (H_x / n_cells) * marginal_j_evaluations

    # Case 3: F, T
    marginal_i_vmapped = vmap(compute_i_marginal_probability, (None, 0, None), 0)
    marginal_i_evaluations = marginal_i_vmapped(R, detections_i, X)
    term_1 = term_1 + prob_s_3 * marginal_i_evaluations * (H_x / n_cells)

    # Case 4: Unscattered
    joint_vmapped = vmap(compute_joint_probability, (None, 0, 0, None, None, None), 0)
    joint_evaluations = joint_vmapped(R, detections_i, detections_j, X, gamma, unifs)
    term_1 = term_1 + prob_s_4 * joint_evaluations

    return np.sum(np.log(term_1)) - rate * T * G_x


@jit
def single_particle_scattered_likelihood_v2(R, H, n_cells, rate, T, detections_i, detections_j, X, scattering_dens,
                                            gamma, unifs):
    # TODO: A hotfix
    X = X[[1, 0, 2], ]

    G_x, H_x = G_integral(R, H, X), H_integral(R, H, X)

    joint_vmapped = vmap(_compute_scattering_joint_density, [None, 0, 0] + [None] * 6, 0)
    joint_eval = joint_vmapped(R, detections_i, detections_j, X, H_x, n_cells, scattering_dens, gamma, unifs)

    return np.sum(np.log(joint_eval)) - rate * T * G_x


@jit
def multiple_particle_scattered_likelihood(R, H, n_cells, rates, T, detections_i, detections_j, X, scattering_dens_X,
                                           gamma, unifs):
    # Assuming k particles
    # X (k, 3) = [ [x_0, y_0, z_0], ... , [x_k, y_k, z_k] ]
    # rates (k,) = [rate_0, ... , rate_k]
    # scattering_dens_X (k, 4) = [ [P(T,T | 0) P(T, F | 0) P(F, T | 0) P(F, F | 0)] , ... , [ P(T,T | k) ... ] ]

    # TODO: A hotfix
    X = X[:, [1, 0, 2]]

    # (k,), (k,)
    G_x, H_x = vmap(G_integral, (None, None, 0), 0)(R, H, X), vmap(H_integral, (None, None, 0), 0)(R, H, X)

    # P(i, j | X_k) (n, k)
    joint_vmapped_1 = vmap(_compute_scattering_joint_density, (None, None, None, 0, 0, None, 0, None, None), 0)
    joint_vmapped_2 = vmap(joint_vmapped_1, (None, 0, 0, None, None, None, None, None, None), 0)
    joint_evals = joint_vmapped_2(R, detections_i, detections_j, X, H_x, n_cells, scattering_dens_X, gamma, unifs)

    return np.sum(np.log(np.tensordot(rates, joint_evals, (0, 1)))) - T * np.dot(rates, G_x)
