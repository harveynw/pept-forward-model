import jax.numpy as np

from jax import jit, vmap
from inversion.integrals import G_integral, H_integral
from jax_implementation import compute_joint_probability, compute_j_marginal_probability, compute_i_marginal_probability


@jit
def single_dimensional_likelihood(R, H, rate, T, detections_i, detections_j, X, gamma, unifs):
    joint_vmapped = vmap(compute_joint_probability, (None, 0, 0, None, None, None), 0)
    joint_evaluations = joint_vmapped(R, detections_i, detections_j, X, gamma, unifs)

    term_1 = np.sum(np.log(rate * T * joint_evaluations))
    term_2 = - rate * T * G_integral(R, H, X)
    return term_1 + term_2


@jit
def single_dimensional_scattered_likelihood(R, H, n_cells, rate, T, detections_i, detections_j, X, scattering_dens,
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