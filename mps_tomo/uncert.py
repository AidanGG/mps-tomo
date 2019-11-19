from functools import reduce

import numpy as np


def pauli_proj(dens_mat, pauli_prod):
    return np.trace(dens_mat @ pauli_prod) * pauli_prod / np.size(dens_mat, 0)


def R_hat(sigmas):
    sigma_size = np.size(sigmas[0], axis=0)
    dim = sigma_size * (2 ** (len(sigmas) - 1))

    def kron_sigma(qubit, sigma):
        left_size = 2 ** qubit
        right_size = dim // (left_size * sigma_size)
        left, right = (
            np.eye(left_size, dtype=np.complex128),
            np.eye(right_size, dtype=np.complex128),
        )
        return np.kron(np.kron(left, sigma), right)

    return reduce(np.add, (kron_sigma(*i) for i in enumerate(sigmas)))
