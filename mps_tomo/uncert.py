import numpy as np


def pauli_proj(dens_mat, pauli_prod):
    return np.trace(dens_mat @ pauli_prod) * pauli_prod / np.size(dens_mat, 0)
