from functools import reduce

import numpy as np

from mps_tomo.utils import kron_embed, pauli_group


def pauli_proj(dens_mat, pauli_prod):
    return np.einsum("ij,ji", dens_mat, pauli_prod) * pauli_prod / np.size(dens_mat, 0)


def R_hat(sigmas, num_qubits):
    return sum(kron_embed(*i, num_qubits) for i in enumerate(sigmas))


def iteration(k, sigmas, num_qubits, max_its=100, delta=0.1):
    dim = 2 ** num_qubits

    R = R_hat(sigmas, num_qubits)

    Y = np.zeros((dim, dim), dtype=np.complex128)

    for i in range(max_its):
        w, v = np.linalg.eigh(Y)
        y_val, y_vec = w[-1], v[:, -1]

        paulis = (
            kron_embed(q, p, num_qubits)
            for p in pauli_group(k)
            for q in range(num_qubits - k + 1)
        )

        X = y_val / dim * reduce(np.add, (y_vec.conj() @ p @ y_vec * p for p in paulis))
        Y += delta * (R - X)

    return y_vec
