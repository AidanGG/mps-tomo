import numpy as np

from mps_tomo.utils import kron_embed, pauli_group


def pauli_proj(dens_mat, pauli_prod):
    return np.einsum("ij,ji", dens_mat, pauli_prod) * pauli_prod / np.size(dens_mat, 0)


def R_hat(sigmas, num_qubits):
    return sum(kron_embed(*i, num_qubits) for i in enumerate(sigmas))


def iteration(k, sigmas, num_qubits, max_its=100, delta=0.1):
    dim = 2 ** num_qubits
    paulis = pauli_group(k)

    R = R_hat(sigmas, num_qubits)

    Y = np.zeros((dim, dim), dtype=np.complex128)
    for _ in range(max_its):
        w, v = np.linalg.eigh(Y)
        y_val, y_vec = w[-1], v[:, -1]

        X = np.zeros_like(Y)
        for qubit in range(num_qubits - k + 1):
            y_vec_reshaped = np.reshape(y_vec, (2 ** qubit, 2 ** k, -1))
            for p in paulis:
                X += np.dot(
                    y_vec.conj(), np.ravel(np.einsum("ij,ajb->aib", p, y_vec_reshaped))
                ) * kron_embed(qubit, p, num_qubits)
        X *= y_val / dim

        Y += delta * (R - X)

    return y_vec
