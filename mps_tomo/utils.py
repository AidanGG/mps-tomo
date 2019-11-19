from functools import lru_cache, reduce

import numpy as np

I = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


@lru_cache(maxsize=None)
def pauli_group(num_qubits):
    PAULIS = [I, X, Y, Z]

    def pauli_prod(bitflag):
        indices = ((bitflag >> (2 * i)) & 3 for i in reversed(range(num_qubits)))
        return reduce(np.kron, (PAULIS[i] for i in indices))

    return [pauli_prod(i) for i in range(4 ** num_qubits)]


def kron_embed(qubit, op, qubits):
    left_size = 2 ** qubit
    op_size = np.size(op, axis=0)
    right_size = (2 ** qubits) // (left_size * op_size)

    left, right = (
        np.eye(left_size, dtype=np.complex128),
        np.eye(right_size, dtype=np.complex128),
    )

    return np.kron(np.kron(left, op), right)
