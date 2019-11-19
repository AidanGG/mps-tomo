from functools import reduce

import numpy as np

I = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def pauli_group(num_qubits):
    PAULIS = [I, X, Y, Z]

    def pauli_prod(bitflag):
        indices = ((bitflag >> (2 * i)) & 3 for i in reversed(range(num_qubits)))
        return reduce(np.kron, (PAULIS[i] for i in indices))

    return (pauli_prod(i) for i in range(4 ** num_qubits))
