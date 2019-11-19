from functools import reduce

import numpy as np
from numpy.testing import assert_allclose
from qiskit.quantum_info import pauli_group, random_density_matrix

from mps_tomo.uncert import pauli_proj, R_hat
from mps_tomo.utils import pauli_group


def test_pauli_proj():
    N_QUBITS = 3
    SEED = 7777
    TOL = 100 * np.spacing(np.complex128(1.0).real)

    dens_mat = random_density_matrix(2 ** N_QUBITS, seed=SEED)
    reconstructed = reduce(
        np.add, (pauli_proj(dens_mat, p) for p in pauli_group(N_QUBITS))
    )

    assert_allclose(reconstructed, dens_mat, rtol=0.0, atol=TOL)


def test_R_hat():
    TOL = 100 * np.spacing(np.complex128(1.0).real)
    np.random.seed(7777)

    sigmas = [np.random.randn(4, 4) + 1j * np.random.randn(4, 4) for i in range(3)]

    example = (
        np.kron(sigmas[0], np.eye(4))
        + np.kron(np.kron(np.eye(2), sigmas[1]), np.eye(2))
        + np.kron(np.eye(4), sigmas[2])
    )

    assert_allclose(R_hat(sigmas), example, rtol=0.0, atol=TOL)
