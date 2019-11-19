from functools import reduce

import numpy as np
from numpy.testing import assert_allclose
from qiskit.quantum_info import pauli_group, random_density_matrix

import mps_tomo.uncert as uncert
import mps_tomo.utils


def test_pauli_proj():
    N_QUBITS = 3
    SEED = 7777
    TOL = 100 * np.spacing(np.complex128(1.0).real)

    dens_mat = random_density_matrix(2 ** N_QUBITS, seed=SEED)
    reconstructed = reduce(
        np.add,
        (uncert.pauli_proj(dens_mat, p) for p in mps_tomo.utils.pauli_group(N_QUBITS)),
    )

    assert_allclose(reconstructed, dens_mat, rtol=0.0, atol=TOL)
