# `mps_tomo`

`mps_tomo` is our entry for Qiskit Camp Asia 2019. We perform quantum state tomography on states well-approximated by matrix product states according to the procedure in [[1]](#ref-q).

We use [Poetry](https://poetry.eustace.io/) for packaging. With it installed, you can call

```shell
poetry install
```

to install this package along with its dependencies. A dedicated Python virtual environment is recommended, and Python 3.7 is supported. An example Jupyter notebook showing its use is found in `examples/`, and tests are found in `tests/`.

## Background

In order for quantum device hardware to continue to improve, characterisation and benchmarking techniques need to stay ahead of the curve. State tomography is an example of device characterisation, checking that the intended final state at the end of a circuit is the one actually produced. A major limitation, however, is that the number of experiments required to produce the density matrix estimate scales like `3^n`.

An alternative approach is to model the final state as a matrix product state (MPS). The MPS formalism offers a different picture of quantum states, scaling with the entanglement (maximum Schmidt rank) of the system instead of the number of qubits. While state vector sizes grow like `2^n`, most MPSs are `poly(n)`. Applying techniques to find the MPS, tomography can be performed in a significantly reduced number of steps, and verified a-posteriori. This technique would be an ideal addition to Qiskit Ignis, and could be rigorously benchmarked. Further, because MPSs naturally describe the correlation present in the system, this technique could be used for investigation into relevant physical questions, such as the dynamical spread of quantum information.

[1]<a name="ref-1"></a> [Cramer, M. _et al_. Nat Commun **1**, 149 (2010)](https://doi.org/10.1038/ncomms1147)
