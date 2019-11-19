import numpy as np
import qiskit
from qiskit import QuantumRegister, QuantumCircuit
from qiskit import Aer
from qiskit.ignis.verification.tomography import state_tomography_circuits
from qiskit.ignis.verification.tomography import StateTomographyFitter
from qiskit import IBMQ

import Qconfig #Not 100% sure how this credentials section is going to work

qx_config = {
    "APItoken": Qconfig.APItoken,
    "url": Qconfig.config['url'],
    "hub": Qconfig.config['hub'],
    "group": Qconfig.config['group'],
    "project": Qconfig.config['project']}

print('Qconfig loaded from %s.' % Qconfig.__file__)

IBMQ.enable_account(qx_config['APItoken'])
provider = IBMQ.get_provider(hub='ibm-q-melbourne', group='internal', project='default')
list_of_backends = provider.backends()
print('\nYou have access to')
print(list_of_backends)

class tomographicConstruction():
    """Class to implement the tomographic circuits necessary for MPS reconstruction

    Initialisation:
    ------------------------------
    R : int
        ansatz bond dimension
    nQ : int
        number of qubits in the circuit
    initial_circuit : qiskit QuantumCircuit object
        Circuit which produces the state which is the object of characterisation
    backend : IBMQ backend
        Hardware on which the circuits are to be run. Defaults to 'ibmq_qasm_simulator'
    """



    def __init__(self,R,nQ,initial_circuit,backend=provider.get_backend('ibmq_qasm_simulator')):
        self.R = R
        self.nQ = nQ
        self.inital_circuit = initial_circuit


    def collect_tomography_data():
        K = int(np.ceil(np.log2(self.R)) + 1)
        nDMs = self.nQ - K + 1
        DM_list = []
        q = self.initial_circuit.qregs[0]
        for i in range(nDMs):
            tomography_circuits = state_tomography_circuits(self.initial_circuit, q[i:i+K])
            job = execute(tomography_circuits, backend, shots=8192, optimization_level=3)
            result = job.result()
            state_tom = StateTomographyFitter(result, tomography_circuits)
            DM = state_tom.fit(method='cvx')
            DM_list.append(DM)
        self.DM_list = DM_list
        return DM_list
