import numpy as np
import time


def updatePauliOrdering(curOrdering):
    indexIncremented = False
    for i in range(len(curOrdering)):
        if curOrdering[i] == 3:
            curOrdering[i] = 0
        else:
            curOrdering[i] += 1
            indexIncremented = True
            break
    if not indexIncremented:
        # We are out of combinations
        curOrdering[0] = -1
    return


def findMaxElements(a, N):
    return np.argsort(a)[::-1][:N]


def buildPauli(curPauliChain, paulis):
    # Build the pauli op for the current chain
    pauliOp = paulis[curPauliChain[0]]
    for j in range(1, len(curPauliChain)):
        pauliOp = np.kron(pauliOp, paulis[curPauliChain[j]])
    return pauliOp


def samplePauliChains(numQubits, opSpaceDim, paulis, psiMps, m):
    # compute density matrix
    rhoMps = np.outer(psiMps, np.conj(psiMps))
    # Probability distribution
    probDist = np.zeros(opSpaceDim)
    # MPS expectation values
    mpsExpVals = np.zeros(opSpaceDim)
    # The current Pauli chain
    curPauliChain = [0] * numQubits
    # List of the pauli chains
    pauliChains = [0] * opSpaceDim
    # Store the chain number we're up to
    pauliChainIndex = 0

    # Iterate through all possible pauli chains
    while curPauliChain[0] != -1:
        # Build the pauli op for the current chain
        pauliOp = buildPauli(curPauliChain, paulis)
        # print("Pauli chain: " + str(curPauliChain) + "\n")
        # print("Pauli op: " + str(pauliOp) + "\n")
        # Compute the conditional probability for the operator
        expVal = np.trace(pauliOp.dot(rhoMps))
        # print("MatrixProd: " + str(pauliOp.dot(rhoMps)) + "\n")
        # print("MPSExp: " + str(expVal) + "\n")
        # Store the prob and the corresponding pauli chain
        mpsExpVals[pauliChainIndex] = np.abs(expVal)
        probDist[pauliChainIndex] = np.abs(expVal) ** 2
        pauliChains[pauliChainIndex] = curPauliChain
        pauliChainIndex += 1
        # Compute the next pauli chain
        updatePauliOrdering(curPauliChain)
    # Extract the chains with the m largest probabilities
    # print("Number of exp vals: " + str(len(mpsExpVals)) + "\n")
    maxChainIndices = findMaxElements(probDist, m)
    maxPauliChains = [0] * m
    maxPauliExpVals = np.zeros(m)
    for i in range(m):
        maxPauliChains[i] = pauliChains[maxChainIndices[i]]
        maxPauliExpVals[i] = mpsExpVals[maxChainIndices[i]]
    return [maxPauliChains, maxPauliExpVals]


def computeFidelity(labExpVals, mpsExpVals, maxPauliChains, m, paulis, numQubits):
    normalisation = (1 / (2 ** numQubits)) ** 0.5
    fidEst = 0
    # Get the first pauli op
    curPauliChain = maxPauliChains[0]
    pauliOp = normalisation * buildPauli(curPauliChain, paulis)
    # First term in the expansion of the density matrices
    rhoActApprox = labExpVals[0] * pauliOp
    rhoMpsApprox = mpsExpVals[0] * pauliOp
    for i in range(1, m):
        curPauliChain = maxPauliChains[i]
        pauliOp = normalisation * buildPauli(curPauliChain, paulis)
        rhoActApprox += labExpVals[i] * pauliOp
        rhoMpsApprox += mpsExpVals[i] * pauliOp
    print(
        "Traces: MPS: "
        + str(np.trace(rhoMpsApprox))
        + " lab: "
        + str(np.trace(rhoActApprox))
        + "\n"
    )
    print(rhoActApprox)
    print(rhoMpsApprox)
    fidEst = np.trace(rhoActApprox * rhoMpsApprox)
    return fidEst


"""
Main driver function to estimate the fidelity between
"""


def estimateFidelity(numQubits, rhoMps, numSamples, opSpaceDim):
    # Single qubit pauli matrices
    id = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    paulis = np.array([id, X, Y, Z])

    # Get samples of the numSamples dominant pauli chains (now have all sigma k)
    pauliSamples = samplePauliChains(numQubits, opSpaceDim, paulis, rhoMps, numSamples)
    pauliChains = pauliSamples[0]
    mpsExpVals = pauliSamples[1]

    # Get the numSamples expectation values from greg
    # labExpVals = samplePauliChains(numQubits, opSpaceDim, paulis, rhoMps, numSamples)

    # compute fidelity estimate
    # print(mpsExpsVals)
    fidEst = computeFidelity(
        mpsExpVals, mpsExpVals, pauliChains, numSamples, paulis, numQubits
    )
    return fidEst


if __name__ == "__main__":
    numQubits = 8
    dim = 2 ** numQubits
    opSpaceDim = dim ** 2
    # Test rho
    rhoMps = (1 / dim) * np.eye(dim)
    numSamples = 10
    t0 = time.time()
    print(estimateFidelity(numQubits, rhoMps, numSamples, opSpaceDim))
    print(time.time() - t0)
