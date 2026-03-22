from qiskit import *
from benchmarks.generate_benchmark import *
from qiskit_gridsynth_plugin.decompose import clifford_t_transpile
from qiskit.circuit.library import MCXGate
from qiskit import QuantumCircuit, transpile


def generate_balanced_target(nqubits):
    bits = ['0'] * (nqubits // 2) + ['1'] * (nqubits // 2)
    random.shuffle(bits)
    return ''.join(bits)

def qec_grover(nqubits, resource):
    target = generate_balanced_target(nqubits)
    
    anc_count = resource - nqubits
    
    if anc_count >= nqubits - 2:
        circuit = QuantumCircuit(nqubits + nqubits - 2)
        anc_qubits = [i for i in range(nqubits, nqubits + nqubits - 2)]
        mode='v-chain'
    elif anc_count >= 1:
        circuit = QuantumCircuit(nqubits + 1)
        anc_qubits = [i for i in range(nqubits, nqubits + 1)]
        mode='recursion'
    else:
        circuit = QuantumCircuit(nqubits)
        anc_qubits = [i for i in range(nqubits, resource)]
        mode='noancilla'

    # Step 1: Hadamard
    circuit.h(range(nqubits))

    # Step 2: Oracle
    for i, bit in enumerate(target):
        if bit == '0':
            circuit.x(i)
    circuit.h(nqubits - 1)
    circuit.append(MCXGate(nqubits - 1), list(range(nqubits)))
    circuit.h(nqubits - 1)
    for i, bit in enumerate(target):
        if bit == '0':
            circuit.x(i)

    # Step 3: Diffusion
    circuit.h(range(nqubits))
    circuit.x(range(nqubits))
    circuit.h(nqubits - 1)
    circuit.mcx(list(range(nqubits - 1)), nqubits - 1, ancilla_qubits=anc_qubits, mode=mode)
    circuit.h(nqubits - 1)
    circuit.x(range(nqubits))
    circuit.h(range(nqubits))
    circuit = circuit.decompose().decompose().decompose()
    decomposed = clifford_t_transpile(circuit, epsilon=1e-6)
    return decomposed

def pauli_strings_to_qiskit_circuit(pauli_strings, keep_length=None):
    if keep_length is not None:
        pauli_strings = pauli_strings[:keep_length]
    circ = QuantumCircuit(len(pauli_strings[0]))
    for paulis in pauli_strings:
        plist = []
        for i in range(len(paulis)):
            if paulis[i] == "X":
                circ.h(i)
                plist.append(i)
            elif paulis[i] == "Y":
                circ.h(i)
                circ.sdg(i)
                plist.append(i)
            elif paulis[i] == "Z":
                plist.append(i)
        if len(plist) > 1:
            for i in plist[1:]:
                circ.cx(i, plist[0])
            circ.rz(np.pi / 8, plist[0])  # change to an rather arbitrary angle
            for i in plist[1:]:
                circ.cx(i, plist[0])
        elif len(plist) == 1:
            circ.rz(np.pi / 8, plist[0])
        for i in range(len(paulis)):
            if paulis[i] == "X":
                circ.h(i)
            elif paulis[i] == "Y":
                circ.s(i)
                circ.h(i)
            elif paulis[i] == "Z":
                pass
    circ = transpile(circ, basis_gates=["u3", "id", "cz"], optimization_level = 2)
    return circ

class QsimRandBenchmark():
    def __init__(self, n_qubits, keep_length, p, i):
        super().__init__()
        self.n_qubits = n_qubits
        self.keep_length = keep_length
        self.p = p
        self.i = i
        self.path = f"qsim/rand/q{n_qubits}_{keep_length}_p{p}/i{i}.txt"
        # print(self.path)

        with open(self.path, "r") as fid:
            self.pauli_strings = eval(fid.read())[0:keep_length]

        self.circ = pauli_strings_to_qiskit_circuit(
            self.pauli_strings, keep_length=self.keep_length
        )

def qec_qsim(n_qubits: int):
    qc = QsimRandBenchmark(n_qubits, 10, 0.3, 0).circ
    decomposed = clifford_t_transpile(qc, epsilon=1e-6)
    return decomposed

def eft_qaoa(n):
    circ = genQiskitQAOA(n)

    tcirc = transpile(circ, basis_gates=['cx', 'h', 's', 'sdg', 'x', 'y', 'z', 'rz'])

    eft_circ = QuantumCircuit(n)
    for item in tcirc.data:
        op = item.operation.name
        qubits = [q._index for q in item.qubits]
        if op == 'cx':
            eft_circ.cx(qubits[0], qubits[1])
        elif op == 'h':
            eft_circ.h(qubits[0])
        elif op == 's':
            eft_circ.s(qubits[0])
        elif op == 'sdg':
            eft_circ.sdg(qubits[0])
        elif op == 'x':
            eft_circ.x(qubits[0])
        elif op == 'y':
            eft_circ.y(qubits[0])
        elif op == 'z':
            eft_circ.z(qubits[0])
        elif op == 'rz':
            eft_circ.t(qubits[0])
    return eft_circ

def qec_qaoa(n):
    circ = genQiskitQAOA(n)

    ft_circ = clifford_t_transpile(circ, 10e-6)
    return ft_circ

def qec_qft(n):
    circ = genQiskitQFT(n)

    ft_circ = clifford_t_transpile(circ, 10e-3)    
    return ft_circ

