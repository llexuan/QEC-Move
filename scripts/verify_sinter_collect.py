import os
import sys

import sinter
from qiskit import QuantumCircuit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stabilizer_code import extract, surface_code, to_qiskit
from src.simulate import tostim
from lib.TrapSIMD.grid_iontrap import grid_compiler


def gen_noise_profile2(p_trans=2e-4):
    return {
        "intra-move": p_trans,
        "intra-swap": p_trans,
        "inter-move": 2 * p_trans,
        "inter-swap": 4 * p_trans,
        "2q-gate": 18.3e-4,
    }


def main() -> None:
    path = "QEC-Code"
    os.makedirs(path, exist_ok=True)

    d = 3
    stab_fname = os.path.join(path, f"surface-code-{d}.stab")
    qasm_fname = os.path.join(path, f"surface-code-{d}.qasm")

    surface_code(stab_fname, d)
    is_css, is_ordered, stabilizers, logicals, data_num = extract(stab_fname)
    to_qiskit(qasm_fname, is_css, is_ordered, stabilizers, data_num, iter=d * 3)

    qc = QuantumCircuit.from_qasm_file(qasm_fname)
    _, node_list = grid_compiler(5, 3, 3, qc, False)

    circ = tostim(
        gen_noise_profile2(2e-4),
        stabilizers,
        logicals,
        node_list,
        qc.num_qubits,
        qc.num_clbits,
    )
    stats = sinter.collect(
        num_workers=2,
        tasks=[sinter.Task(circuit=circ, json_metadata={"d": d, "p": 2e-4})],
        decoders=["pymatching"],
        max_shots=200,
        print_progress=False,
    )
    print("ok", stats[0].shots, stats[0].errors, stats[0].discards)


if __name__ == "__main__":
    main()
