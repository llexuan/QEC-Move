"""Run the corrected GenCode surface-code parameter sweep."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import sinter
from qiskit import QuantumCircuit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.TrapSIMD.grid_iontrap import grid_compiler
from src.simulate import detector_error_model_gauge, tostim
from src.stabilizer_code import extract, surface_code, to_qiskit


def noise_profile(p_trans: float) -> dict[str, float]:
    return {
        "intra-move": p_trans,
        "intra-swap": p_trans,
        "inter-move": 2 * p_trans,
        "inter-swap": 4 * p_trans,
        "2q-gate": 18.3e-4,
    }


def build_tasks() -> list[sinter.Task]:
    tasks: list[sinter.Task] = []
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        for distance in (3, 5, 7):
            stab_path = root / f"surface-code-{distance}.stab"
            qasm_path = root / f"surface-code-{distance}.qasm"
            surface_code(stab_path, distance)
            is_css, is_ordered, stabilizers, logicals, data_num = extract(stab_path)
            rounds = distance * 3
            to_qiskit(
                qasm_path,
                is_css,
                is_ordered,
                stabilizers,
                data_num,
                iter=rounds,
            )

            circuit = QuantumCircuit.from_qasm_file(qasm_path)
            _, operations = grid_compiler(5, 3, 3, circuit, False)

            for p_trans in (1e-4, 1.5e-4, 2e-4):
                stim_circuit = tostim(
                    noise_profile(p_trans),
                    stabilizers,
                    logicals,
                    operations,
                    circuit.num_qubits,
                    circuit.num_clbits,
                )
                tasks.append(
                    sinter.Task(
                        circuit=stim_circuit,
                        detector_error_model=detector_error_model_gauge(stim_circuit),
                        json_metadata={
                            "d": distance,
                            "r": rounds,
                            "p": p_trans,
                        },
                    )
                )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-shots", type=int, default=200_000)
    parser.add_argument("--max-errors", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    stats = sinter.collect(
        num_workers=args.workers,
        tasks=build_tasks(),
        decoders=["pymatching"],
        max_shots=args.max_shots,
        max_errors=args.max_errors,
        print_progress=True,
    )

    print("distance,p_trans,shots,errors,logical_error_per_round")
    for sample in sorted(
        stats,
        key=lambda item: (item.json_metadata["d"], item.json_metadata["p"]),
    ):
        shot_error_rate = sample.errors / sample.shots
        per_round = sinter.shot_error_rate_to_piece_error_rate(
            shot_error_rate=shot_error_rate,
            pieces=sample.json_metadata["r"],
        )
        print(
            f"{sample.json_metadata['d']},"
            f"{sample.json_metadata['p']},"
            f"{sample.shots},"
            f"{sample.errors},"
            f"{per_round}"
        )


if __name__ == "__main__":
    main()
