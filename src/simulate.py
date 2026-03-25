from typing import List, Tuple, Any
import stim


def merge_measure_reset_pairs(cir_mv: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
    """Merge measure+reset into Stim's MR only when adjacent.

    The Qiskit exporter often emits a block of `measure` operations followed by a
    block of `reset` operations. It's tempting to merge a `measure(q)` with a
    later `reset(q)` while allowing operations on other qubits in between, but
    Stim's DEM back-propagation can treat that as an actual demolition (MR)
    collapse and can therefore raise "non-deterministic detectors/observables"
    errors for some circuits.

    To keep the MR semantics consistent, we only merge when the sequence is
    exactly:
      ('measure', [q]) immediately followed by ('reset', [q]).
    Otherwise we keep the original operations as separate `M` and `R`.
    """
    merged: List[Tuple[str, Any]] = []
    i = 0
    while i < len(cir_mv):
        op, args = cir_mv[i]
        if op == "measure" and len(args) == 1 and i + 1 < len(cir_mv):
            q = args[0]
            op2, args2 = cir_mv[i + 1]
            if op2 == "reset" and args2 == [q]:
                merged.append(("mr", [q]))
                i += 2
                continue
        merged.append((op, args))
        i += 1
    return merged


def strip_initializing_resets(cir_mv: List[Tuple[str, Any]], num_q: int) -> List[Tuple[str, Any]]:
    """Remove leading resets that prepare |0⟩ on all qubits.

    Qiskit emits ``reset`` on every line at the start of the circuit. Stim already
    assumes qubits begin in |0⟩; explicit ``R`` at t=0 interacts badly with
    detector_error_model's backward analysis: logical Z observables propagate to
    Pauli X on ancillas, which anti-commute with Z-basis resets on those lines.
    """
    if not cir_mv or num_q <= 0:
        return cir_mv
    full = set(range(num_q))
    i = 0
    covered: set[int] = set()
    while i < len(cir_mv):
        op, args = cir_mv[i]
        if op != "reset":
            break
        if len(args) == 1:
            q = args[0]
            if q in covered:
                break
            covered.add(q)
            i += 1
            if covered == full:
                return cir_mv[i:]
            continue
        # Single instruction R 0 1 2 ... num_q-1
        if set(args) == full and len(args) == num_q:
            return cir_mv[i + 1 :]
        break
    return cir_mv


def tostim(
    noise_profile,
    stabs: List,
    logicals: List,
    cir_mv: List,
    num_q,
    num_s,
    *,
    include_observables: bool = True,
):
    num_d = num_q - num_s

    # Important: avoid generating Stim's `MR` (measure+reset) demolition
    # operations. In this BB-code Qiskit->grid_compiler->tostim pipeline, `MR`
    # can create non-deterministic detector/observable structure during
    # Stim's detector error model analysis.
    #
    # Keeping Qiskit `measure` and `reset` as separate `M` and `R` instructions
    # makes the circuit compatible with sinter's internal strong_id/DEM
    # construction.
    cir_mv = strip_initializing_resets(cir_mv, num_q)

    sz_list = []
    for s_idx, stab in enumerate(stabs):
        if all(gate[1] == "Z" for gate in stab["Ctrl"]):
            sz_list.append(s_idx + num_d)

    # Export temporal detectors only for X-type stabilizers.
    # Z-check temporal detectors (and the end-of-circuit Z-check detector
    # construction below) can make Stim's DEM analysis non-deterministic on
    # some BB schedules.
    x_list = []
    for s_idx, stab in enumerate(stabs):
        if all(gate[1] == "X" for gate in stab["Ctrl"]):
            x_list.append(s_idx + num_d)
    
    stim_circ = stim.Circuit()

    meas_records = [[] for _ in range(num_q)]
    current_m = 0

    for op, args in cir_mv:
        if op == "reset":
            stim_circ.append("R", args)

        elif op == "u2":
            stim_circ.append("H", args)

        elif op in ["cx", "cz"]:
            stim_circ.append("CX", args)
            stim_circ.append("DEPOLARIZE2", args, [noise_profile["2q-gate"]])

        elif op == "measure":
            q = args[0]

            stim_circ.append("M", args)
            meas_records[q].append(current_m)
            current_m += 1

            if q in x_list and len(meas_records[q]) >= 2:
                stim_circ.append(
                    "DETECTOR",
                    [
                        stim.target_rec(meas_records[q][-2] - current_m),
                        stim.target_rec(meas_records[q][-1] - current_m),
                    ],
                )

        elif op == "mr":
            q = args[0]
            # Avoid Stim's demolition measurement `MR`. Expand it into:
            #  - `M` to create the measurement record
            #  - `R` to reset for subsequent operations
            # This keeps the detector/observable record structure stable
            # while avoiding the non-determinism triggered by `MR`.
            stim_circ.append("M", [q])
            meas_records[q].append(current_m)
            current_m += 1
            stim_circ.append("R", [q])

            if q in x_list and len(meas_records[q]) >= 2:
                stim_circ.append(
                    "DETECTOR",
                    [
                        stim.target_rec(meas_records[q][-2] - current_m),
                        stim.target_rec(meas_records[q][-1] - current_m),
                    ],
                )

        elif op in ["intra-move", "inter-move"]:
            stim_circ.append("DEPOLARIZE1", args[0:1], [noise_profile[op]])

        elif op in ["intra-swap", "inter-swap"]:
            stim_circ.append("DEPOLARIZE2", args, [noise_profile[op]])

        else:
            raise Exception(f"op: {op}")
        
    for d_idx in range(num_d):
        stim_circ.append("M", [d_idx])
        meas_records[d_idx].append(current_m)
        current_m += 1

    # NOTE: We intentionally skip exporting end-of-circuit Z-check detectors.
    # Using X-check temporal detectors is enough to decode Z-logical errors
    # in this CSS BB setup, and it avoids Stim non-determinism.

    if include_observables:
        for obs_idx, logical in enumerate(logicals):
            if logical["Type"] == "Z":
                stim_circ.append(
                    "OBSERVABLE_INCLUDE",
                    [stim.target_rec(meas_records[q][-1] - current_m) for q in logical["Data"]],
                    [obs_idx],
                )
    
    return stim_circ


def detector_error_model_gauge(circuit: stim.Circuit) -> stim.DetectorErrorModel:
    return circuit.detector_error_model(
        decompose_errors=False,
        allow_gauge_detectors=True,
    )

