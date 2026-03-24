from typing import List
import stim

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

    sz_list = []
    for s_idx, stab in enumerate(stabs):
        if all(gate[1] == "Z" for gate in stab["Ctrl"]):
            sz_list.append(s_idx + num_d)
    
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

            if q in sz_list and len(meas_records[q]) >= 2:
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

    for s_idx, stab in enumerate(stabs):
        if all(gate[1] == "Z" for gate in stab["Ctrl"]):
            check_list = [stim.target_rec(meas_records[s_idx + num_d][-1]-current_m)]
            for q, _, _ in stab["Ctrl"]:
                check_list.append(stim.target_rec(meas_records[q][-1]-current_m))
            stim_circ.append("DETECTOR", check_list)

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

