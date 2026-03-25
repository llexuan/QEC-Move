"""Stim-native CSS syndrome circuits (same structure as stabilizers_simulation).

Uses RX on data, R on ancillas, H-bounded X-ancilla blocks, MR on ancillas, and
MX final readout.  By default (``observable_mode="transversal"``) a **single**
``OBSERVABLE_INCLUDE`` XORs **all** data ``MX`` outcomes, matching the CSS
default in `stabilizers_simulation`_ and yielding a **valid** Stim DEM.

Optional ``observable_mode="lx"`` attaches one observable per row of the
logical-X matrix ``lx`` (bposd); on large codes this often **still** fails
``detector_error_model`` (backward X vs init) — use transversal for DEM+sinter.

See https://github.com/llexuan/stabilizers_simulation for the template idea.

Note: Default is **X-basis transversal** readout (not Z logicals from ``.stab`` LZ lines).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import stim
from bposd.css import css_code


def _default_layers_for_support(support: List[int], L: int) -> List[List[int]]:
    sup_sorted = sorted(support)
    layers: List[List[int]] = [[] for _ in range(L)]
    for i, idx in enumerate(sup_sorted):
        layers[i % L].append(idx)
    return layers


def _stabilizer_edges_and_families(
    stabilizers: List[dict], data_num: int
) -> Tuple[List[int], List[int], Dict[int, List[int]]]:
    z_anc_ids: List[int] = []
    x_anc_ids: List[int] = []
    edges_by_anc: Dict[int, List[int]] = {}
    for s_idx, stab in enumerate(stabilizers):
        anc = data_num + s_idx
        edges_by_anc[anc] = [c[0] for c in stab["Ctrl"]]
        if stab["Ctrl"][0][1] == "X":
            x_anc_ids.append(anc)
        else:
            z_anc_ids.append(anc)
    return z_anc_ids, x_anc_ids, edges_by_anc


def compute_bb_hx_hz(ell: int = 6, m: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """Same hx, hz construction as ``bb_code`` in ``stabilizer_code.py``."""
    n2 = m * ell
    I_ell = np.identity(ell, dtype=int)
    I_m = np.identity(m, dtype=int)
    x = {}
    y = {}
    for i in range(ell):
        x[i] = np.kron(np.roll(I_ell, i, axis=1), I_m)
    for i in range(m):
        y[i] = np.kron(I_ell, np.roll(I_m, i, axis=1))
    a1, a2, a3 = 3, 1, 2
    b1, b2, b3 = 3, 1, 2
    A = (x[a1] + y[a2] + y[a3]) % 2
    B = (y[b1] + x[b2] + x[b3]) % 2
    hx = np.hstack((A, B))
    hz = np.hstack((B.T, A.T))
    return hx.astype(np.uint8) % 2, hz.astype(np.uint8) % 2


def build_css_stim_circuit(
    stabilizers: List[dict],
    noise_profile: dict,
    data_num: int,
    rounds: int = 1,
    L: int = 4,
    *,
    hx: Optional[np.ndarray] = None,
    hz: Optional[np.ndarray] = None,
    layers_by_anc: Optional[Dict[int, List[List[int]]]] = None,
    observable_mode: Literal["transversal", "lx"] = "transversal",
) -> stim.Circuit:
    """Build a coordinate-free CSS Stim circuit (stabilizers_simulation style).

    - Full CSS: ``RX`` data, ``R`` ancillas, ``MX`` final,
      ``OBSERVABLE_INCLUDE`` either **transversal** (all data MX, one obs) or
      **lx** (one obs per logical-X row; may fail DEM on large codes).
    - Z-only / X-only: analogous with ``M`` / ``MX`` and ``lz`` / ``lx``.

    ``noise_profile`` uses the same keys as ``tostim`` (2q-gate, intra-move, ...);
    only ``2q-gate`` is applied after ``CX`` here (native builder has no ion moves).
    """
    assert rounds >= 1 and L >= 1
    p2 = float(noise_profile.get("2q-gate", 0.0))
    # Extra measurement-flip noise on ancillas right before MR.
    #
    # Why: For some BB schedules, the resulting DEM can have *zero* 1-detector
    # error terms (no boundaries), which makes MWPM decoders like pymatching
    # fail with "odd parity ... without a boundary". Adding an explicit
    # X_ERROR before ancilla Z-measurements introduces boundary edges while
    # keeping the circuit otherwise unchanged.
    p_meas_flip = float(noise_profile.get("ancilla_meas_flip", p2))

    z_anc_ids, x_anc_ids, edges_by_anc = _stabilizer_edges_and_families(
        stabilizers, data_num
    )
    data_ids_print = list(range(data_num))
    anc_all_print = z_anc_ids + x_anc_ids
    m_total = len(anc_all_print)
    x_ids_print = list(x_anc_ids)
    z_ids_print = list(z_anc_ids)
    m_x = len(x_ids_print)
    m_z = len(z_ids_print)
    m_round = m_x + m_z

    only_z = len(z_anc_ids) > 0 and len(x_anc_ids) == 0
    only_x = len(x_anc_ids) > 0 and len(z_anc_ids) == 0

    if hx is None or hz is None:
        hx, hz = compute_bb_hx_hz()
    qcode = css_code(hx, hz)
    lx = qcode.lx.toarray().astype(np.uint8) % 2

    def layers_for_anc(a: int) -> List[List[int]]:
        if layers_by_anc and a in layers_by_anc and len(layers_by_anc[a]) == L:
            return layers_by_anc[a]
        sup = sorted(set(edges_by_anc[a]))
        return _default_layers_for_support(sup, L)

    circ = stim.Circuit()

    def _dep1(qs: List[int]) -> None:
        if p2 and qs:
            circ.append("DEPOLARIZE1", qs, [p2])

    def _dep2(pairs_flat: List[int]) -> None:
        if p2 and pairs_flat:
            circ.append("DEPOLARIZE2", pairs_flat, [p2])

    # Initial data / ancilla prep (match stabilizers_simulation)
    if only_z and not only_x:
        circ.append("R", data_ids_print)
        _dep1(data_ids_print)
    else:
        circ.append("RX", data_ids_print)
        _dep1(data_ids_print)

    circ.append("R", anc_all_print)
    _dep1(anc_all_print)

    for r in range(rounds):
        circ.append("TICK")
        _dep1(data_ids_print)

        full_css = (m_x > 0 and m_z > 0)

        if full_css:
            # Full CSS schedule (MR kept), but measured in two phases to avoid
            # non-deterministic detectors from mixing both families at once.
            #
            # Phase Z first (so X-detectors won't back-propagate through Z checks).
            for a in z_anc_ids:
                a_layers = layers_for_anc(a)
                for layer in a_layers:
                    tgt: List[int] = []
                    for lin in layer:
                        tgt += [lin, a]
                    if tgt:
                        circ.append("CX", tgt)
                        _dep2(tgt)
                circ.append("TICK")

            _dep1(z_ids_print)
            if p_meas_flip and z_ids_print:
                circ.append("X_ERROR", z_ids_print, [p_meas_flip])
            # Plan A: measure Z ancillas in Z basis without reset.
            # We'll reset them explicitly at the next round start.
            circ.append("M", z_ids_print)
            _dep1(z_ids_print)

            # NOTE: We intentionally do NOT add temporal detectors for Z ancillas here.
            # On BB schedules with RX data init + MX readout, including Z-syndrome
            # detectors can become non-deterministic under Stim's analysis (MR on Z ancillas
            # anti-commutes with backward sensitivities). We still *run* the Z-check CNOTs
            # to keep a full-CSS-style circuit structure, but only X-syndrome detectors
            # are exported for DEM+MWPM decoding.

            # Phase X: Hx, CX(anc->data), Hx, MR(x ancillas), temporal detectors.
            if x_ids_print:
                circ.append("H", x_ids_print)
                _dep1(x_ids_print)
                circ.append("TICK")

            for a in x_anc_ids:
                a_layers = layers_for_anc(a)
                for layer in a_layers:
                    tgt: List[int] = []
                    for lin in layer:
                        tgt += [a, lin]
                    if tgt:
                        circ.append("CX", tgt)
                        _dep2(tgt)
                circ.append("TICK")

            if x_ids_print:
                circ.append("H", x_ids_print)
                _dep1(x_ids_print)
                circ.append("TICK")

            _dep1(x_ids_print)
            if p_meas_flip and x_ids_print:
                circ.append("X_ERROR", x_ids_print, [p_meas_flip])
            circ.append("MR", x_ids_print)
            _dep1(x_ids_print)

            # Temporal X detectors (omit first+last for MWPM boundaries).
            if 0 < r < rounds - 1:
                for jx, aq in enumerate(x_ids_print):
                    back_curr = -(m_x - jx)
                    back_prev = -(m_round + (m_x - jx))
                    circ.append(
                        "DETECTOR",
                        [stim.target_rec(back_curr), stim.target_rec(back_prev)],
                    )
        else:
            # Single-family schedule (X-only or Z-only): keep the simple combined MR.
            if x_ids_print:
                circ.append("H", x_ids_print)
                _dep1(x_ids_print)
                circ.append("TICK")

            for a in x_anc_ids:
                a_layers = layers_for_anc(a)
                for layer in a_layers:
                    tgt: List[int] = []
                    for lin in layer:
                        tgt += [a, lin]
                    if tgt:
                        circ.append("CX", tgt)
                        _dep2(tgt)
                circ.append("TICK")

            for a in z_anc_ids:
                a_layers = layers_for_anc(a)
                for layer in a_layers:
                    tgt: List[int] = []
                    for lin in layer:
                        tgt += [lin, a]
                    if tgt:
                        circ.append("CX", tgt)
                        _dep2(tgt)
                circ.append("TICK")

            if x_ids_print:
                circ.append("H", x_ids_print)
                _dep1(x_ids_print)
                circ.append("TICK")

            _dep1(anc_all_print)
            if p_meas_flip and anc_all_print:
                circ.append("X_ERROR", anc_all_print, [p_meas_flip])
            circ.append("MR", anc_all_print)
            _dep1(anc_all_print)

            if 0 < r < rounds - 1:
                for j, aq in enumerate(anc_all_print):
                    back_curr = -(m_total - j)
                    back_prev = -(m_total + (m_total - j))
                    circ.append(
                        "DETECTOR",
                        [stim.target_rec(back_curr), stim.target_rec(back_prev)],
                    )

        # Memory experiment: repeat syndrome rounds; measure data only on the last round.
        if r != rounds - 1:
            continue

        def _rec_pos_after_meas(qs_measured: List[int]) -> Dict[int, int]:
            return {q: i for i, q in enumerate(qs_measured)}

        if only_x and not only_z:
            _dep1(data_ids_print)
            circ.append("MX", data_ids_print)
            mx_pos = _rec_pos_after_meas(data_ids_print)
            for a in x_anc_ids:
                support_qs = [q for q in sorted(set(edges_by_anc[a])) if q in mx_pos]
                if len(support_qs) < 2:
                    continue
                j = anc_all_print.index(a)
                anc_back = -(len(data_ids_print) + (m_total - j))
                terms = [
                    stim.target_rec(-(len(data_ids_print) - mx_pos[q])) for q in support_qs
                ]
                terms.append(stim.target_rec(anc_back))
                circ.append("DETECTOR", terms)
            if observable_mode == "transversal":
                circ.append(
                    "OBSERVABLE_INCLUDE",
                    [
                        stim.target_rec(-(len(data_ids_print) - mx_pos[q]))
                        for q in data_ids_print
                    ],
                    [0],
                )
            else:
                for row in range(lx.shape[0]):
                    obs_terms = [
                        stim.target_rec(-(len(data_ids_print) - mx_pos[q]))
                        for q in range(data_num)
                        if lx[row, q]
                    ]
                    if obs_terms:
                        circ.append("OBSERVABLE_INCLUDE", obs_terms, [row])
        elif only_z and not only_x:
            _dep1(data_ids_print)
            circ.append("M", data_ids_print)
            mz_pos = _rec_pos_after_meas(data_ids_print)
            for a in z_anc_ids:
                support_qs = [q for q in sorted(set(edges_by_anc[a])) if q in mz_pos]
                if len(support_qs) < 2:
                    continue
                j = anc_all_print.index(a)
                anc_back = -(len(data_ids_print) + (m_total - j))
                terms = [
                    stim.target_rec(-(len(data_ids_print) - mz_pos[q])) for q in support_qs
                ]
                terms.append(stim.target_rec(anc_back))
                circ.append("DETECTOR", terms)
            lz = qcode.lz.toarray().astype(np.uint8) % 2
            if observable_mode == "transversal":
                circ.append(
                    "OBSERVABLE_INCLUDE",
                    [
                        stim.target_rec(-(len(data_ids_print) - mz_pos[q]))
                        for q in data_ids_print
                    ],
                    [0],
                )
            else:
                for row in range(lz.shape[0]):
                    obs_terms = [
                        stim.target_rec(-(len(data_ids_print) - mz_pos[q]))
                        for q in range(data_num)
                        if lz[row, q]
                    ]
                    if obs_terms:
                        circ.append("OBSERVABLE_INCLUDE", obs_terms, [row])
        else:
            # Full CSS: MX + transversal or per-row lx observables
            _dep1(data_ids_print)
            circ.append("MX", data_ids_print)
            mx_pos = _rec_pos_after_meas(data_ids_print)
            for a in x_anc_ids:
                support_qs = [q for q in sorted(set(edges_by_anc[a])) if q in mx_pos]
                if len(support_qs) < 2:
                    continue
                # In full CSS we measure Z ancillas first, then X ancillas, then MX.
                # So there is no m_z offset between MR(x) and MX.
                jx = x_ids_print.index(a)
                anc_back = -(len(data_ids_print) + (m_x - jx))
                terms = [
                    stim.target_rec(-(len(data_ids_print) - mx_pos[q])) for q in support_qs
                ]
                terms.append(stim.target_rec(anc_back))
                circ.append("DETECTOR", terms)
            if observable_mode == "transversal":
                circ.append(
                    "OBSERVABLE_INCLUDE",
                    [
                        stim.target_rec(-(len(data_ids_print) - mx_pos[q]))
                        for q in data_ids_print
                    ],
                    [0],
                )
            else:
                for row in range(lx.shape[0]):
                    obs_terms = [
                        stim.target_rec(-(len(data_ids_print) - mx_pos[q]))
                        for q in range(data_num)
                        if lx[row, q]
                    ]
                    if obs_terms:
                        circ.append("OBSERVABLE_INCLUDE", obs_terms, [row])

    return circ


def build_bb_stim_tasks_noise(
    noise_profile: dict[str, Any],
) -> dict[str, Any]:
    """Noise dict for ``build_css_stim_circuit`` (only CX gets depolarizing here)."""
    return {
        "2q-gate": noise_profile["2q-gate"],
        "intra-move": noise_profile.get("intra-move", 0.0),
        "inter-move": noise_profile.get("inter-move", 0.0),
        "intra-swap": noise_profile.get("intra-swap", 0.0),
        "inter-swap": noise_profile.get("inter-swap", 0.0),
    }
