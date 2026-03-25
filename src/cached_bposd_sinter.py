"""Faster sinter integration for ldpc BpOsdDecoder.

The upstream ``ldpc.sinter_decoders.SinterBpOsdDecoder`` only implements
``decode_via_files``. sinter then falls back to ``DiskDecoder``, which calls
``decode_via_files`` once per batch. That re-reads the DEM from disk and rebuilds
check matrices + ``BpOsdDecoder`` every time — often minutes per batch on large
BB-style DEMs, and looks like a "stuck" run.

This module provides a drop-in replacement that implements
``compile_decoder_for_dem`` so setup is amortized across all shots.
"""

from __future__ import annotations

import numpy as np
import stim
import sinter
from ldpc.bposd_decoder import BpOsdDecoder
from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices


def _pack_obs_row_b8(pred: np.ndarray, num_obs: int) -> np.ndarray:
    """Bit-pack one row of observable bits (stim ``b8``, little-endian per byte)."""
    nbytes = (num_obs + 7) // 8
    out = np.zeros(nbytes, dtype=np.uint8)
    for b in range(nbytes):
        v = 0
        for bit in range(8):
            idx = b * 8 + bit
            if idx < num_obs and pred[idx]:
                v |= 1 << bit
        out[b] = v
    return out


class _CompiledCachedBpOsd(sinter.CompiledDecoder):
    def __init__(
        self,
        *,
        dem: stim.DetectorErrorModel,
        max_iter: int,
        bp_method: str,
        ms_scaling_factor: float,
        schedule: str,
        omp_thread_count: int,
        serial_schedule_order,
        osd_method: str,
        osd_order: int,
    ) -> None:
        self._num_det = dem.num_detectors
        self._num_obs = dem.num_observables
        self._matrices = detector_error_model_to_check_matrices(
            dem, allow_undecomposed_hyperedges=True
        )
        self._bposd = BpOsdDecoder(
            self._matrices.check_matrix,
            error_channel=list(self._matrices.priors),
            max_iter=max_iter,
            bp_method=bp_method,
            ms_scaling_factor=ms_scaling_factor,
            schedule=schedule,
            omp_thread_count=omp_thread_count,
            serial_schedule_order=serial_schedule_order,
            osd_method=osd_method,
            osd_order=osd_order,
        )
        self._obs_mat = self._matrices.observables_matrix

    def decode_shots_bit_packed(
        self,
        *,
        bit_packed_detection_event_data: np.ndarray,
    ) -> np.ndarray:
        n = bit_packed_detection_event_data.shape[0]
        n_obs_b = (self._num_obs + 7) // 8
        out = np.zeros((n, n_obs_b), dtype=np.uint8)
        for i in range(n):
            syn = np.unpackbits(
                bit_packed_detection_event_data[i],
                bitorder="little",
            )[: self._num_det]
            corr = self._bposd.decode(syn)
            pred = (self._obs_mat @ corr) % 2
            out[i] = _pack_obs_row_b8(pred.astype(np.uint8), self._num_obs)
        return out


class CachedBpOsdSinterDecoder(sinter.Decoder):
    """Same parameters as ``ldpc.sinter_decoders.SinterBpOsdDecoder``."""

    def __init__(
        self,
        max_iter: int = 0,
        bp_method: str = "ms",
        ms_scaling_factor: float = 0.625,
        schedule: str = "parallel",
        omp_thread_count: int = 1,
        serial_schedule_order=None,
        osd_method: str = "osd0",
        osd_order: int = 0,
    ) -> None:
        self.max_iter = max_iter
        self.bp_method = bp_method
        self.ms_scaling_factor = ms_scaling_factor
        self.schedule = schedule
        self.omp_thread_count = omp_thread_count
        self.serial_schedule_order = serial_schedule_order
        self.osd_method = osd_method
        self.osd_order = osd_order

    def compile_decoder_for_dem(
        self,
        *,
        dem: stim.DetectorErrorModel,
    ) -> sinter.CompiledDecoder:
        return _CompiledCachedBpOsd(
            dem=dem,
            max_iter=self.max_iter,
            bp_method=self.bp_method,
            ms_scaling_factor=self.ms_scaling_factor,
            schedule=self.schedule,
            omp_thread_count=self.omp_thread_count,
            serial_schedule_order=self.serial_schedule_order,
            osd_method=self.osd_method,
            osd_order=self.osd_order,
        )
