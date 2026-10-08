"""
Anomaly Detectors
=================

High-level, self-contained streaming anomaly detectors.
Pure Python and NumPy. Zero dependencies on heavy frameworks.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Sequence, Union
import numpy as np
import pandas as pd

from .dynamics import (
    DualTimescaleMembrane,
    TsodyksMarkramSynapse,
    dendritic_coactivation_gate,
)
from .normalizer import WelfordNormalizer

TimestampType = Union[float, int, datetime, np.datetime64, None]


def _to_seconds(ts: TimestampType, prev_ts: TimestampType) -> float | None:
    if prev_ts is None or ts is None:
        return None
    try:
        if isinstance(ts, (datetime, pd.Timestamp)) and isinstance(prev_ts, (datetime, pd.Timestamp)):
            return float((ts - prev_ts).total_seconds())
        return float(ts) - float(prev_ts)
    except Exception:
        return None


class CascadeAnomalyDetector:
    """
    M41: Co-Activation Dendritic Gated Leaky Spillover Cascade.

    Replaces discrete propositional consensus (boolean min-rules) with a continuous
    multi-stage hierarchy of leaky retention chambers and inter-chamber percolation.

    Architecture:
    1. Multi-scale dual-timescale membranes (tau_slow in [0.75h, 1.5h, 3.0h, 6.0h], tau_ratio = 8.0).
    2. Causal tension normalization via streaming Welford estimators.
    3. Coupled leaky retention chambers with continuous spillover transfer:
       spill_i = max(0, chamber_i - theta_i) * gain_i.
    4. Non-linear dendritic co-activation gating:
       inflow_i = spill_{i-1} * sigma(z_i), preventing quiescent chambers from passing energy.
    5. Action-potential-triggered Tsodyks-Markram synaptic depression (tau_rec = 3.0h, u = 0.95),
       eliminating post-spike alarm bursts.
    """

    def __init__(
        self,
        scales_hours: Sequence[float] = (0.75, 1.5, 3.0, 6.0),
        tau_ratio: float = 8.0,
        thetas: Sequence[float] = (1.5, 1.2, 1.0, 0.8),
        gains: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
        beta_readout: float = 3.0,
        synaptic_recovery_hours: float = 3.0,
        synaptic_u: float = 0.95,
        action_potential_threshold: float = 0.5,
        coactivation_gate: bool = True,
        probationary_period: int = 750,
    ):
        self.scales_hours = tuple(float(s) for s in scales_hours)
        self.tau_ratio = float(tau_ratio)
        self.thetas = tuple(float(th) for th in thetas)
        self.gains = tuple(float(g) for g in gains)
        self.beta_readout = float(beta_readout)
        self.synaptic_recovery_hours = float(synaptic_recovery_hours)
        self.synaptic_u = float(synaptic_u)
        self.action_potential_threshold = float(action_potential_threshold)
        self.coactivation_gate = bool(coactivation_gate)
        self.probationary_period = int(probationary_period)

        self._reset_state()

    def _reset_state(self) -> None:
        self.input_normalizer = WelfordNormalizer(eps=1e-6, clip_val=5.0)
        self.membranes = [
            DualTimescaleMembrane(
                tau_fast_seconds=(tau_h / self.tau_ratio) * 3600.0,
                tau_slow_seconds=tau_h * 3600.0,
            )
            for tau_h in self.scales_hours
        ]
        self.tension_normalizers = [
            WelfordNormalizer(eps=1e-6, clip_val=10.0) for _ in self.scales_hours
        ]
        self.chambers = [0.0] * len(self.scales_hours)
        self.synapse = TsodyksMarkramSynapse(
            tau_rec_seconds=self.synaptic_recovery_hours * 3600.0,
            u=self.synaptic_u,
        )
        self.record_count = 0
        self._prev_ts: TimestampType = None

    def step(self, value: float, timestamp: TimestampType = None) -> float:
        """
        Processes one scalar streaming observation.

        Parameters
        ----------
        value : float
            Raw numerical observation.
        timestamp : datetime, int, float, or None
            Physical timestamp. If None or non-positive interval, uses unit interval.

        Returns
        -------
        score : float
            Continuous anomaly score in [0, 1].
        """
        raw_val = float(value)
        self.record_count += 1

        dt = _to_seconds(timestamp, self._prev_ts)
        self._prev_ts = timestamp

        # Passive vesicular recovery
        self.synapse.recover(dt)

        # Standardize input causally
        x_norm = self.input_normalizer.update_and_normalize(raw_val)

        # Step each dual-timescale membrane and obtain standardized tension
        zs: list[float] = []
        for membrane, normalizer in zip(self.membranes, self.tension_normalizers):
            _, _, delta = membrane.step(x_norm, dt)
            mu_delta, std_delta = normalizer.update(delta)
            z = (delta - mu_delta) / max(std_delta, 1e-6)
            zs.append(z)

        # Dynamic inter-chamber leaky spillover cascade
        spill = 0.0
        if dt is not None and dt > 0.0:
            # First chamber (fastest scale)
            tau_s0 = self.scales_hours[0] * 3600.0
            decay0 = math.exp(-dt / tau_s0)
            z0_pos = max(0.0, zs[0])
            self.chambers[0] = self.chambers[0] * decay0 + (1.0 - decay0) * z0_pos
            spill = max(0.0, self.chambers[0] - self.thetas[0]) * self.gains[0]
            self.chambers[0] = min(self.chambers[0], self.thetas[0])

            # Downstream chambers
            for i in range(1, len(self.scales_hours)):
                tau_si = self.scales_hours[i] * 3600.0
                decayi = math.exp(-dt / tau_si)
                zi_pos = max(0.0, zs[i])
                inflow = spill
                if self.coactivation_gate:
                    gate = dendritic_coactivation_gate(zi_pos, center=0.5, slope=2.0)
                    inflow = inflow * gate

                self.chambers[i] = self.chambers[i] * decayi + (1.0 - decayi) * zi_pos + inflow
                spill = max(0.0, self.chambers[i] - self.thetas[i]) * self.gains[i]
                self.chambers[i] = min(self.chambers[i], self.thetas[i])
        else:
            spill = 0.0

        if self.record_count <= self.probationary_period:
            return 0.0

        # Calibrated non-linear terminal readout
        scaled_drive = self.beta_readout * max(0.0, spill)
        s_raw = 1.0 - math.exp(-0.5 * (scaled_drive ** 2))
        s_eff = self.synapse.resource * s_raw

        # Action-potential-triggered vesicular depletion
        if s_raw > self.action_potential_threshold:
            self.synapse.discharge()

        return float(np.clip(s_eff, 0.0, 1.0))

    def predict_series(
        self,
        values: Sequence[float],
        timestamps: Sequence[TimestampType] | None = None,
    ) -> np.ndarray:
        """Processes an entire time series iteratively, maintaining causal state."""
        self._reset_state()
        n = len(values)
        scores = np.zeros(n, dtype=np.float64)
        has_ts = timestamps is not None and len(timestamps) == n

        for idx, val in enumerate(values):
            ts = timestamps[idx] if has_ts else None
            scores[idx] = self.step(val, ts)

        return scores


class MembraneAnomalyDetector:
    """
    M26: Multi-Scale Coincidence Bank with Tsodyks-Markram Synaptic Depression.

    Applies the minimum-agreement rule across 4 temporal scales coupled with
    vesicular depression (tau_rec = 3.0h, u = 0.95).
    """

    def __init__(
        self,
        scales_hours: Sequence[float] = (0.75, 1.5, 3.0, 6.0),
        tau_ratio: float = 8.0,
        synaptic_recovery_hours: float = 3.0,
        synaptic_u: float = 0.95,
        probationary_period: int = 750,
    ):
        self.scales_hours = tuple(float(s) for s in scales_hours)
        self.tau_ratio = float(tau_ratio)
        self.synaptic_recovery_hours = float(synaptic_recovery_hours)
        self.synaptic_u = float(synaptic_u)
        self.probationary_period = int(probationary_period)
        self._reset_state()

    def _reset_state(self) -> None:
        self.input_normalizer = WelfordNormalizer(eps=1e-6, clip_val=5.0)
        self.membranes = [
            DualTimescaleMembrane(
                tau_fast_seconds=(tau_h / self.tau_ratio) * 3600.0,
                tau_slow_seconds=tau_h * 3600.0,
            )
            for tau_h in self.scales_hours
        ]
        self.tension_normalizers = [
            WelfordNormalizer(eps=1e-6, clip_val=10.0) for _ in self.scales_hours
        ]
        self.synapse = TsodyksMarkramSynapse(
            tau_rec_seconds=self.synaptic_recovery_hours * 3600.0,
            u=self.synaptic_u,
        )
        self.record_count = 0
        self._prev_ts: TimestampType = None

    def step(self, value: float, timestamp: TimestampType = None) -> float:
        raw_val = float(value)
        self.record_count += 1

        dt = _to_seconds(timestamp, self._prev_ts)
        self._prev_ts = timestamp

        self.synapse.recover(dt)
        x_norm = self.input_normalizer.update_and_normalize(raw_val)

        zs: list[float] = []
        for membrane, normalizer in zip(self.membranes, self.tension_normalizers):
            _, _, delta = membrane.step(x_norm, dt)
            mu_delta, std_delta = normalizer.update(delta)
            z = (delta - mu_delta) / max(std_delta, 1e-6)
            zs.append(z)

        if self.record_count <= self.probationary_period:
            return 0.0

        z_min = min(zs)
        drive = max(0.0, z_min)
        s_raw = 1.0 - math.exp(-0.5 * (drive ** 2))
        s_eff = self.synapse.resource * s_raw

        if drive > 1.0:
            self.synapse.discharge()

        return float(np.clip(s_eff, 0.0, 1.0))

    def predict_series(
        self,
        values: Sequence[float],
        timestamps: Sequence[TimestampType] | None = None,
    ) -> np.ndarray:
        self._reset_state()
        n = len(values)
        scores = np.zeros(n, dtype=np.float64)
        has_ts = timestamps is not None and len(timestamps) == n

        for idx, val in enumerate(values):
            ts = timestamps[idx] if has_ts else None
            scores[idx] = self.step(val, ts)

        return scores
