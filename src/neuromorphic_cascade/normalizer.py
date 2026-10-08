"""
Streaming Normalizers
====================

Strictly causal, zero-lookahead online estimators of running mean and variance.
Maintains O(1) memory and computation per timestep.
"""

from __future__ import annotations

import math
import numpy as np


class WelfordNormalizer:
    """
    Causal streaming running mean and variance estimation via Welford's algorithm (1962).

    Strictly zero lookahead: maintains running state solely over past observed values.
    Never relies on global dataset extrema (min/max).
    """

    def __init__(self, eps: float = 1e-6, clip_val: float = 5.0):
        self.eps = float(eps)
        self.clip_val = float(clip_val)
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, val: float) -> tuple[float, float]:
        """
        Updates running statistics with a new scalar observation.
        Returns current (running_mean, running_std).
        """
        val_f = float(val)
        self.count += 1
        delta = val_f - self.mean
        self.mean += delta / self.count
        delta2 = val_f - self.mean
        self.M2 += delta * delta2
        var = self.M2 / max(1, self.count - 1) if self.count > 1 else 0.0
        std = float(math.sqrt(max(0.0, var)))
        return self.mean, std

    @property
    def variance(self) -> float:
        return self.M2 / max(1, self.count - 1) if self.count > 1 else 0.0

    @property
    def std(self) -> float:
        return float(math.sqrt(max(0.0, self.variance)))

    def normalize(self, val: float) -> float:
        """
        Standardizes value against current running statistics and clips to [-clip_val, clip_val].
        Returns 0.0 when fewer than 2 records have been observed.
        """
        if self.count < 2:
            return 0.0
        s = self.std
        denom = s if s > self.eps else self.eps
        z = (float(val) - self.mean) / denom
        return float(np.clip(z, -self.clip_val, self.clip_val))

    def update_and_normalize(self, val: float) -> float:
        """Atomic streaming update and standardization."""
        self.update(val)
        return self.normalize(val)

    def reset(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0


class TimeWeightedNormalizer:
    """
    Online mean and variance estimator whose memory is a continuous physical time constant,
    rather than a discrete sample count.

    Each record carries weight alpha = 1 - exp(-dt / tau).
    """

    def __init__(self, tau_seconds: float, eps: float = 1e-6, clip_val: float = 10.0):
        self.tau = float(tau_seconds)
        self.eps = float(eps)
        self.clip_val = float(clip_val)
        self.mean = 0.0
        self.var = 0.0
        self.weight = 0.0
        self.started = False

    @property
    def std(self) -> float:
        return float(math.sqrt(max(0.0, self.var)))

    def update(self, val: float, dt_seconds: float | None = None) -> tuple[float, float]:
        x = float(val)
        if dt_seconds is None or float(dt_seconds) <= 0.0:
            return self.mean, self.std

        alpha = 1.0 - math.exp(-float(dt_seconds) / self.tau)

        if not self.started:
            self.mean = x
            self.var = 0.0
            self.weight = alpha
            self.started = True
            return self.mean, self.std

        d = x - self.mean
        self.mean += alpha * d
        self.var = (1.0 - alpha) * self.var + alpha * d * (x - self.mean)
        self.weight = (1.0 - alpha) * self.weight + alpha
        return self.mean, self.std

    def normalize(self, val: float) -> float:
        if not self.started:
            return 0.0
        s = self.std
        denom = s if s > self.eps else self.eps
        z = (float(val) - self.mean) / denom
        return float(np.clip(z, -self.clip_val, self.clip_val))

    def update_and_normalize(self, val: float, dt_seconds: float | None = None) -> float:
        self.update(val, dt_seconds)
        return self.normalize(val)

    def reset(self) -> None:
        self.mean = 0.0
        self.var = 0.0
        self.weight = 0.0
        self.started = False
