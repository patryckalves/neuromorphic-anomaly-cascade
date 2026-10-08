"""
Continuous Biophysical Dynamics
===============================

Differential equations governing membrane relaxation, Tsodyks-Markram short-term
synaptic depression, and sigmoidal dendritic co-activation gating.

All continuous equations are integrated causally with exact zero-order hold (ZOH)
exponential solutions for variable sampling intervals dt.
"""

from __future__ import annotations

import math


class DualTimescaleMembrane:
    """
    Continuous dual-timescale leaky membrane integrator.

    Solves the coupled relaxation differential equations:
        tau_fast * d(u_fast)/dt = x(t) - u_fast(t)
        tau_slow * d(u_slow)/dt = x(t) - u_slow(t)

    Using exact zero-order hold (ZOH) exponential integration over variable dt:
        u(t + dt) = exp(-dt / tau) * u(t) + (1 - exp(-dt / tau)) * x(t)

    The instantaneous biophysical tension is the differential excursion:
        Delta(t) = |u_fast(t) - u_slow(t)|
    """

    def __init__(self, tau_fast_seconds: float, tau_slow_seconds: float):
        if tau_fast_seconds <= 0 or tau_slow_seconds <= 0:
            raise ValueError("Time constants must be strictly positive.")
        if tau_fast_seconds >= tau_slow_seconds:
            raise ValueError("tau_fast must be strictly smaller than tau_slow.")

        self.tau_fast = float(tau_fast_seconds)
        self.tau_slow = float(tau_slow_seconds)
        self.u_fast = 0.0
        self.u_slow = 0.0
        self.initialized = False

    def step(self, x: float, dt: float | None = None) -> tuple[float, float, float]:
        """
        Advances the membrane by one continuous interval dt (in seconds).
        Returns (u_fast, u_slow, delta_tension = |u_fast - u_slow|).
        """
        x_val = float(x)
        if not self.initialized:
            self.u_fast = x_val
            self.u_slow = x_val
            self.initialized = True
            return self.u_fast, self.u_slow, 0.0

        if dt is None or dt <= 0.0:
            return self.u_fast, self.u_slow, abs(self.u_fast - self.u_slow)

        decay_fast = math.exp(-float(dt) / self.tau_fast)
        decay_slow = math.exp(-float(dt) / self.tau_slow)

        self.u_fast = decay_fast * self.u_fast + (1.0 - decay_fast) * x_val
        self.u_slow = decay_slow * self.u_slow + (1.0 - decay_slow) * x_val

        delta = abs(self.u_fast - self.u_slow)
        return self.u_fast, self.u_slow, delta

    def reset(self) -> None:
        self.u_fast = 0.0
        self.u_slow = 0.0
        self.initialized = False


class TsodyksMarkramSynapse:
    """
    Biophysical short-term synaptic depression (Tsodyks & Markram, 1997).

    Models neurotransmitter vesicle depletion and passive recovery:
        dR/dt = (1 - R) / tau_rec

    Upon activation / action-potential discharge:
        R <- R * (1 - u)

    The effective transmission is multiplicatively scaled by the available resource R in [0, 1]:
        s_eff = R * s_raw
    """

    def __init__(self, tau_rec_seconds: float, u: float = 0.95):
        if tau_rec_seconds <= 0:
            raise ValueError("tau_rec must be strictly positive.")
        if not (0.0 < u <= 1.0):
            raise ValueError("Depletion fraction u must be in (0, 1].")

        self.tau_rec = float(tau_rec_seconds)
        self.u = float(u)
        self.resource = 1.0

    def recover(self, dt: float | None) -> float:
        """
        Applies passive vesicular replenishment over elapsed time dt.
        Returns the updated resource fraction R in [0, 1].
        """
        if dt is not None and dt > 0.0:
            decay = math.exp(-float(dt) / self.tau_rec)
            self.resource = 1.0 - (1.0 - self.resource) * decay
        return self.resource

    def discharge(self) -> float:
        """
        Discharges neurotransmitter vesicles upon action-potential trigger.
        Returns the depleted resource fraction.
        """
        self.resource = max(0.0, self.resource * (1.0 - self.u))
        return self.resource

    def reset(self) -> None:
        self.resource = 1.0


def dendritic_coactivation_gate(z: float, center: float = 0.5, slope: float = 2.0) -> float:
    """
    Sigmoidal dendritic co-activation gating function.

    Modulates inter-chamber percolation so that receiving chambers only accept
    spillover energy if their local physiological state is non-quiescent:
        sigma(z) = 1 / (1 + exp(-slope * (z - center)))
    """
    z_val = float(z)
    arg = -float(slope) * (z_val - float(center))
    # Bound argument to prevent numerical overflow
    arg = max(-50.0, min(50.0, arg))
    return 1.0 / (1.0 + math.exp(arg))
