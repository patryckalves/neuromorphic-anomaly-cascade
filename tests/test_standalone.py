"""
Unit tests for neuromorphic_cascade standalone package.
"""

from datetime import datetime, timedelta
import math
import numpy as np
import pytest

from neuromorphic_cascade import (
    CascadeAnomalyDetector,
    MembraneAnomalyDetector,
    DualTimescaleMembrane,
    TsodyksMarkramSynapse,
    WelfordNormalizer,
    dendritic_coactivation_gate,
)


def test_welford_normalizer():
    norm = WelfordNormalizer()
    assert norm.normalize(10.0) == 0.0

    # Feed sequence with known mean=5.0, var ~ 2.5
    data = [4.0, 5.0, 6.0, 4.0, 5.0, 6.0]
    for x in data:
        norm.update(x)

    assert abs(norm.mean - 5.0) < 1e-4
    assert norm.std > 0.0
    z = norm.normalize(7.0)
    assert z > 0.0


def test_dual_timescale_membrane():
    m = DualTimescaleMembrane(tau_fast_seconds=60.0, tau_slow_seconds=600.0)
    # Initial step should initialize without delta
    u_f, u_s, d = m.step(10.0)
    assert u_f == 10.0
    assert u_s == 10.0
    assert d == 0.0

    # Step with sudden jump
    u_f, u_s, d = m.step(50.0, dt=10.0)
    # Fast membrane reacts faster than slow membrane
    assert u_f > u_s
    assert d > 0.0


def test_tsodyks_markram_synapse():
    syn = TsodyksMarkramSynapse(tau_rec_seconds=3600.0, u=0.95)
    assert syn.resource == 1.0

    # Discharge
    syn.discharge()
    assert syn.resource < 0.1

    # Recover over 1 hour
    syn.recover(3600.0)
    assert syn.resource > 0.6


def test_dendritic_coactivation_gate():
    # Quiescent state (< 0.5) should yield small transmission
    low_gate = dendritic_coactivation_gate(0.0)
    assert low_gate < 0.3

    # High state (> 1.0) should open gate
    high_gate = dendritic_coactivation_gate(2.0)
    assert high_gate > 0.9


def test_cascade_anomaly_detector_streaming():
    det = CascadeAnomalyDetector(probationary_period=10)
    base_time = datetime(2026, 1, 1, 0, 0, 0)

    # Normal sinusoidal baseline
    for i in range(50):
        t = base_time + timedelta(minutes=5 * i)
        val = 10.0 + math.sin(i * 0.1)
        score = det.step(val, t)
        assert 0.0 <= score <= 1.0

    # Massive anomaly pulse
    t_pulse = base_time + timedelta(minutes=5 * 51)
    spike_score = det.step(100.0, t_pulse)
    assert 0.0 <= spike_score <= 1.0


def test_cascade_anomaly_detector_predict_series():
    det = CascadeAnomalyDetector(probationary_period=5)
    ts = [datetime(2026, 1, 1) + timedelta(minutes=5 * i) for i in range(30)]
    vals = [10.0] * 30
    vals[20] = 50.0  # Anomaly

    scores = det.predict_series(vals, ts)
    assert len(scores) == 30
    assert (scores >= 0.0).all() and (scores <= 1.0).all()
