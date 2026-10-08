"""
Neuromorphic Anomaly Cascade (NAC)
==================================

Continuous biophysical relaxation dynamics for streaming time series anomaly detection.

Evaluated on the Numenta Anomaly Benchmark (NAB).
"""

from .detector import CascadeAnomalyDetector, MembraneAnomalyDetector
from .dynamics import (
    DualTimescaleMembrane,
    TsodyksMarkramSynapse,
    dendritic_coactivation_gate,
)
from .normalizer import WelfordNormalizer, TimeWeightedNormalizer

__all__ = [
    "CascadeAnomalyDetector",
    "MembraneAnomalyDetector",
    "DualTimescaleMembrane",
    "TsodyksMarkramSynapse",
    "dendritic_coactivation_gate",
    "WelfordNormalizer",
    "TimeWeightedNormalizer",
]

__version__ = "0.1.0"
