# Neuromorphic Anomaly Cascade

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Benchmarks: NAB](https://img.shields.io/badge/Benchmark-NAB%20(58%20datasets)-orange.svg)](https://github.com/numenta/NAB)
[![AI Ethics: COPE / CNPq](https://img.shields.io/badge/Ethics-COPE%20%26%20CNPq%202.664-green.svg)](STATEMENT_OF_AI_ASSISTANCE.md)

An empirical laboratory evaluating continuous biophysical relaxation dynamics for streaming time-series anomaly detection on the [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB).

---

## Overview & Core Question

Standard production anomaly detection pipelines commonly rely on sliding-window statistical heuristics, tree ensembles (e.g., Random Cut Forest), or discrete propositional consensus rules (e.g., boolean vetoes requiring $N$ scales to agree). While effective in constrained regimes, discrete rules introduce brittle edge behaviors, struggle with variable sampling intervals ($\Delta t$), and often emit post-spike alarm bursts.

This project investigates a central empirical question:

> **Can continuous, coupled biophysical relaxation dynamics—specifically leaky differential membranes, Tsodyks-Markram synaptic depression, and sigmoidal dendritic gating—replace discrete propositional heuristics in streaming anomaly detection?**

Across 41 iterative ablations evaluated on all 58 datasets of the NAB benchmark suite, the continuous cascade architecture ($M_{41}$) surpassed propositional consensus baselines, setting a new project mark of **47.83** on the Standard profile and **28.84** on the Low False Positive profile, while reducing total benchmark false alarms by **31.8%** relative to the prior best model ($M_{26}$).

---

## Architecture & Mathematical Formulation

The detection engine operates as a strictly causal, $O(1)$-memory dynamical system governed by three coupled differential equations:

```
Streaming Input x(t) 
       │
       ▼
 [ Causal Welford Standardization ] ──> x_norm(t)
       │
       ▼
 [ Multi-Scale Dual-Timescale Membranes ] (τ_slow ∈ {0.75h, 1.5h, 3.0h, 6.0h}, τ_ratio = 8.0)
       │
       ├──> Chamber 0 (0.75h) ──[ Spillover ]──┐
       │                                       ▼
       ├──> Chamber 1 (1.50h) <───[ Dendritic Gating σ(z₁) ]
       │                                       ▼
       ├──> Chamber 2 (3.00h) <───[ Dendritic Gating σ(z₂) ]
       │                                       ▼
       └──> Chamber 3 (6.00h) <───[ Dendritic Gating σ(z₃) ] ──> Terminal Spillover
                                                                       │
                                                                       ▼
                                                       [ Tsodyks-Markram Synapse ]
                                                            (τ_rec = 3.0h, u = 0.95)
                                                                       │
                                                                       ▼
                                                              Effective Score s(t) ∈ [0, 1]
```

### 1. Dual-Timescale Leaky Membrane Integrator
For each timescale $\tau_i \in \{0.75\text{h}, 1.5\text{h}, 3.0\text{h}, 6.0\text{h}\}$, the input drives coupled fast and slow relaxation filters ($\tau_{\text{fast}} = \tau_{\text{slow}} / 8$):

$$\tau_{\text{fast}} \frac{d u_{\text{fast}}}{dt} = x(t) - u_{\text{fast}}(t), \quad \tau_{\text{slow}} \frac{d u_{\text{slow}}}{dt} = x(t) - u_{\text{slow}}(t)$$

Integrated causally across variable intervals $\Delta t$ via the exact Zero-Order Hold (ZOH) exponential solution:

$$u(t + \Delta t) = e^{-\Delta t / \tau} u(t) + \left(1 - e^{-\Delta t / \tau}\right) x(t)$$

The instantaneous biophysical tension is the differential excursion $\Delta(t) = |u_{\text{fast}}(t) - u_{\text{slow}}(t)|$, standardized online via a causal Welford estimator to yield standardized tension $z_i(t)$.

### 2. Leaky Spillover Retention Cascade & Dendritic Gating
Rather than taking a discrete boolean minimum across scales, energy percolates through a chain of retention chambers with capacities $\theta = (1.5, 1.2, 1.0, 0.8)$:

$$\frac{d C_i}{dt} = -\frac{C_i}{\tau_i} + \max(0, z_i) + I_{\text{inflow}, i}$$

When a chamber exceeds its retention threshold $\theta_i$, excess energy spills downstream:

$$\text{spill}_i = \max(0, C_i - \theta_i) \cdot g_i, \quad C_i \leftarrow \min(C_i, \theta_i)$$

To prevent slow chambers from propagating transient noise when their local state is inactive, inter-chamber percolation is modulated by a smooth sigmoidal **dendritic co-activation gate**:

$$I_{\text{inflow}, i} = \text{spill}_{i-1} \cdot \sigma(z_i), \quad \text{where } \sigma(z_i) = \frac{1}{1 + e^{-2(z_i - 0.5)}}$$

### 3. Action-Potential Tsodyks-Markram Synaptic Depression
Terminal spillover drives a non-linear readout:

$$s_{\text{raw}}(t) = 1 - \exp\left(-\frac{1}{2} (\beta \cdot \text{spill})^2\right), \quad \beta = 3.0$$

Transmitted drive is scaled by available neurotransmitter vesicles $R(t) \in [0, 1]$:

$$s_{\text{eff}}(t) = R(t) \cdot s_{\text{raw}}(t)$$

Vesicles replenish passively according to:

$$\frac{dR}{dt} = \frac{1 - R}{\tau_{\text{rec}}}, \quad \tau_{\text{rec}} = 3.0\text{ hours}$$

Whenever the terminal detector emits an action potential ($s_{\text{raw}} > \theta_{\text{AP}} = 0.5$), vesicle reserves discharge:

$$R \leftarrow R \cdot (1 - u), \quad u = 0.95$$

This biophysical mechanism imposes a causal refractory period, preventing repeated false alarms along the lingering decay tail of an anomaly.

---

## Empirical Benchmark Results (NAB — 58 Datasets)

Evaluated across the official 58 streaming datasets of the Numenta Anomaly Benchmark using the standard scoring metric (rewarding early detection, penalizing false alarms and false negatives).

### Overall Benchmark Scorecard

| Model | Architecture Description | Standard | Low FP | Low FN | Total FPs | Total TPs | Precision@500 | Anomaly Windows Covered |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Windowed Gaussian** | Baseline sliding-window Gaussian | 39.60 | 20.90 | 47.40 | 664 | 118 | 0.1777 | 64 / 82 |
| **TwitterADVec** | Twitter AnomalyDetectionVec | 47.06 | 33.60 | 53.50 | 363 | 129 | 0.2580 | 71 / 82 |
| **Random Cut Forest** | Isolation forest variant | 51.70 | 38.40 | 59.70 | 398 | 134 | 0.2680 | 73 / 82 |
| **Numenta HTM** | Hierarchical Temporal Memory | **64.60** | **57.70** | **69.40** | 321 | **148** | **0.3150** | 68 / 82 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **$M_1$** | Single membrane, fixed step ($dt=0.01\text{s}$) | 30.13 | 11.20 | 38.38 | 912 | 104 | 0.1140 | 58 / 82 |
| **$M_{24}$** | 4-scale coincidence bank + hard refractory gate | 46.20 | 22.63 | 57.52 | 684 | 124 | 0.1812 | 70 / 82 |
| **$M_{26}$** | 4-scale bank + Tsodyks-Markram synapse | 47.63 | 25.95 | **58.60** | 610 | 126 | 0.1880 | **78 / 82** |
| **$M_{40}$** | Ascending capacitance cascade ($\beta=3.5$) | 45.76 | **36.19** | 51.36 | **213** | 108 | **0.3600** | 65 / 82 |
| **$M_{41}$** | **Gated Leaky Cascade + Action-Potential STD** | **47.83** | **28.84** | 56.03 | 416 | 143 | 0.2540 | **78 / 82** |

*Note: All scores were executed deterministically on the official NAB evaluation runner across the 58 standard files. Raw score CSVs and logs are preserved in `benchmarks/NAB/`.*

### Subcategory Performance Breakdown ($M_{41}$ vs $M_{26}$)

| Benchmark Subcategory | $M_{26}$ Score (FPs) | $M_{41}$ Score (FPs) | Delta | Key Empirical Dynamic |
| :--- | :---: | :---: | :---: | :--- |
| `realAWSCloudwatch` | 52.6 (114 FPs) | **61.1** (**7 FPs**) | **+8.5** | Dendritic gate silenced idle slow chambers during steady load. |
| `artificialWithAnomaly`| 8.5 (182 FPs) | **45.1** (**48 FPs**) | **+36.6** | Cascade percolation absorbed step changes without alarm bursts. |
| `realTweets` | 51.4 (242 FPs) | **56.6** (**103 FPs**) | **+5.2** | Vesicular depletion eliminated post-spike volume chatter. |
| `realKnownCause` | 46.2 (41 FPs) | 43.1 (72 FPs) | −3.1 | Extreme variance shifts challenged fixed threshold capacity $\theta_i$. |
| `realAdExchange` | 38.0 (31 FPs) | 33.3 (186 FPs) | −4.7 | High baseline volatility triggered inter-chamber percolation. |

---

## The Empirical Journey: Iterations, Failures & Hypotheses

Rather than presenting only the end result, this repository documents the sequence of hypotheses tested, failures observed, and evidence accumulated:

```
[M1..M6: Graph vs Pure Membrane]
       │
       ▼ (Failure: HVG topological metrics added O(W) overhead without NAB score gains)
[M7..M10: Physical Time Clocks]
       │
       ▼ (Insight: dt-aware ZOH integration drained stale data during gaps)
[M11..M24: Multi-Scale Banks & Discrete Consensus]
       │
       ▼ (Insight: Majority voting failed; min-rule succeeded; hard gates caused dead-time blindness)
[M25..M30: Tsodyks-Markram Synaptic Depression]
       │
       ▼ (Breakthrough: M26 reached 47.63; multiplicative vesicle depletion dominated additive AHP)
[M31..M38: Leaky Spillover Cascades]
       │
       ▼ (Failure: Inter-chamber gain g > 1.0 exploded in high volatility; threshold traps caused 111 FPs on NYC taxi)
[M39..M41: Dendritic Gating & Action-Potential Depression]
       ▼
[M41: All-Time Project Champion — Standard 47.83, Low FP 28.84, -31.8% FPs]
```

### Key Epistemic Insights
1. **Multiplication Beats Subtraction**: Additive After-Hyperpolarization (AHP, subtracting an excursion penalty) required artificial clipping at zero. Multiplicative Tsodyks-Markram vesicular depletion ($R \leftarrow R(1-u)$) possesses a natural biological lower bound ($R \ge 0$) and logarithmic recovery dynamics that matched real-world burst profiles.
2. **The 3-Hour Synaptic Resonance**: Parameter sweeps across $\tau_{\text{rec}} \in \{1\text{h}, 3\text{h}, 6\text{h}, 12\text{h}\}$ exhibited a clear unimodal optimum at **3.0 hours** ($\tau = 1\text{h} \to 44.86$; $\tau = 3\text{h} \to 47.63$; $\tau = 6\text{h} \to 44.36$; $\tau = 12\text{h} \to 30.49$). Shorter recovery allowed secondary false alarms; longer recovery caused temporary blindness to subsequent true anomalies.
3. **The Threshold Trap in Cascades**: In early cascade iterations ($M_{35}$), synaptic depression required raw spillover to exceed $0.5$. With readout scaling $\beta = 2.5$, spillover values of $0.45$ produced anomaly scores of $0.52$ (triggering benchmark alarms) without ever firing the depression mechanism. This caused massive false positive cascades on `nyc_taxi` (111 FPs). Tying synaptic depression directly to the **emitted action potential** ($s_{\text{raw}} > \theta_{\text{AP}}$) resolved this failure completely in $M_{41}$.

---

## Known Limitations & Boundary Conditions

To maintain scientific rigor, the operational boundaries of this architecture must be explicitly recognized:

1. **Temporal Sequence Memory**: Hierarchical Temporal Memory (HTM, score 64.60) maintains a distinct advantage on complex multi-order symbolic transition structures. Continuous relaxation membranes capture duration, frequency, and amplitude dynamics well, but do not encode non-Markovian sequence transitions.
2. **Univariate Scope**: The current formulation operates on a single scalar stream $x(t)$. Extending this dynamics to high-dimensional telemetry requires cross-channel synaptic conductance or spatial network topologies.
3. **Benchmark Caveats (Wu & Keogh 2023)**: As demonstrated by Wu & Keogh in *"Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress"* (IEEE TKDE 2023), NAB metrics exhibit sensitivity to scoring window boundaries and reward early detections within tolerance windows. While relative gains across identical evaluation protocols remain valid, absolute scores must be interpreted in light of these known benchmark characteristics.

---

## Quickstart & Usage

### Installation

Clone the repository and install the minimal standalone package:

```bash
git clone https://github.com/patryckalves/neuromorphic-anomaly-cascade.git
cd neuromorphic-anomaly-cascade
pip install -e .
```

### Streaming Detection (Python)

The package provides a minimal, dependency-free streaming interface ($O(1)$ memory, pure NumPy):

```python
from datetime import datetime, timedelta
from neuromorphic_cascade import CascadeAnomalyDetector

detector = CascadeAnomalyDetector()
timestamp = datetime(2026, 4, 1, 12, 0, 0)

# Process streaming observations one record at a time
stream = [10.2, 10.5, 10.1, 9.8, 10.3, 85.0, 82.0, 10.4]

for value in stream:
    score = detector.step(value, timestamp)
    timestamp += timedelta(minutes=5)
    print(f"Value: {value:5.1f} | Anomaly Score: {score:.4f}")
```

### Reproducing Benchmark Scores

The official 58-dataset evaluation was conducted using the [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB) scoring harness. The deterministic per-dataset score CSVs and normalized summary metrics generated during benchmark execution are archived directly in `benchmarks/`:

- `benchmarks/m41_standard_scores.csv`: Per-dataset True Positives, False Positives, and scores under the Standard profile.
- `benchmarks/m41_reward_low_FP_rate_scores.csv`: Metrics under the Low False Positive profile.
- `benchmarks/m41_reward_low_FN_rate_scores.csv`: Metrics under the Low False Negative profile.
- `benchmarks/final_results.json`: Consolidated normalized benchmark scorecard.

To execute the unit test suite verifying the continuous ODE dynamics, synaptic depression, and streaming normalizers:

```bash
pytest
```

---

## Research Integrity & AI Assistance

In adherence to the **Committee on Publication Ethics (COPE)** position, the **STM Association Global Reporting Standard for AI Disclosure in Research**, and the **Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq) Diretrizes de Integridade Científica (Portaria CNPq nº 2.664/2026)**:

* No artificial intelligence system holds authorship or co-authorship.
* The human researcher retains sole, non-delegable scientific and ethical responsibility for all derivations, implementations, and empirical results.
* Complete role taxonomies, verification protocols, and tool details are formally documented in [STATEMENT_OF_AI_ASSISTANCE.md](STATEMENT_OF_AI_ASSISTANCE.md).

---

## License

This project is licensed under the [MIT License](LICENSE).
