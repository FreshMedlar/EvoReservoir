# Hybrid Reservoir: Differences from a Standard ESN

This document lists every architectural deviation of the
[HybridReservoir](file:///home/medlar/Projects/EvoReservoir/hybrid_reservoir/model.py)
from a textbook Echo State Network (ESN).

---

## 1. Augmented Readout Features

| Standard ESN | Hybrid Reservoir |
|---|---|
| Readout: $\hat{y} = W_{out} \cdot x(t)$ | Readout: $\hat{y} = W_{out} \cdot [1,\; u(t),\; x(t),\; x(t)^2]$ |

The feature vector fed to the linear readout is augmented with:
- **A bias term** (constant 1), eliminating the need for a separate bias parameter.
- **The raw input** $u(t)$, giving the readout a direct input-to-output pathway.
- **Squared states** $x(t)^2$, providing second-order nonlinear features without additional hidden layers.

This roughly doubles the feature dimension but allows the linear solver to capture richer input-output mappings.

---

## 2. Per-Neuron Heterogeneous Leaking Rates

| Standard ESN | Hybrid Reservoir |
|---|---|
| Single global leaking rate $\alpha$ | Per-neuron $\alpha_i$ drawn log-uniformly from $[\alpha_{lo}, \alpha_{hi}]$ |

The update rule becomes:

$$x_i(t) = (1 - \alpha_i)\, x_i(t-1) \;+\; \alpha_i\, \tanh(\ldots)$$

Neurons with small $\alpha_i$ act as slow integrators (long memory), while neurons with large $\alpha_i$ respond quickly to input changes. This creates a **multi-timescale reservoir** that simultaneously captures fast transients and slow dynamics without needing to tune a single global parameter.

---

## 3. Learned State Feedback (Ehlers et al.)

| Standard ESN | Hybrid Reservoir |
|---|---|
| Fixed recurrent matrix $W$ | Effective matrix $\bar{A} = W + W_{in} \cdot V^T$ with learned $V$ |

A feedback matrix $V \in \mathbb{R}^{N_{res} \times d_{in}}$ is trained via gradient descent to modify the effective recurrent dynamics. The optimisation includes a **spectral-norm stability constraint**: $\|\bar{A}\|_2 < \rho$, enforced via projected gradient descent with backtracking line search.

This is based on: *"Improving the Performance of Echo State Networks Through State Feedback"* by Ehlers, Nurdin & Soh.

Training phases:
1. Plasticity run (self-organise topology).
2. Gradient descent on $V$ (multiple passes, re-fitting readout each step).
3. Final readout fit with optimal $V$.

---

## 4. Excitatory-Inhibitory (E-I) Constraints (Dale's Principle)

| Standard ESN | Hybrid Reservoir |
|---|---|
| Mixed-sign weights per neuron | Columns partitioned: 80% excitatory (≥ 0), 20% inhibitory (≤ 0) |

The recurrent weight matrix enforces biological Dale's principle: excitatory neurons have only non-negative outgoing weights, inhibitory neurons have only non-positive outgoing weights. This constraint is maintained throughout plasticity operations via clamping.

---

## 5. Input Synaptic Decay Filter

| Standard ESN | Hybrid Reservoir |
|---|---|
| Input applied directly: $W_{in} \cdot u(t)$ | Filtered input: $f(t) = (1-\gamma)\, f(t-1) + \gamma\, u(t)$ |

A first-order exponential moving average is applied to the input before projection. When $\gamma < 1$, this creates temporal smoothing that retains traces of recent inputs, adding temporal depth to the input representation without spike encoding overhead.

---

## 6. Homeostatic Gain Control

| Standard ESN | Hybrid Reservoir |
|---|---|
| Fixed gain (no scaling) | Per-neuron adaptive scaling $s_i \in [0.1, 5.0]$ |

Each neuron has a scaling factor applied before `tanh`:

$$\tilde{x}(t) = \tanh\bigl(s_i \cdot (x_{rec} + x_{in})\bigr)$$

The scaling is adjusted based on a slow trailing average of activation magnitudes:
- **Low activity** ($\bar{|x|} < 0.15$): increase gain by 0.01 per step.
- **High activity** ($\bar{|x|} > 0.85$): decrease gain by 0.01 per step.
- **Normal range**: reset to 1.0.

This prevents neurons from becoming permanently saturated or silent.

---

## 7. Structural Plasticity (Genesis / Pruning)

| Standard ESN | Hybrid Reservoir |
|---|---|
| Static weights | Dynamic weight creation and removal during training |

During the plasticity phase:
- **Genesis**: Neurons with very low trailing activity ($< 0.10$) gain a new random incoming connection (+0.1 / −0.1 respecting E-I sign).
- **Pruning**: Neurons with very high trailing activity ($> 0.90$) have a random incoming connection weakened.
- Optional **spontaneous genesis** at a configurable random rate.

Plasticity is **frozen** before readout fitting and during inference.

---

## 8. Phase-Locked Training Pipeline

| Standard ESN | Hybrid Reservoir |
|---|---|
| Single pass: harvest states → fit readout | Three-phase: plasticity → feedback → readout |

1. **Phase 1 (Plasticity)**: Run the full training sequence with evolution enabled to self-organise the reservoir topology.
2. **Phase 2 (Feedback)**: Freeze plasticity, train feedback matrix $V$ via gradient descent (optional, configurable number of steps).
3. **Phase 3 (Readout)**: Fit readout weights via centred ridge regression on augmented features.

---

## 9. Centred Ridge Regression

| Standard ESN | Hybrid Reservoir |
|---|---|
| Standard ridge: $W = YX^T(XX^T + \mu I)^{-1}$ | Centred ridge: subtract means before solving, compute bias separately |

The readout fitting centres both features and targets before solving. This produces a separate bias term $C_{out}$ and is more numerically stable when feature distributions are non-zero-mean.
