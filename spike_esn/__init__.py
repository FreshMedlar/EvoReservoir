"""
Spike Echo State Network (Spike-ESN) Implementation
=====================================================

Brain-Inspired Spike Echo State Network Dynamics for Intelligent Fault Prediction.

Based on the paper by Mo-Ran Liu, Tao Sun, and Xi-Ming Sun:
"Brain-Inspired Spike Echo State Network Dynamics for Aero-engine
 Intelligent Fault Prediction" (IEEE).

The Spike-ESN model consists of three components:
  1. Spike Input Layer  - Poisson-based spike encoding of time series data
  2. Spike Reservoir    - High-dimensional sparse nonlinear projection
  3. Output Layer       - Ridge regression readout

Usage:
    from spike_esn import SpikeESN
    model = SpikeESN(N_res=100, N_sam=100, rho=0.9, eta=0.1, mu=1e-8, psi=5000)
    model.fit(u_train, y_train, washout=200)
    y_pred = model.predict(u_test)
"""

from .model import SpikeESN
from .spike_encoding import SpikeEncoder
from .reservoir import SpikeReservoir

__all__ = ["SpikeESN", "SpikeEncoder", "SpikeReservoir"]
__version__ = "1.0.0"
