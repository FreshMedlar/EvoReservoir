"""
Spike Echo State Network (Spike-ESN) — full model.

Implements Algorithm 1: Spike Input Layer + Spike Reservoir + Ridge Regression.
"""

from __future__ import annotations
import torch
import numpy as np
from .spike_encoding import SpikeEncoder
from .reservoir import SpikeReservoir


class SpikeESN:
    """Brain-Inspired Spike Echo State Network for time series prediction.

    Parameters
    ----------
    N_res : int  — Number of reservoir neurons (default: 100).
    N_sam : int  — Spike sampling times (default: 100).
    rho   : float — Spectral radius (default: 0.9).
    eta   : float — Reservoir sparsity (default: 0.1).
    mu    : float — Ridge regularisation (default: 1e-8).
    psi   : float — Synaptic time constant (default: 5000).
    input_scaling : float — W_in scaling (default: 0.8).
    N_in  : int or None — Number of input-driven neurons.
    locality : float — Locality factor for topology (default: 0.0).
    seed  : int or None — Random seed.
    device : str or torch.device — Target device ('cpu', 'cuda', etc.).
    """

    def __init__(self, N_res=100, N_sam=100, rho=0.9, eta=0.1,
                 mu=1e-8, psi=5000.0, input_scaling=0.8, N_in=None, 
                 locality=0.0, seed=None, device="cpu"):
        self.N_res = N_res
        self.N_sam = N_sam
        self.rho = rho
        self.eta = eta
        self.mu = mu
        self.psi = psi
        self.input_scaling = input_scaling
        self.N_in = N_in
        self.locality = locality
        self.seed = seed
        self.device = device

        self.encoder = SpikeEncoder(N_sam=N_sam, device=device)
        self.reservoir = SpikeReservoir(
            N_res=N_res, N_sam=N_sam, rho=rho, eta=eta,
            psi=psi, input_scaling=input_scaling, N_in=N_in, 
            locality=locality, seed=seed, device=device
        )
        self.W_out: torch.Tensor | None = None
        self._train_states: torch.Tensor | None = None

    def fit(self, u, y, washout=200):
        """Train the Spike-ESN (Algorithm 1).

        Parameters
        ----------
        u : ndarray or torch.Tensor (T,) — Input time series.
        y : ndarray or torch.Tensor (T,) or (T-washout,) — Target output.
        washout : int — Initial steps to discard (default: 200).
        """
        T = len(u)
        if self.seed is not None:
            rng = torch.Generator(device=self.device).manual_seed(self.seed)
        else:
            rng = None

        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, device=self.device)
        else:
            u = u.to(self.device)

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=self.device)
        else:
            y = y.to(self.device)

        # Steps 1-7: Spike-encode entire input
        spike_matrix = self.encoder.encode_series(u, rng=rng)

        # Steps 10-13: Drive reservoir and collect states (Eq. 10)
        X, _ = self.reservoir.harvest_states(spike_matrix, washout=washout)
        self._train_states = X

        # Align target
        T_eff = X.shape[1]
        if len(y) == T:
            y_target = y[washout:washout + T_eff].reshape(1, -1)
        elif len(y) == T_eff:
            y_target = y.reshape(1, -1)
        else:
            raise ValueError(
                f"Target length {len(y)} doesn't match T={T} or T_eff={T_eff}")

        # Step 14: Ridge regression (Eq. 13)
        # W_out = y · X^T · (X·X^T + μ·I)^{-1}
        XXT = X @ X.T
        reg = self.mu * torch.eye(self.N_res, dtype=torch.float64, device=self.device)
        self.W_out = y_target.to(torch.float64) @ X.T @ torch.linalg.inv(XXT + reg)
        return self

    def predict(self, u, washout=0):
        """Predict using trained model.

        Parameters
        ----------
        u : ndarray or torch.Tensor (T,) — Input time series.
        washout : int — Steps to discard (default: 0).

        Returns
        -------
        y_hat : torch.Tensor (T-washout,) — Predicted output.
        """
        if self.W_out is None:
            raise RuntimeError("Call fit() first.")

        if self.seed is not None:
            rng = torch.Generator(device=self.device).manual_seed(self.seed)
        else:
            rng = None

        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, device=self.device)
        else:
            u = u.to(self.device)

        spike_matrix = self.encoder.encode_series(u, rng=rng)
        X, _ = self.reservoir.harvest_states(spike_matrix, washout=washout)
        return (self.W_out @ X).flatten()

    @staticmethod
    def rmse(y_true, y_pred):
        """Root Mean Square Error."""
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true)
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.tensor(y_pred)
        return float(torch.sqrt(torch.mean((y_true.to(y_pred.device) - y_pred) ** 2)))

    @staticmethod
    def mape(y_true, y_pred):
        """Mean Absolute Percentage Error."""
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true)
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.tensor(y_pred)
        y_true = y_true.to(y_pred.device)
        mask = y_true != 0
        if not torch.any(mask):
            return float("inf")
        return float(torch.mean(torch.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))

    def get_state_matrix(self):
        """Return state collection matrix X from last fit()."""
        return self._train_states

    def get_output_weights(self):
        """Return trained W_out."""
        return self.W_out

    def __repr__(self):
        return (f"SpikeESN(N_res={self.N_res}, N_sam={self.N_sam}, "
                f"rho={self.rho}, eta={self.eta}, mu={self.mu}, psi={self.psi}, device={self.device})")

