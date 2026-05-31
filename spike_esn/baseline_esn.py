"""
Baseline Echo State Network (ESN) — for comparison with Spike-ESN.

Standard ESN without spike encoding, sharing the same reservoir
parameters (N_res, rho, eta, mu, input_scaling) as the paper's setup.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


class ESN:
    """Standard Echo State Network (baseline).

    Parameters
    ----------
    N_res : int  — Number of reservoir neurons.
    rho   : float — Spectral radius.
    eta   : float — Reservoir sparsity.
    mu    : float — Ridge regularisation.
    input_scaling : float — Input weight scaling.
    seed  : int or None — Random seed.
    """

    def __init__(self, N_res=100, rho=0.9, eta=0.1,
                 mu=1e-8, input_scaling=0.8, seed=None):
        self.N_res = N_res
        self.rho = rho
        self.eta = eta
        self.mu = mu
        self.input_scaling = input_scaling
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.W_in = self.rng.uniform(-1, 1, size=(N_res, 1)) * input_scaling
        self.W_res = self._init_reservoir()
        self.W_out = None

    def _init_reservoir(self):
        N = self.N_res
        W = self.rng.uniform(-1, 1, size=(N, N))
        mask = self.rng.random(size=(N, N)) < self.eta
        W = W * mask
        eigs = np.linalg.eigvals(W)
        lmax = np.max(np.abs(eigs))
        if lmax == 0:
            return self._init_reservoir()
        return self.rho * (W / lmax)

    def fit(self, u, y, washout=200, initial_state=None):
        T = len(u)
        X = np.zeros((self.N_res, T))
        if initial_state is not None:
            x = initial_state.copy()
        else:
            x = np.zeros(self.N_res)
            
        for t in range(T):
            x = np.tanh(self.W_in.flatten() * u[t] + self.W_res @ x)
            X[:, t] = x
            
        X_eff = X[:, washout:]
        T_eff = X_eff.shape[1]
        
        if len(y) == T:
            y_target = y[washout:washout + T_eff].reshape(1, -1)
        else:
            y_target = y[:T_eff].reshape(1, -1)
            
        XXT = X_eff @ X_eff.T
        reg = self.mu * np.eye(self.N_res)
        self.W_out = y_target @ X_eff.T @ np.linalg.inv(XXT + reg)
        return self, x

    def predict(self, u, washout=0, initial_state=None):
        if self.W_out is None:
            raise RuntimeError("Call fit() first.")
        T = len(u)
        X = np.zeros((self.N_res, T))
        if initial_state is not None:
            x = initial_state.copy()
        else:
            x = np.zeros(self.N_res)
            
        for t in range(T):
            x = np.tanh(self.W_in.flatten() * u[t] + self.W_res @ x)
            X[:, t] = x
            
        X_eff = X[:, washout:]
        return (self.W_out @ X_eff).flatten(), x

    @staticmethod
    def rmse(y_true, y_pred):
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mape(y_true, y_pred):
        mask = y_true != 0
        if not np.any(mask):
            return float("inf")
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))
