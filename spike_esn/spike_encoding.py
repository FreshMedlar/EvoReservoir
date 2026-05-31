"""
Spike Input Layer — Poisson-distribution-based spike encoding.

Implements Section III-A of the paper:
  1. Compute the average spike interval h_κ(t) from normalized input  (Eq. 2)
  2. Sample spike intervals κ_l(t) from a Poisson distribution        (Eq. 3-5)
  3. Generate binary spike sequences s_i(t)                            (Eq. 6-7)

The spike input layer converts a scalar input u(t) into a binary vector
of length N_sam, where 1s indicate spike activations.
"""

from __future__ import annotations

import torch
import numpy as np


class SpikeEncoder:
    """Poisson-distribution spike encoder for time series data.

    Parameters
    ----------
    N_sam : int
        Spike sampling times — length of the output spike sequence.
        Higher values project inputs into a higher temporal dimension,
        improving feature extraction at the cost of computation.
    N_int : int or None
        Number of spike intervals. If None, it is adaptively determined
        from the Poisson sampling process.
    device : str or torch.device
        Target device for PyTorch computations ('cpu', 'cuda', etc.).
    """

    def __init__(
        self,
        N_sam: int = 100,
        N_int: int | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.N_sam = N_sam
        self.N_int = N_int
        self.device = device

    # ------------------------------------------------------------------
    # Eq. 2 — average spike interval
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_avg_interval(
        u: float | torch.Tensor,
        U_max: float,
        U_min: float,
        N_sam: int,
    ) -> float | torch.Tensor:
        """Compute h_κ(t) = N_sam × (U_max − u(t)) / (U_max − U_min).

        When u(t) is large (close to U_max), h_κ is small → high spike
        frequency.  When u(t) is small, h_κ is large → low spike frequency.
        """
        denom = U_max - U_min
        if denom == 0:
            # Constant signal: return a neutral interval
            return N_sam / 2.0
        return N_sam * (U_max - u) / denom

    # ------------------------------------------------------------------
    # Eq. 3-6 — generate a single spike sequence from one scalar input
    # ------------------------------------------------------------------
    def encode_scalar(
        self,
        u: float | torch.Tensor,
        U_max: float,
        U_min: float,
        rng: torch.Generator | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Encode a single scalar value into a spike sequence of length N_sam.

        Parameters
        ----------
        u : float or torch.Tensor
            The (normalised) input value at time *t*.
        U_max, U_min : float
            Global max / min of the input time series (for normalisation).
        rng : torch.Generator, optional
            Random number generator for reproducibility.
        deterministic : bool
            If True, use a seed derived from 'u' to make the encoding 
            consistent for the same input value.
        """
        val_float = float(u) if isinstance(u, torch.Tensor) else float(u)

        if deterministic:
            # Create a stable seed from the value of u
            # Using a fixed precision to avoid float noise
            seed = int(abs(hash(round(val_float, 8))) % (2**32))
            local_rng = torch.Generator(device=self.device)
            local_rng.manual_seed(seed)
        elif rng is None:
            local_rng = None
        else:
            local_rng = rng

        N_sam = self.N_sam
        # Eq. 2 — average interval
        h_kappa = self._compute_avg_interval(val_float, U_max, U_min, N_sam)
        h_kappa = max(h_kappa, 1.0)

        intervals: list[int] = []
        cumsum = 0
        
        # We perform Poisson sampling. Since this is scalar and loop-based, 
        # we can sample using torch.poisson on device.
        h_kappa_tensor = torch.tensor(h_kappa, device=self.device, dtype=torch.float32)
        while cumsum < N_sam:
            if local_rng is not None:
                kappa = int(torch.poisson(h_kappa_tensor, generator=local_rng).item())
            else:
                kappa = int(torch.poisson(h_kappa_tensor).item())
            kappa = max(kappa, 1)
            if cumsum + kappa > N_sam:
                break
            intervals.append(kappa)
            cumsum += kappa

        # If a fixed N_int was requested, truncate or extend
        if self.N_int is not None and len(intervals) > self.N_int:
            intervals = intervals[: self.N_int]

        # Eq. 6 — build spike sequence from intervals
        spike_seq = torch.zeros(N_sam, dtype=torch.int8, device=self.device)
        pos = 0
        for kappa in intervals:
            pos += kappa
            if pos <= N_sam:
                spike_seq[pos - 1] = 1  # 0-indexed → pos-1

        return spike_seq

    # ------------------------------------------------------------------
    # Batch helper — encode an entire time series
    # ------------------------------------------------------------------
    def encode_series(
        self,
        u_series: torch.Tensor | np.ndarray,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Encode a 1-D time series into a matrix of spike sequences.

        Parameters
        ----------
        u_series : torch.Tensor or ndarray of shape (T,)
            Input time series.
        rng : torch.Generator, optional
            PyTorch random number generator.

        Returns
        -------
        spikes : torch.Tensor of shape (T, N_sam) on self.device
            Each row is the spike sequence for the corresponding time step.
        """
        if not isinstance(u_series, torch.Tensor):
            u_series = torch.tensor(u_series, dtype=torch.float64, device=self.device)
        else:
            u_series = u_series.to(device=self.device, dtype=torch.float64)

        T = len(u_series)
        U_max = float(torch.max(u_series))
        U_min = float(torch.min(u_series))
        denom = U_max - U_min

        if denom == 0:
            h_kappas = torch.full((T,), self.N_sam / 2.0, dtype=torch.float64, device=self.device)
        else:
            h_kappas = self.N_sam * (U_max - u_series) / denom

        h_kappas = torch.clamp(h_kappas, min=1.0)

        # Fast vectorised Poisson sampling using torch.poisson
        # Reshape h_kappas to (T, 1) and expand to (T, N_sam)
        lam_grid = h_kappas.unsqueeze(1).expand(T, self.N_sam).to(torch.float32)
        if rng is not None:
            kappas = torch.poisson(lam_grid, generator=rng)
        else:
            kappas = torch.poisson(lam_grid)

        kappas = torch.clamp(kappas, min=1).to(torch.int64)
        cumsums = torch.cumsum(kappas, dim=1)

        valid = cumsums <= self.N_sam
        b_idx, i_idx = torch.nonzero(valid, as_tuple=True)
        spike_pos = cumsums[b_idx, i_idx] - 1

        spikes = torch.zeros((T, self.N_sam), dtype=torch.int8, device=self.device)
        spikes[b_idx, spike_pos] = 1

        return spikes

