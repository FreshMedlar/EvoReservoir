"""
Spike Reservoir — sparse recurrent network with spike-current input processing.

Implements Section III-B of the paper:
  1. Generate internal weight matrix W_res with sparsity η and spectral radius ρ  (Eq. 8)
  2. Generate input weight matrix W_in                                            (Alg. 1, step 8)
  3. Compute spike-based input current f_spike via exponential kernel              (Eq. 9)
  4. Update reservoir internal state x(t) with tanh activation                     (Eq. 9)
  5. Collect all states into the state collection matrix X                         (Eq. 10)
"""

from __future__ import annotations

import torch
import numpy as np


class SpikeReservoir:
    """Spike reservoir with exponential synaptic current model.

    Parameters
    ----------
    N_res : int
        Number of neurons in the reservoir.
    N_sam : int
        Length of the spike sequence (temporal dimension).
    rho : float
        Spectral radius — controls the echo state property. Must be < 1
        for the reservoir to have fading memory.
    eta : float
        Sparsity of the reservoir weight matrix (fraction of non-zero entries).
    psi : float
        Time constant of synaptic currents (ψ in the paper).
        Controls the magnitude and decay of the exponential current kernel.
    input_scaling : float
        Scaling factor applied to W_in (set to 0.8 in the paper).
    N_in : int or None
        Number of neurons that receive input. If None, all N_res neurons
        receive input. If provided, only the first N_in neurons are connected.
    locality : float
        Locality factor for topology. 0 = random (Erdos-Renyi). 
        > 0 = spatially local connections where the probability of a synapse 
        decays as 1/(dist+1)^locality. Creates a small-world network.
    seed : int or None
        Random seed for reproducibility.
    device : str or torch.device
        Target device for PyTorch computations ('cpu', 'cuda', etc.).
    """

    def __init__(
        self,
        N_res: int = 100,
        N_sam: int = 100,
        rho: float = 0.9,
        eta: float = 0.1,
        psi: float = 5000.0,
        input_scaling: float = 0.8,
        N_in: int | None = None,
        locality: float = 0.0,
        seed: int | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.N_res = N_res
        self.N_sam = N_sam
        self.rho = rho
        self.eta = eta
        self.psi = psi
        self.input_scaling = input_scaling
        self.N_in = N_in if N_in is not None else N_res
        self.locality = locality
        self.rng = np.random.default_rng(seed)
        self.device = device

        # Precompute the exponential kernel for fast spike current calculation
        t_seq = np.arange(1, self.N_sam + 1, dtype=np.float64)
        spike_kernel_np = np.exp(-(t_seq[:, None] - t_seq[None, :]) / self.psi).T
        self._spike_kernel = torch.tensor(spike_kernel_np, dtype=torch.float64, device=self.device)

        # Initialise weight matrices on host first, then push to device
        W_in_np = self._init_input_weights()
        W_res_np = self._init_reservoir_weights()
        self.W_in = torch.tensor(W_in_np, dtype=torch.float64, device=self.device)
        self.W_res = torch.tensor(W_res_np, dtype=torch.float64, device=self.device)

    # ------------------------------------------------------------------
    # Weight initialisation (done with numpy for exact matching)
    # ------------------------------------------------------------------
    def _init_input_weights(self) -> np.ndarray:
        """Generate W_in ∈ ℝ^{N_res × N_sam} from Uniform(−1, 1), scaled.
        If self.N_in < self.N_res, only the first N_in rows are non-zero.
        """
        # Start with a zero matrix
        W_in = np.zeros((self.N_res, self.N_sam), dtype=np.float64)

        # Only initialize the first N_in rows
        if self.N_in > 0:
            active_rows = self.rng.uniform(-1, 1, size=(self.N_in, self.N_sam))
            W_in[:self.N_in, :] = active_rows

        return W_in * self.input_scaling

    def _init_reservoir_weights(self) -> np.ndarray:
        """Generate W_res with sparsity η and spectral radius ρ  (Eq. 8).

        Steps:
          1. Sample W from Uniform(−1, 1).
          2. Apply sparsity mask (keep only η fraction of entries).
          3. Scale so that spectral radius equals ρ.
          """
        N = self.N_res

        # Random matrix in [-1, 1]
        W = self.rng.uniform(-1, 1, size=(N, N))

        # Apply sparsity mask
        if self.locality <= 0:
            # Standard uniform random sparsity
            mask = self.rng.random(size=(N, N)) < self.eta
        else:
            # Distance-dependent sparsity (Small World)
            # 1. Compute distances on a ring
            idx = np.arange(N)
            dist_mat = np.abs(idx[:, None] - idx[None, :])
            # Account for ring wrap-around
            dist_mat = np.minimum(dist_mat, N - dist_mat)

            # 2. Compute probabilities: P(d) = 1 / (d + 1)^locality
            # We then scale these probabilities so the mean matches self.eta
            p_mat = 1.0 / (dist_mat + 1.0)**self.locality
            
            # Zero out self-connections
            np.fill_diagonal(p_mat, 0)
            
            # Normalize so the overall sparsity matches eta
            current_mean = np.mean(p_mat)
            if current_mean > 0:
                p_mat = p_mat * (self.eta / current_mean)
            p_mat = np.clip(p_mat, 0, 1)

            mask = self.rng.random(size=(N, N)) < p_mat

        W = W * mask

        # Compute maximum eigenvalue
        eigenvalues = np.linalg.eigvals(W)
        lambda_max = np.max(np.abs(eigenvalues))

        if lambda_max == 0:
            # Degenerate case — re-initialise with slightly denser matrix
            return self._init_reservoir_weights()

        # Eq. 8 — scale to desired spectral radius
        W_res = self.rho * (W / lambda_max)
        return W_res

    # ------------------------------------------------------------------
    # Spike current computation  (Eq. 9 — f_spike)
    # ------------------------------------------------------------------
    def compute_spike_current(
        self, spike_seq: torch.Tensor
    ) -> torch.Tensor:
        """Compute the spike-based input current vector f_spike.

        Processes the input in blocks of self.N_sam to support multi-channel
        (one-hot) encoding correctly. Each channel gets its own local
        timeline from 1 to N_sam.
        """
        n_channels = len(spike_seq) // self.N_sam
        spikes_2d = spike_seq.reshape(n_channels, self.N_sam).to(self.device, dtype=torch.float64)
        
        # Fast matrix multiplication instead of looping over channels
        f_spike_2d = spikes_2d @ self._spike_kernel
        
        return f_spike_2d.flatten()

    # ------------------------------------------------------------------
    # Reservoir state update  (Eq. 9, first line)
    # ------------------------------------------------------------------
    def update_state(
        self,
        f_spike: torch.Tensor,
        x_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Compute x(t) = tanh(W_in · f_spike + W_res · x(t−1)).

        Parameters
        ----------
        f_spike : torch.Tensor of shape (N_sam,)
            Spike-based input current vector.
        x_prev : torch.Tensor of shape (N_res,)
            Previous reservoir state.

        Returns
        -------
        x_new : torch.Tensor of shape (N_res,)
            Updated reservoir state.
        """
        return torch.tanh(self.W_in @ f_spike + self.W_res @ x_prev)

    # ------------------------------------------------------------------
    # Harvest states from a full spike-encoded series  (Eq. 10)
    # ------------------------------------------------------------------
    def harvest_states(
        self,
        spike_matrix: torch.Tensor,
        washout: int = 0,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Drive the reservoir with a spike-encoded time series and collect states.

        Parameters
        ----------
        spike_matrix : torch.Tensor of shape (T, N_sam)
            Each row is the spike sequence for one time step.
        washout : int
            Number of initial time steps to discard (reservoir warm-up).
        initial_state : torch.Tensor of shape (N_res,) or None
            The starting state of the reservoir. Defaults to zeros.

        Returns
        -------
        X : torch.Tensor of shape (N_res, T − washout)
            State collection matrix (Eq. 10), each column is x(t).
        final_state : torch.Tensor of shape (N_res,)
            The state of the reservoir after the last time step.
        """
        T = spike_matrix.shape[0]
        n_channels = spike_matrix.shape[1] // self.N_sam
        
        # 1. Fast vectorised computation of f_spike for ALL time steps at once
        # Reshape to (T, C, N_sam) and multiply by kernel (N_sam, N_sam)
        spikes_3d = spike_matrix.reshape(T, n_channels, self.N_sam).to(self.device, dtype=torch.float64)
        f_spike_3d = spikes_3d @ self._spike_kernel
        f_spike_all = f_spike_3d.reshape(T, n_channels * self.N_sam)
        
        # 2. Precompute the W_in projection for all time steps
        W_in_f_spike = self.W_in @ f_spike_all.T
        
        X_all = torch.zeros((self.N_res, T), dtype=torch.float64, device=self.device)
        if initial_state is not None:
            x = initial_state.clone().to(self.device, dtype=torch.float64)
        else:
            x = torch.zeros(self.N_res, dtype=torch.float64, device=self.device)  # x(0) = 0

        for t in range(T):
            # 3. Only the recurrent state update remains in the sequential loop!
            x = torch.tanh(W_in_f_spike[:, t] + self.W_res @ x)
            X_all[:, t] = x

        # Discard washout period
        return X_all[:, washout:], x

