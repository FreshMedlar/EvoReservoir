import numpy as np
import torch
import torch.nn as nn


class HybridReservoir(nn.Module):
    """Hybrid Reservoir combining E-I constraints, leaky integration,
    synaptic filtering, structural plasticity, augmented readout,
    heterogeneous per-neuron time constants, and learned state feedback.
    """
    def __init__(
        self,
        res_size=100,
        output_dim=1,
        input_scaling=0.8,
        rho=0.95,
        density=0.1,
        e_ratio=0.8,
        leaking_rate_range=(0.05, 1.0),
        synaptic_decay=1.0,
        spontaneous_rate=0.0,
        mu=1e-8,
        feedback_steps=0,
        feedback_lr=25.0,
        feedback_eps_a=1e-5,
        augmented_readout=True,
        device="cpu",
    ):
        super().__init__()
        self.res_size = res_size
        self.output_dim = output_dim
        self.input_scaling = input_scaling
        self.rho = rho
        self.density = density
        self.e_ratio = e_ratio
        self.e_size = int(res_size * e_ratio)
        self.synaptic_decay = synaptic_decay
        self.spontaneous_rate = spontaneous_rate
        self.mu = mu
        self.feedback_steps = feedback_steps
        self.feedback_lr = feedback_lr
        self.feedback_eps_a = feedback_eps_a
        self.augmented_readout = augmented_readout
        self.device = device

        # ── Per-neuron heterogeneous leaking rates ──────────────────────
        # Log-uniform distribution across the range so the reservoir
        # captures both fast transients and slow dynamics simultaneously.
        lo, hi = leaking_rate_range
        self.leaking_rates = torch.exp(
            torch.linspace(np.log(lo), np.log(hi), res_size, device=device)
        )

        # ── Recurrent weight matrix W (with E-I constraints) ───────────
        W = torch.randn(res_size, res_size, device=device)
        mask = torch.rand(res_size, res_size, device=device) < density
        W = W * mask

        # Excitatory columns positive, inhibitory columns negative
        W[:, :self.e_size] = torch.abs(W[:, :self.e_size])
        W[:, self.e_size:] = -torch.abs(W[:, self.e_size:])

        # Spectral radius scaling
        radius = torch.max(torch.abs(torch.linalg.eigvals(W)))
        if radius > 0:
            W = W * (rho / radius)
        self.register_buffer('W', W)

        # ── Input weight matrix ────────────────────────────────────────
        W_in = torch.randn(res_size, output_dim, device=device) * input_scaling
        self.register_buffer('W_in', W_in)

        # ── State feedback matrix V (learned) ──────────────────────────
        # Following Ehlers et al.: A_bar = W + W_in @ V.T
        # V is (res_size, output_dim), initialised to 0.
        self.V = nn.Parameter(
            torch.zeros(res_size, output_dim, device=device)
        )

        # ── State variables ────────────────────────────────────────────
        self.state = torch.zeros(res_size, device=device)
        self.f_in = torch.zeros(output_dim, device=device)
        self.scaling = torch.ones(res_size, device=device)

        # Homeostasis trailing activations
        self.fast_trail = torch.ones(res_size, device=device) * 0.5
        self.slow_trail = torch.ones(res_size, device=device) * 0.5

        # ── Readout weights ────────────────────────────────────────────
        # With augmented readout the feature dim is:
        #   1 (bias) + output_dim (input) + res_size (state) + res_size (state²)
        self._readout_dim = (
            (1 + output_dim + res_size + res_size) if augmented_readout
            else res_size
        )
        self.W_out = None  # shape: (output_dim, readout_dim)
        self.C_out = None  # bias from centred regression

        self.evolution_enabled = True

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────
    def _build_features(self, state, u):
        """Build the augmented feature vector [1, u, x, x²]."""
        if self.augmented_readout:
            return torch.cat([
                torch.ones(1, device=self.device),
                u,
                state,
                state ** 2,
            ])
        return state

    def _effective_W(self):
        """Compute A_bar = W + W_in @ V.T (state-feedback modified recurrent matrix)."""
        return self.W + self.W_in @ self.V.t()

    def reset_state(self):
        self.state = torch.zeros(self.res_size, device=self.device)
        self.f_in = torch.zeros(self.output_dim, device=self.device)
        self.scaling = torch.ones(self.res_size, device=self.device)
        self.fast_trail = torch.ones(self.res_size, device=self.device) * 0.5
        self.slow_trail = torch.ones(self.res_size, device=self.device) * 0.5

    # ──────────────────────────────────────────────────────────────────
    # Forward step
    # ──────────────────────────────────────────────────────────────────
    def step(self, u, W_eff=None):
        """Advance the reservoir by one time step.

        Parameters
        ----------
        u : torch.Tensor of shape (output_dim,)
            Current input vector.
        W_eff : torch.Tensor or None
            Pre-computed effective recurrent matrix (A_bar). If None,
            self.W is used directly (without feedback).
        """
        # Input synaptic filter
        self.f_in = (1.0 - self.synaptic_decay) * self.f_in + self.synaptic_decay * u

        # Input and recurrent projections
        x_in = torch.matmul(self.W_in, self.f_in)
        if W_eff is not None:
            x_rec = torch.matmul(W_eff, self.state)
        else:
            x_rec = torch.matmul(self.W, self.state)

        # Leaky integrator with per-neuron time constants
        state_tilde = torch.tanh(self.scaling * (x_rec + x_in))
        self.state = (1.0 - self.leaking_rates) * self.state + self.leaking_rates * state_tilde

        # Trailing activation averages
        abs_state = torch.abs(self.state)
        self.slow_trail = 0.99 * self.slow_trail + 0.01 * abs_state
        self.fast_trail = 0.8 * self.fast_trail + 0.2 * abs_state

        # Plasticity (only during evolution phase)
        if self.evolution_enabled:
            self._apply_plasticity()

        return self.state

    def _apply_plasticity(self):
        """Homeostatic gain control and structural genesis/pruning."""
        abs_slow = self.slow_trail
        low_mask = abs_slow < 0.15
        high_mask = abs_slow > 0.85
        normal_mask = ~(low_mask | high_mask)

        self.scaling[low_mask] += 0.01
        self.scaling[high_mask] -= 0.01
        self.scaling[normal_mask] = 1.0
        self.scaling = torch.clamp(self.scaling, min=0.1, max=5.0)

        # Genesis / Pruning of recurrent weights
        weak_mask = abs_slow < 0.10
        strong_mask = abs_slow > 0.90

        if self.spontaneous_rate > 0:
            exploratory_mask = torch.rand(self.res_size, device=self.device) < self.spontaneous_rate
            weak_mask = weak_mask | exploratory_mask

        weak_indices = torch.where(weak_mask)[0]
        strong_indices = torch.where(strong_mask)[0]

        if len(weak_indices) > 0:
            src = torch.randint(0, self.res_size, (len(weak_indices),), device=self.device)
            delta = torch.where(src < self.e_size, 0.1, -0.1)
            with torch.no_grad():
                self.W[weak_indices, src] += delta

        if len(strong_indices) > 0:
            src = torch.randint(0, self.res_size, (len(strong_indices),), device=self.device)
            delta = torch.where(src < self.e_size, -0.1, 0.1)
            with torch.no_grad():
                self.W[strong_indices, src] += delta
                self.W[:, :self.e_size] = torch.clamp(self.W[:, :self.e_size], min=0.0)
                self.W[:, self.e_size:] = torch.clamp(self.W[:, self.e_size:], max=0.0)

    # ──────────────────────────────────────────────────────────────────
    # State harvesting
    # ──────────────────────────────────────────────────────────────────
    def harvest_states(self, u_series, washout=200, W_eff=None):
        """Drive the reservoir and collect (augmented) feature vectors."""
        self.reset_state()
        features = []
        for t in range(len(u_series)):
            u_t = u_series[t]
            if not isinstance(u_t, torch.Tensor):
                u_t = torch.tensor(u_t, dtype=torch.float32, device=self.device)
            if u_t.ndim == 0:
                u_t = u_t.unsqueeze(0)

            self.step(u_t, W_eff=W_eff)

            if t >= washout:
                features.append(self._build_features(self.state, u_t).clone())

        return torch.stack(features)  # (T_eff, readout_dim)

    # ──────────────────────────────────────────────────────────────────
    # Readout fitting (ridge regression)
    # ──────────────────────────────────────────────────────────────────
    def _fit_readout(self, X, y_target):
        """Centred ridge regression → sets self.W_out, self.C_out."""
        N = X.shape[0]

        X_mean = X.mean(dim=0, keepdim=True)
        Y_mean = y_target.mean(dim=0, keepdim=True)

        Xc = X - X_mean
        Yc = y_target - Y_mean

        K_xx = (Xc.T @ Xc) / N
        K_xy = (Xc.T @ Yc) / N

        I = torch.eye(self._readout_dim, device=self.device)
        W = torch.linalg.solve(K_xx + self.mu * I, K_xy)  # (readout_dim, output_dim)
        C = Y_mean - X_mean @ W                            # (1, output_dim)

        self.W_out = W.T.detach()   # (output_dim, readout_dim)
        self.C_out = C.detach()     # (1, output_dim)
        return W, C

    # ──────────────────────────────────────────────────────────────────
    # Feedback training (Ehlers et al. projected gradient descent)
    # ──────────────────────────────────────────────────────────────────
    def _train_feedback(self, u_series, y_target, washout):
        """Train V via gradient descent with spectral-norm stability constraint.

        This follows the approach of Ehlers, Nurdin & Soh:
          A_bar = W + W_in @ V.T
        V is optimised to minimise readout MSE while keeping
        ||A_bar||_2 < a  (where a = rho, the spectral radius bound).
        """
        a = float(self.rho)

        with torch.no_grad():
            self.V.zero_()

        loss_history = []
        for step_i in range(self.feedback_steps):
            self.V.grad = None

            # Forward pass with current V (graph needed for V grad)
            W_eff = self._effective_W()
            self.evolution_enabled = False  # no plasticity during feedback opt
            X = self.harvest_states(u_series, washout=washout, W_eff=W_eff)

            # Fit readout for current states (no grad through solve)
            with torch.no_grad():
                W_opt, C_opt = self._fit_readout(X, y_target)

            # Loss
            preds = X @ W_opt + C_opt
            loss = 0.5 * torch.mean((preds - y_target) ** 2)
            loss_history.append(loss.item())
            loss.backward()

            # Gradient step on V with stability projection
            grad_V = self.V.grad.detach()
            delta_V = -self.feedback_lr * grad_V
            delta_V = self._project_delta_V(self.V.detach(), delta_V, a)

            with torch.no_grad():
                self.V.copy_(self.V + delta_V)

        # Final readout fit with learned V
        with torch.no_grad():
            W_eff = self._effective_W()
            X = self.harvest_states(u_series, washout=washout, W_eff=W_eff)
            self._fit_readout(X, y_target)

        return loss_history

    def _project_delta_V(self, V, delta_V, a):
        """Project delta_V so that ||A_bar_next||_2 < a."""
        A_bar_curr = self.W + self.W_in @ V.t()
        _, s_curr, _ = torch.linalg.svd(A_bar_curr)

        V_next = V + delta_V
        A_bar_next = self.W + self.W_in @ V_next.t()
        s_next = torch.linalg.svdvals(A_bar_next)

        if s_next[0] < a:
            return delta_V

        # Right singular vector for largest singular value
        _, _, Vt_curr = torch.linalg.svd(A_bar_curr)
        umax = Vt_curr[0]

        w2 = self.W_in.t() @ A_bar_curr @ umax
        w1 = delta_V.t() @ umax

        delta_lambda2 = 2.0 * torch.dot(w1, w2)
        Delta = s_curr[0] ** 2 + delta_lambda2 - a ** 2 + self.feedback_eps_a

        if Delta > 0:
            w2_norm_sq = torch.sum(w2 ** 2)
            if w2_norm_sq > 1e-12:
                correction = (Delta / (2.0 * w2_norm_sq)) * torch.outer(umax, w2)
                delta_V = delta_V - correction

        # Backtracking line search
        alpha = 1.0
        for _ in range(20):
            V_test = V + alpha * delta_V
            A_bar_test = self.W + self.W_in @ V_test.t()
            s_test = torch.linalg.svdvals(A_bar_test)
            if s_test[0] < a:
                break
            alpha *= 0.5

        return alpha * delta_V

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────
    def fit(self, u_series, y_series, washout=200):
        """Train the reservoir: plasticity → feedback → readout.

        1. Harvest states with evolution/plasticity active (self-organisation).
        2. Freeze plasticity, then optionally train feedback matrix V.
        3. Fit augmented readout via centred ridge regression.
        """
        # Prepare targets
        y_target = y_series[washout:]
        if not isinstance(y_target, torch.Tensor):
            y_target = torch.tensor(y_target, dtype=torch.float32, device=self.device)
        if y_target.ndim == 1:
            y_target = y_target.unsqueeze(1)

        # Phase 1: plasticity run (self-organise topology)
        self.evolution_enabled = True
        self.harvest_states(u_series, washout=washout)
        self.evolution_enabled = False

        # Phase 2: feedback optimisation (if enabled)
        if self.feedback_steps > 0:
            self._train_feedback(u_series, y_target, washout)
        else:
            # No feedback: just harvest + fit readout
            W_eff = self._effective_W()
            X = self.harvest_states(u_series, washout=washout, W_eff=W_eff)
            self._fit_readout(X, y_target)

    def predict(self, u_series, washout=0):
        """Predict using the trained model (evolution frozen)."""
        self.evolution_enabled = False
        W_eff = self._effective_W()

        y_pred = []
        for t in range(len(u_series)):
            u_t = u_series[t]
            if not isinstance(u_t, torch.Tensor):
                u_t = torch.tensor(u_t, dtype=torch.float32, device=self.device)
            if u_t.ndim == 0:
                u_t = u_t.unsqueeze(0)

            self.step(u_t, W_eff=W_eff)

            if t >= washout:
                feat = self._build_features(self.state, u_t)
                pred = (self.W_out @ feat) + self.C_out.squeeze(0)
                y_pred.append(pred.clone())

        return torch.stack(y_pred)
