"""
Echo State Network (ESN) edge-of-chaos exploration summary

- Create a sparse random reservoir matrix (W) with adjustable spectral radius.
- Simulate autonomous dynamics (no input) and detect stability/explosion.
- Estimate largest Lyapunov exponent using two nearby trajectories.
- Simulate reservoir response to external input (e.g., sine wave).
- Train a linear readout (Ridge regression) to predict future input from reservoir states.

Main functions:
- make_sparse_reservoir(N, density, scale, random_state): returns sparse matrix W
- spectral_radius_numpy(W): returns spectral radius of W
- rescale_to_spectral_radius(W, target_rho): rescales W
- run_autonomous(W, x0, steps, explosion_threshold): simulates x[t+1] = tanh(W x[t])
- estimate_lyapunov_two_trajectories(W, x0, perturb, steps): estimates Lyapunov exponent
- generate_sine_input(T, freq, amplitude): creates input signal
- make_input_weights(N, input_dim, scale, random_state): returns input weights
- run_with_input(W, W_in, x0, inputs, steps, explosion_threshold): simulates reservoir with input
- train_and_test_readout(states, inputs, target_steps_ahead, train_fraction, alpha): trains linear readout

Usage:
1. Build a random reservoir and rescale to desired spectral radius.
2. Simulate autonomous and input-driven dynamics.
3. Estimate Lyapunov exponent for chaos detection.
4. Train and evaluate a linear readout for time series prediction.

This script summarizes the notebook logic for LLM input. Full code and plots are in chaos.ipynb.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# ----------------------
# Utilities
# ----------------------
def make_sparse_reservoir(N, density=0.02, scale=1.0, random_state=None):
    rng = np.random.RandomState(random_state)
    mask = rng.rand(N, N) < density
    W = rng.randn(N, N) * mask
    row_sums = np.abs(W).sum(axis=1)
    zero_rows = np.where(row_sums == 0)[0]
    for r in zero_rows:
        j = rng.randint(0, N)
        W[r, j] = rng.randn()
    W *= scale
    return W

def spectral_radius_numpy(W):
    vals = np.linalg.eigvals(W)
    return np.max(np.abs(vals))

def rescale_to_spectral_radius(W, target_rho):
    sr = spectral_radius_numpy(W)
    if sr == 0:
        return W
    return W * (target_rho / sr)

# ----------------------
# Dynamics, explosion check, Lyapunov estimate
# ----------------------
def run_autonomous(W, x0, steps=1000, explosion_threshold=1e6):
    N = W.shape[0]
    xs = np.zeros((steps, N))
    x = x0.copy()
    for t in range(steps):
        x = np.tanh(W @ x)
        xs[t] = x
        norm = np.linalg.norm(x)
        if np.isnan(norm) or np.isinf(norm) or norm > explosion_threshold:
            return xs[:t+1], True
    return xs, False

def estimate_lyapunov_two_trajectories(W, x0, perturb=1e-8, steps=500, renormalize_every=1):
    N = W.shape[0]
    rng = np.random.RandomState(0)
    x = x0.copy()
    y = x0.copy() + rng.randn(N) * perturb
    d = np.linalg.norm(y - x)
    if d == 0:
        y += perturb * np.ones_like(x)
        d = np.linalg.norm(y - x)
    logs = []
    for t in range(steps):
        x = np.tanh(W @ x)
        y = np.tanh(W @ y)
        d_new = np.linalg.norm(y - x)
        if np.isnan(d_new) or np.isinf(d_new) or d_new > 1e12:
            return float('nan'), True
        logs.append(np.log((d_new + 1e-16) / (d + 1e-16)))
        if (t + 1) % renormalize_every == 0 and d_new != 0:
            y = x + (y - x) * (perturb / d_new)
            d = np.linalg.norm(y - x)
        else:
            d = d_new
    lyap = np.mean(logs)
    return lyap, False

# ----------------------
# Input generation and reservoir with input
# ----------------------
def generate_sine_input(T, freq=0.05, amplitude=1.0):
    return amplitude * np.sin(2 * np.pi * freq * np.arange(T))

def make_input_weights(N, input_dim, scale=1.0, random_state=None):
    rng = np.random.RandomState(random_state)
    return rng.randn(N, input_dim) * scale

def run_with_input(W, W_in, x0, inputs, steps=None, explosion_threshold=1e6):
    N = W.shape[0]
    if steps is None:
        steps = inputs.shape[0]
    T = steps
    xs = np.zeros((T, N))
    x = x0.copy()
    for t in range(T):
        u = inputs[t] if t < inputs.shape[0] else np.zeros(W_in.shape[1])
        x = np.tanh(W @ x + W_in @ u)
        xs[t] = x
        norm = np.linalg.norm(x)
        if np.isnan(norm) or np.isinf(norm) or norm > explosion_threshold:
            return xs[:t+1], True
    return xs, False

# ----------------------
# Readout training
# ----------------------
def train_and_test_readout(states, inputs, target_steps_ahead=1, train_fraction=0.8, alpha=1e-6, plot_results=True):
    T = states.shape[0]
    targets = inputs[target_steps_ahead:T]
    states_aligned = states[:T - target_steps_ahead]
    split = int(train_fraction * states_aligned.shape[0])
    X_train, X_test = states_aligned[:split], states_aligned[split:]
    y_train, y_test = targets[:split], targets[split:]
    readout = Ridge(alpha=alpha)
    readout.fit(X_train, y_train)
    train_pred = readout.predict(X_train)
    test_pred = readout.predict(X_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    if plot_results:
        plt.figure(figsize=(8,4))
        plt.plot(y_test, label="Target (test)")
        plt.plot(test_pred, label="Prediction", linestyle="--")
        plt.title(f"{target_steps_ahead}-step prediction (Test RMSE: {test_rmse:.4f})")
        plt.legend()
        plt.show()
    return readout, train_rmse, test_rmse

# ----------------------
# Main experiment
# ----------------------
if __name__ == "__main__":
    N = 150
    input_dim = 1
    density = 0.02
    spectral_radii = np.array([0.8, 0.9, 0.95, 0.99, 1.0, 1.01, 1.05, 1.1])
    sim_steps = 600
    lyap_steps = 300
    explosion_threshold = 1e6
    random_state = 12345

    # Build base reservoir
    base_W = make_sparse_reservoir(N, density=density, random_state=random_state)
    base_sr = spectral_radius_numpy(base_W)
    print(f"base spectral radius (before scaling) = {base_sr:.6f}")

    results = []
    for rho in spectral_radii:
        W = base_W * (rho / base_sr)
        x0 = np.random.RandomState(random_state + 1).randn(N) * 1e-4
        xs, exploded = run_autonomous(W, x0, steps=sim_steps, explosion_threshold=explosion_threshold)
        lyap, lyap_exploded = estimate_lyapunov_two_trajectories(W, x0, perturb=1e-8, steps=lyap_steps)
        final_norm = np.linalg.norm(xs[-1])
        results.append({
            "rho": rho,
            "exploded_run": exploded,
            "lyapunov": lyap,
            "lyap_exploded": lyap_exploded,
            "final_norm": final_norm,
            "states": xs
        })
        print(f"rho={rho:.3f} | exploded_run={exploded} | lyapunov={lyap:.6e} | final_norm={final_norm:.3e}")

    # Plot Lyapunov vs rho
    rhos = np.array([r["rho"] for r in results])
    lyaps = np.array([r["lyapunov"] for r in results])
    plt.figure(figsize=(6,4))
    plt.plot(rhos, lyaps, marker='o')
    plt.axhline(0, linestyle='--')
    plt.title("Estimated Lyapunov exponent vs spectral radius (discrete-time, per-step)")
    plt.xlabel("Target spectral radius")
    plt.ylabel("Estimated Lyapunov (mean log growth per step)")
    plt.tight_layout()
    plt.show()

    # pick the rho closest to 1.0 for time-series plots
    idx_closest = np.argmin(np.abs(rhos - 1.0))
    chosen = results[idx_closest]
    xs_chosen = chosen["states"]
    T = xs_chosen.shape[0]
    times = np.arange(T)

    # Plot reservoir norm over time
    norms = np.linalg.norm(xs_chosen, axis=1)
    plt.figure(figsize=(8,4))
    plt.plot(times, norms)
    plt.title(f"Reservoir state norm over time (rho={chosen['rho']:.3f})")
    plt.xlabel("Time step")
    plt.ylabel("||x||_2")
    plt.tight_layout()
    plt.show()

    # Plot first few neuron activations
    num_neurons_plot = min(6, xs_chosen.shape[1])
    plt.figure(figsize=(8,4))
    for i in range(num_neurons_plot):
        plt.plot(times, xs_chosen[:, i], label=f"n{i}")
    plt.title(f"Example neuron activations (first {num_neurons_plot}) rho={chosen['rho']:.3f}")
    plt.xlabel("Time step")
    plt.ylabel("activation (tanh)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Now with input and readout
    W_in = make_input_weights(N, input_dim, scale=0.5, random_state=random_state)
    inputs = generate_sine_input(sim_steps).reshape(-1, input_dim)
    xs_input, exploded_input = run_with_input(W, W_in, x0, inputs, steps=sim_steps, explosion_threshold=explosion_threshold)
    # Use the first neuron for readout (or flatten if input_dim==1)
    readout, train_rmse, test_rmse = train_and_test_readout(
        states=xs_input,
        inputs=inputs.flatten(),
        target_steps_ahead=1,
        alpha=1e-6
    )
    print(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")