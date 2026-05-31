#!/usr/bin/env python3
import os
import sys
import time
import torch
import numpy as np

# Ensure we can import from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spike_esn.model import SpikeESN
from tesi.model.reservoir import Reservoir

def generate_mackey_glass(n_steps=3000, tau_mg=17, delta_t=0.1, seed=42):
    rng = np.random.default_rng(seed)
    history_len = max(tau_mg * 10, 200)
    total = n_steps + history_len
    x = np.zeros(total)
    x[:history_len] = 0.9 + 0.05 * rng.random(history_len)

    for t in range(history_len, total):
        x_tau = x[t - tau_mg]
        x[t] = x[t - 1] + delta_t * (
            0.2 * x_tau / (1.0 + x_tau ** 10) - 0.1 * x[t - 1]
        )

    return x[history_len:]

def normalise(data):
    dmin, dmax = data.min(), data.max()
    if dmax == dmin:
        return np.zeros_like(data)
    return (data - dmin) / (dmax - dmin)

def run_tesi_reservoir(u_train, y_train, u_test, y_test, res_size=100, density=0.1, mu=1e-8, input_scaling=0.8, washout=200, use_evolution=True):
    # Initialize reservoir
    reservoir = Reservoir(res_size=res_size, output_dim=1, input_scaling=input_scaling, density=density)
    
    # 1. Training Phase: harvest states
    reservoir.reset_state()
    X_train = []
    
    # Convert inputs to torch tensors
    u_train_t = torch.tensor(u_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    u_test_t = torch.tensor(u_test, dtype=torch.float32)
    
    for t in range(len(u_train)):
        x_in = torch.tensor([u_train_t[t]], dtype=torch.float32)
        with torch.no_grad():
            if use_evolution:
                reservoir.step(x_in)
            else:
                reservoir.step_no_evolution(x_in)
        if t >= washout:
            X_train.append(reservoir.state.clone())
            
    X_train = torch.stack(X_train)  # (train_len - washout, res_size)
    y_train_target = y_train_t[washout:].unsqueeze(1)  # (train_len - washout, 1)
    
    # Ridge regression
    A = torch.matmul(X_train.T, X_train) + mu * torch.eye(res_size)
    B = torch.matmul(X_train.T, y_train_target)
    W_out_T = torch.linalg.solve(A, B)
    W_out = W_out_T.T
    
    # 2. Testing Phase: predict
    # We do NOT reset state, we continue from the training state
    y_pred = []
    for t in range(len(u_test)):
        x_in = torch.tensor([u_test_t[t]], dtype=torch.float32)
        with torch.no_grad():
            if use_evolution:
                reservoir.step(x_in)
            else:
                reservoir.step_no_evolution(x_in)
            
            # Predict
            pred = torch.matmul(W_out, reservoir.state)
            y_pred.append(pred.item())
            
    y_pred = np.array(y_pred)
    
    # Calculate RMSE and MAPE
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1e-5))))
    return y_pred, rmse, mape

def main():
    print("=" * 70)
    print("  Time-Series Comparison: Spike-ESN vs. Normal Reservoir (Tesi)")
    print("=" * 70)

    n_total = 3000
    data = normalise(generate_mackey_glass(n_steps=n_total, seed=42))

    washout = 200
    train_end = 2000

    # Let's test on prediction horizons tau = 1, 5, 10, 20
    for tau in [1, 5, 10, 20]:
        print(f"\n{'─' * 70}")
        print(f"  Prediction step τ = {tau}")
        print(f"{'─' * 70}")

        u_all = data[: n_total - tau]
        y_all = data[tau: n_total]

        u_train = u_all[:train_end]
        y_train = y_all[:train_end]
        u_test  = u_all[train_end:]
        y_test  = y_all[train_end:]

        # 1. Spike-ESN
        spike_esn = SpikeESN(
            N_res=100, N_sam=50, rho=0.9, eta=0.1,
            mu=1e-8, psi=5000, input_scaling=0.8, seed=42, device="cpu"
        )
        t0 = time.time()
        spike_esn.fit(u_train, y_train, washout=washout)
        y_pred_spike = spike_esn.predict(u_test).cpu().numpy()
        t_spike = time.time() - t0
        rmse_spike = SpikeESN.rmse(y_test, y_pred_spike)
        mape_spike = SpikeESN.mape(y_test, y_pred_spike)
        print(f"[Spike-ESN]       RMSE: {rmse_spike:.6f}  |  MAPE: {mape_spike:.6f}  |  Time: {t_spike:.2f}s")

        # 2. Tesi Normal Reservoir (Static - No Evolution)
        t0 = time.time()
        _, rmse_tesi_static, mape_tesi_static = run_tesi_reservoir(
            u_train, y_train, u_test, y_test, 
            res_size=100, density=0.1, mu=1e-8, input_scaling=0.8, 
            washout=washout, use_evolution=False
        )
        t_static = time.time() - t0
        print(f"[Tesi Static]     RMSE: {rmse_tesi_static:.6f}  |  MAPE: {mape_tesi_static:.6f}  |  Time: {t_static:.2f}s")

        # 3. Tesi Normal Reservoir (With Plasticity & Evolution)
        t0 = time.time()
        _, rmse_tesi_evo, mape_tesi_evo = run_tesi_reservoir(
            u_train, y_train, u_test, y_test, 
            res_size=100, density=0.1, mu=1e-8, input_scaling=0.8, 
            washout=washout, use_evolution=True
        )
        t_evo = time.time() - t0
        print(f"[Tesi Plasticity] RMSE: {rmse_tesi_evo:.6f}  |  MAPE: {mape_tesi_evo:.6f}  |  Time: {t_evo:.2f}s")

if __name__ == "__main__":
    main()
