#!/usr/bin/env python3
"""Evaluate HybridReservoir vs Spike-ESN vs Tesi ESN on both
Mackey-Glass (continuous time series) and Tiny Shakespeare (next-char)."""

import os
import sys
import time
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hybrid_reservoir.model import HybridReservoir
from spike_esn.model import SpikeESN
from tesi.model.reservoir import Reservoir

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

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

def load_shakespeare(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for i, c in enumerate(chars)}
    data = np.array([char_to_int[c] for c in text], dtype=np.float64)
    return data, chars, char_to_int, int_to_char

# ---------------------------------------------------------------------------
# Mackey-Glass
# ---------------------------------------------------------------------------

def evaluate_mackey_glass():
    print("\n" + "=" * 80)
    print("  EVALUATION 1: MACKEY-GLASS TIME SERIES PREDICTION")
    print("=" * 80)

    n_total = 3000
    data = normalise(generate_mackey_glass(n_steps=n_total, seed=42))
    washout = 200
    train_end = 2000

    for tau in [1, 5, 10, 20]:
        print(f"\nPrediction step τ = {tau}")
        print("-" * 60)

        u_all = data[: n_total - tau]
        y_all = data[tau: n_total]
        u_train = u_all[:train_end]
        y_train = y_all[:train_end]
        u_test  = u_all[train_end:]
        y_test  = y_all[train_end:]

        # ── Spike-ESN ──────────────────────────────────────────────
        spike_esn = SpikeESN(
            N_res=100, N_sam=50, rho=0.9, eta=0.1,
            mu=1e-8, psi=5000, input_scaling=0.8, seed=42, device="cpu",
        )
        spike_esn.fit(u_train, y_train, washout=washout)
        y_pred_spike = spike_esn.predict(u_test).cpu().numpy()
        rmse_spike = SpikeESN.rmse(y_test, y_pred_spike)

        # ── Tesi Static ───────────────────────────────────────────
        tesi = Reservoir(res_size=100, output_dim=1, input_scaling=0.8, density=0.1)
        tesi.reset_state()
        X_tr = []
        for t in range(len(u_train)):
            x_in = torch.tensor([u_train[t]], dtype=torch.float32)
            with torch.no_grad():
                tesi.step_no_evolution(x_in)
            if t >= washout:
                X_tr.append(tesi.state.clone())
        X_tr = torch.stack(X_tr)
        A = X_tr.T @ X_tr + 1e-8 * torch.eye(100)
        B = X_tr.T @ torch.tensor(y_train[washout:], dtype=torch.float32).unsqueeze(1)
        W_out = torch.linalg.solve(A, B).T

        y_pred_tesi = []
        for t in range(len(u_test)):
            x_in = torch.tensor([u_test[t]], dtype=torch.float32)
            with torch.no_grad():
                tesi.step_no_evolution(x_in)
                y_pred_tesi.append((W_out @ tesi.state).item())
        rmse_tesi = float(np.sqrt(np.mean((y_test - np.array(y_pred_tesi)) ** 2)))

        # ── Hybrid (no feedback) ──────────────────────────────────
        t0 = time.time()
        hybrid = HybridReservoir(
            res_size=100, output_dim=1, input_scaling=0.8, rho=0.95,
            density=0.1, synaptic_decay=0.9, mu=1e-8, device="cpu",
        )
        hybrid.fit(u_train, y_train, washout=washout)
        y_pred_h = hybrid.predict(u_test).detach().numpy()
        t_h = time.time() - t0
        rmse_hybrid = float(np.sqrt(np.mean((y_test - y_pred_h.flatten()) ** 2)))

        # ── Hybrid + Feedback ─────────────────────────────────────
        t0 = time.time()
        hybrid_fb = HybridReservoir(
            res_size=100, output_dim=1, input_scaling=0.8, rho=0.95,
            density=0.1, synaptic_decay=0.9, mu=1e-8,
            feedback_steps=30, feedback_lr=10.0, device="cpu",
        )
        hybrid_fb.fit(u_train, y_train, washout=washout)
        y_pred_hfb = hybrid_fb.predict(u_test).detach().numpy()
        t_hfb = time.time() - t0
        rmse_hybrid_fb = float(np.sqrt(np.mean((y_test - y_pred_hfb.flatten()) ** 2)))

        print(f"  {'Model':<26} {'RMSE':>10} {'Time':>8}")
        print(f"  {'-'*46}")
        print(f"  {'Spike-ESN':<26} {rmse_spike:>10.6f}")
        print(f"  {'Tesi Static':<26} {rmse_tesi:>10.6f}")
        print(f"  {'Hybrid':<26} {rmse_hybrid:>10.6f} {t_h:>7.2f}s")
        print(f"  {'Hybrid + Feedback':<26} {rmse_hybrid_fb:>10.6f} {t_hfb:>7.2f}s")

# ---------------------------------------------------------------------------
# Tiny Shakespeare
# ---------------------------------------------------------------------------

def evaluate_shakespeare():
    print("\n" + "=" * 80)
    print("  EVALUATION 2: TINY SHAKESPEARE NEXT-CHARACTER PREDICTION")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, "tinyshakespeare.txt")
    if not os.path.exists(filepath):
        print(f"[ERROR] Data file not found at {filepath}")
        return

    data, chars, char_to_int, int_to_char = load_shakespeare(filepath)
    vocab_size = len(chars)
    data_int = data.astype(int)

    train_len = 10000
    test_len = 1000
    washout = 200
    res_size = 500

    print(f"  Vocab: {vocab_size}  |  Train: {train_len}  |  Test: {test_len}  |  N_res: {res_size}")

    # ── Tesi ESN ──────────────────────────────────────────────────
    print("  Fitting Tesi ESN ...")
    t0 = time.time()
    tesi = Reservoir(res_size=res_size, output_dim=vocab_size,
                     input_scaling=1.0, density=0.01)
    tesi.reset_state()
    X_tr, Y_tr = [], []
    for t in range(train_len):
        x_oh = torch.zeros(vocab_size); x_oh[data_int[t]] = 1.0
        with torch.no_grad():
            tesi.step(x_oh)
        if t >= washout:
            X_tr.append(tesi.state.clone())
            y_oh = torch.zeros(vocab_size); y_oh[data_int[t + 1]] = 1.0
            Y_tr.append(y_oh)
    X_tr = torch.stack(X_tr); Y_tr = torch.stack(Y_tr)
    A = X_tr.T @ X_tr + 1e-4 * torch.eye(res_size)
    W_out = (torch.linalg.solve(A, X_tr.T @ Y_tr)).T

    correct = 0
    for t in range(train_len, train_len + test_len):
        x_oh = torch.zeros(vocab_size); x_oh[data_int[t]] = 1.0
        with torch.no_grad():
            tesi.step_no_evolution(x_oh)
            if torch.argmax(W_out @ tesi.state).item() == data_int[t + 1]:
                correct += 1
    acc_tesi = correct / test_len
    t_tesi = time.time() - t0

    # ── Hybrid Reservoir ──────────────────────────────────────────
    print("  Fitting Hybrid Reservoir ...")
    t0 = time.time()
    hybrid = HybridReservoir(
        res_size=res_size, output_dim=vocab_size, input_scaling=1.0,
        rho=0.95, density=0.01, synaptic_decay=1.0, mu=1e-4,
        device="cpu",
    )
    u_train = torch.zeros((train_len, vocab_size))
    y_train = torch.zeros((train_len, vocab_size))
    for t in range(train_len):
        u_train[t, data_int[t]] = 1.0
        y_train[t, data_int[t + 1]] = 1.0
    hybrid.fit(u_train, y_train, washout=washout)

    u_test = torch.zeros((test_len, vocab_size))
    for t in range(test_len):
        u_test[t, data_int[train_len + t]] = 1.0
    y_pred = hybrid.predict(u_test).detach()
    correct_h = sum(
        1 for t in range(test_len)
        if torch.argmax(y_pred[t]).item() == data_int[train_len + t + 1]
    )
    acc_hybrid = correct_h / test_len
    t_hybrid = time.time() - t0

    print(f"\n  {'Model':<26} {'Accuracy':>10} {'Time':>8}")
    print(f"  {'-'*46}")
    print(f"  {'Tesi ESN':<26} {acc_tesi*100:>9.2f}% {t_tesi:>7.1f}s")
    print(f"  {'Hybrid Reservoir':<26} {acc_hybrid*100:>9.2f}% {t_hybrid:>7.1f}s")

# ---------------------------------------------------------------------------

def main():
    evaluate_mackey_glass()
    evaluate_shakespeare()

if __name__ == "__main__":
    main()
