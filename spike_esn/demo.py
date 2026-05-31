#!/usr/bin/env python3
"""
Demo script for the Spike-ESN implementation.

Demonstrates the Spike-ESN model on the Mackey-Glass chaotic time series
and compares it against the baseline ESN model, replicating the experimental
methodology from the paper.

Usage:
    .venv/bin/python -m spike_esn.demo
"""

import numpy as np
import time


def generate_mackey_glass(n_steps=3000, tau_mg=17, delta_t=0.1, seed=42):
    """Generate the Mackey-Glass chaotic time series.

    Standard benchmark for reservoir computing — a nonlinear delay
    differential equation producing deterministic chaos.
    """
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
    """Min-max normalisation to [0, 1]."""
    dmin, dmax = data.min(), data.max()
    if dmax == dmin:
        return np.zeros_like(data)
    return (data - dmin) / (dmax - dmin)


def main():
    from spike_esn import SpikeESN
    from spike_esn.baseline_esn import ESN

    print("=" * 60)
    print("  Spike-ESN Demo — Mackey-Glass Time Series Prediction")
    print("=" * 60)

    # ---- Generate and normalise data ----
    n_total = 3000
    data = normalise(generate_mackey_glass(n_steps=n_total, seed=42))

    # Paper-style split
    washout = 200
    train_end = 2000

    for tau in [1, 5, 10, 20]:
        print(f"\n{'─' * 60}")
        print(f"  Prediction step τ = {tau}")
        print(f"{'─' * 60}")

        # Input u(t) predicts y(t) = u(t + τ)
        u_all = data[: n_total - tau]
        y_all = data[tau: n_total]

        u_train = u_all[:train_end]
        y_train = y_all[:train_end]
        u_test  = u_all[train_end:]
        y_test  = y_all[train_end:]

        # ---- Spike-ESN (paper parameters) ----
        print("\n  [Spike-ESN]  N_sam=50, ψ=5000")
        t0 = time.time()
        spike_esn = SpikeESN(
            N_res=100, N_sam=50, rho=0.9, eta=0.1,
            mu=1e-8, psi=5000, input_scaling=0.8, seed=42,
        )
        spike_esn.fit(u_train, y_train, washout=washout)
        y_pred_spike = spike_esn.predict(u_test)
        t_spike = time.time() - t0

        n = min(len(y_test), len(y_pred_spike))
        rmse_s = SpikeESN.rmse(y_test[:n], y_pred_spike[:n])
        mape_s = SpikeESN.mape(y_test[:n], y_pred_spike[:n])
        print(f"    RMSE = {rmse_s:.6f}")
        print(f"    MAPE = {mape_s:.6f}")
        print(f"    Time = {t_spike:.2f}s")

        # ---- Baseline ESN ----
        print("\n  [Baseline ESN]")
        t0 = time.time()
        esn = ESN(
            N_res=100, rho=0.9, eta=0.1,
            mu=1e-8, input_scaling=0.8, seed=42,
        )
        esn.fit(u_train, y_train, washout=washout)
        y_pred_esn = esn.predict(u_test)
        t_esn = time.time() - t0

        n = min(len(y_test), len(y_pred_esn))
        rmse_e = ESN.rmse(y_test[:n], y_pred_esn[:n])
        mape_e = ESN.mape(y_test[:n], y_pred_esn[:n])
        print(f"    RMSE = {rmse_e:.6f}")
        print(f"    MAPE = {mape_e:.6f}")
        print(f"    Time = {t_esn:.2f}s")

        # ---- Comparison ----
        if rmse_e > 0:
            imp = ((rmse_e - rmse_s) / rmse_e) * 100
            print(f"\n  → Spike-ESN RMSE improvement over ESN: {imp:+.2f}%")

    print(f"\n{'=' * 60}")
    print("  Demo complete.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
