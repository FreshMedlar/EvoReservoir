"""
online_train.py — Online evolutionary fine-tuning of a pre-trained Spike-ESN.

After offline ridge-regression pre-training (train_shakespeare.py --save),
this script continuously searches for rank-1 perturbations of W_out and W_res
that improve autoregressive character prediction, then applies a
fitness-weighted average update.

Key ideas
---------
* Rank-1 compression: W_out (R×C) ≈ a·bᵀ  →  R+C parameters instead of R×C.
  W_res (N×N) ≈ a·bᵀ  →  2N parameters instead of N².
* Fitness: length of the longest consecutive streak of correct next-char
  predictions (teacher-forced), stopping at the first mistake.
* Anti-overfitting: require at least --min-winners candidates to beat the
  baseline before applying any update.
* Update: fitness-weighted average of the winning rank-1 factors.

Usage (from project root)
--------------------------
    PYTHONPATH=. .venv/bin/python spike_esn/online_train.py \\
        --model models/esn_pretrained.pkl \\
        --min-winners 5 --sigma 0.01 --washout 100 --max-eval-len 500
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Rank-1 additive perturbation helpers
# ---------------------------------------------------------------------------
# Strategy: instead of *replacing* W with outer(a, b), we keep W_base frozen
# and search over a rank-1 *delta*:
#
#   W_candidate = W_base + outer(u, v)
#
# Search space is 2N floats (the delta vectors u, v) instead of R×C.
# The base echo-state / spectral-radius properties of W_res are preserved,
# and winners are far more likely than when replacing the full matrix.
# ---------------------------------------------------------------------------

def sample_delta(
    n_rows: int, n_cols: int, sigma: float, rng: np.random.Generator
) -> tuple[NDArray, NDArray]:
    """Sample a random rank-1 delta: outer(u, v) with u~N(0,σ), v~N(0,σ).

    Returns u ∈ ℝⁿ_ʳᵒʷˢ, v ∈ ℝⁿ_ᶜᵒˡˢ.
    """
    u = rng.normal(0.0, sigma, size=n_rows)
    v = rng.normal(0.0, sigma, size=n_cols)
    return u, v


def apply_delta(W_base: NDArray, u: NDArray, v: NDArray) -> NDArray:
    """Return W_base + outer(u, v)."""
    return W_base + np.outer(u, v)


# ---------------------------------------------------------------------------
# Shared input projection (candidate-independent)
# ---------------------------------------------------------------------------

def precompute_win_proj(
    spike_window: NDArray,
    W_in: NDArray,
    spike_kernel: NDArray,
    N_sam: int,
) -> NDArray:
    """Compute W_in @ f_spike for every time step in the window at once.

    Since W_in is fixed across all candidates, this cost is paid only once
    per evolutionary round.

    Returns
    -------
    win_proj : ndarray of shape (N_res, T_window)
    """
    T = spike_window.shape[0]
    n_channels = spike_window.shape[1] // N_sam

    spikes_3d = spike_window.reshape(T, n_channels, N_sam)
    f_spike_3d = spikes_3d @ spike_kernel          # (T, C, N_sam)
    f_spike_all = f_spike_3d.reshape(T, n_channels * N_sam)

    return W_in @ f_spike_all.T  # (N_res, T)


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------

def evaluate_streak(
    W_out_c: NDArray,
    W_res_c: NDArray,
    win_proj: NDArray,
    targets_int: NDArray,
    washout: int,
    vocab_size: int,
    encoding: str,
    max_eval_len: int,
) -> int:
    """Drive the reservoir with a candidate's W_res; return the streak length.

    The reservoir is warmed up for `washout` steps (no scoring).
    After warmup, we predict the next character at each step (teacher-forced
    input, so the reservoir receives the *true* character's spike encoding).
    We stop at the first wrong prediction or after `max_eval_len` steps.

    Parameters
    ----------
    W_out_c     : candidate readout matrix (1×N or V×N)
    W_res_c     : candidate recurrent matrix (N×N)
    win_proj    : precomputed W_in @ f_spike  (N_res, T_window)
    targets_int : true next-char indices, shape (T_window,)
                  targets_int[t] = char index that should be predicted after step t
    washout     : number of warm-up steps
    vocab_size  : size of vocabulary
    encoding    : 'scalar' or 'one-hot'
    max_eval_len: safety cap on evaluation length

    Returns
    -------
    streak : int — number of consecutive correct predictions before first mistake
    """
    N_res = W_res_c.shape[0]
    T = win_proj.shape[1]
    x = np.zeros(N_res, dtype=np.float64)
    streak = 0

    for t in range(T):
        x = np.tanh(win_proj[:, t] + W_res_c @ x)

        if t < washout:
            continue

        eval_step = t - washout
        if eval_step >= max_eval_len:
            break

        y_hat = W_out_c @ x  # (1,) or (V,)

        if encoding == "one-hot":
            pred_idx = int(np.argmax(y_hat))
        else:
            pred_idx = int(
                np.clip(round(float(y_hat.flat[0]) * (vocab_size - 1)),
                        0, vocab_size - 1)
            )

        if pred_idx == int(targets_int[t]):
            streak += 1
        else:
            break  # stop at first mistake

    return streak


def evaluate_nll(
    W_out_c: NDArray,
    W_res_c: NDArray,
    win_proj: NDArray,
    targets_int: NDArray,
    washout: int,
    max_eval_len: int,
) -> float:
    """Mean log-probability of the true next char under a softmax distribution.

    W_out_c must be (V, N_res) — i.e. one-hot encoding mode.  Each step t
    (after washout) computes:

        logits = W_out_c @ x(t)          # (V,)
        log_p  = log_softmax(logits)     # numerically stable
        score  += log_p[true_char]       # negative cross-entropy

    Returns the mean log-probability over all eval steps (higher is better,
    maximum is 0 meaning perfect certainty on every character).
    This is a smooth, continuous signal — every candidate gets a meaningful
    score even when no character is 'correctly' predicted.

    Parameters
    ----------
    W_out_c     : (V, N_res) readout matrix
    W_res_c     : (N_res, N_res) recurrent matrix
    win_proj    : precomputed W_in @ f_spike  (N_res, T_window)
    targets_int : true next-char indices, shape (T_window,)
    washout     : warm-up steps before scoring
    max_eval_len: safety cap on number of scored steps

    Returns
    -------
    mean_logp : float — mean log-probability in (-inf, 0]; higher is better.
    """
    N_res = W_res_c.shape[0]
    T = win_proj.shape[1]
    x = np.zeros(N_res, dtype=np.float64)
    total_logp = 0.0
    n_eval = 0

    for t in range(T):
        x = np.tanh(win_proj[:, t] + W_res_c @ x)

        if t < washout:
            continue

        eval_step = t - washout
        if eval_step >= max_eval_len:
            break

        logits = W_out_c @ x             # (V,)
        # Numerically stable log-softmax
        logits = logits - logits.max()
        log_probs = logits - np.log(np.sum(np.exp(logits)))

        total_logp += log_probs[int(targets_int[t])]
        n_eval += 1

    return total_logp / n_eval if n_eval > 0 else float("-inf")


# ---------------------------------------------------------------------------
# Weighted-average update
# ---------------------------------------------------------------------------

def weighted_update(
    winners: list[dict],
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Compute fitness-weighted average of the winning candidates' delta vectors.

    Weights are computed via softmax over the raw fitness scores.
    Returns the accumulated (u_out, v_out, u_res, v_res) delta vectors
    that should be added to the base matrices.
    """
    fitnesses = np.array([w["fitness"] for w in winners], dtype=np.float64)
    fitnesses -= fitnesses.max()          # numerically stable softmax
    weights = np.exp(fitnesses)
    weights /= weights.sum()

    u_out = sum(wt * c["u_out"] for wt, c in zip(weights, winners))
    v_out = sum(wt * c["v_out"] for wt, c in zip(weights, winners))
    u_res = sum(wt * c["u_res"] for wt, c in zip(weights, winners))
    v_res = sum(wt * c["v_res"] for wt, c in zip(weights, winners))

    return u_out, v_out, u_res, v_res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Online evolutionary fine-tuning of a pre-trained Spike-ESN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, metavar="PATH",
                   help="Path to the pre-trained model .pkl file")
    p.add_argument("--data", type=str, default=None, metavar="PATH",
                   help="Text corpus file. Defaults to tinyshakespeare.txt "
                        "next to the project root.")
    p.add_argument("--data-offset", type=int, default=None, metavar="INT",
                   help="Start reading from this character index in the corpus. "
                        "Defaults to model._online_data_offset if set, else 0.")
    p.add_argument("--washout", type=int, default=100,
                   help="Warm-up steps before scoring each candidate.")
    p.add_argument("--max-eval-len", type=int, default=500,
                   help="Safety cap on evaluation length (streak cannot exceed this).")
    p.add_argument("--sigma", type=float, default=0.01,
                   help="Gaussian noise std for perturbing rank-1 factors.")
    p.add_argument("--min-winners", type=int, default=5,
                   help="Minimum number of candidates that must beat the baseline "
                        "before an update is applied (anti-overfitting).")
    p.add_argument("--max-rounds", type=int, default=0,
                   help="Stop after this many successful update rounds. "
                        "0 = run until Ctrl-C.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for mutation RNG.")
    p.add_argument("--progress-every", type=int, default=100,
                   help="Print candidate count every N evaluations (0 = silent).")
    p.add_argument("--fitness", choices=["streak", "log-prob"], default="streak",
                   help="Fitness function for evolution. "
                        "'streak': longest consecutive correct predictions (stops at "
                        "first error). "
                        "'log-prob': mean log-probability of the true next char under "
                        "softmax — continuous signal, winners are far more frequent, "
                        "requires --encoding one-hot at pre-training time.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Load pre-trained model
    # ------------------------------------------------------------------
    model_path = args.model if args.model.endswith(".pkl") else args.model + ".pkl"
    print(f"Loading model from '{model_path}'...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    vocab_size   = model._online_vocab_size
    encoding     = model._online_encoding
    int_to_char  = model._online_int_to_char
    reservoir    = model.reservoir
    encoder      = model.encoder
    N_res        = model.N_res
    N_sam        = model.N_sam

    print(f"  N_res={N_res}, N_sam={N_sam}, encoding={encoding}, vocab={vocab_size}")

    # Validate fitness / encoding combination
    if args.fitness == "log-prob" and encoding != "one-hot":
        print(
            "[ERROR] --fitness log-prob requires --encoding one-hot at pre-training time.\n"
            "        Re-run train_shakespeare.py with --encoding one-hot --save <path>,\n"
            "        then retry online_train.py with that model."
        )
        raise SystemExit(1)
    use_logprob = args.fitness == "log-prob"
    fitness_label = "log-prob" if use_logprob else "streak"

    # ------------------------------------------------------------------
    # Load corpus
    # ------------------------------------------------------------------
    if args.data is not None:
        corpus_path = args.data
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        corpus_path = os.path.join(base_dir, "tinyshakespeare.txt")

    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Build index arrays using the vocabulary stored in the model
    char_to_int = model._online_char_to_int
    data_int = np.array([char_to_int[c] for c in text if c in char_to_int],
                        dtype=np.int64)
    data_norm = data_int / (vocab_size - 1)

    # ------------------------------------------------------------------
    # Data offset
    # ------------------------------------------------------------------
    if args.data_offset is not None:
        data_offset = args.data_offset
    elif hasattr(model, "_online_data_offset"):
        data_offset = model._online_data_offset
        print(f"  Resuming from saved data_offset={data_offset}")
    else:
        data_offset = 0
        print("  No data offset found — starting from the beginning of the corpus.")

    # ------------------------------------------------------------------
    # Base weight matrices (kept as reference; deltas are evolved)
    # ------------------------------------------------------------------
    W_out_base = model.W_out.copy()      # (1, N_res) or (V, N_res)
    W_res_base = reservoir.W_res.copy()  # (N_res, N_res)

    # Accumulated delta vectors (start at zero — no perturbation yet)
    u_out = np.zeros(W_out_base.shape[0], dtype=np.float64)  # (1,) or (V,)
    v_out = np.zeros(W_out_base.shape[1], dtype=np.float64)  # (N_res,)
    u_res = np.zeros(W_res_base.shape[0], dtype=np.float64)  # (N_res,)
    v_res = np.zeros(W_res_base.shape[1], dtype=np.float64)  # (N_res,)

    out_delta_params = u_out.size + v_out.size
    res_delta_params = u_res.size + v_res.size
    print(f"\nAdditive rank-1 delta search space:")
    print(f"  W_out delta: {W_out_base.shape} → {out_delta_params} params "
          f"(vs {W_out_base.size} full)")
    print(f"  W_res delta: {W_res_base.shape} → {res_delta_params} params "
          f"(vs {W_res_base.size} full)")
    print(f"  Total: {out_delta_params + res_delta_params} params\n")

    # RNG (separate streams for mutation and encoding)
    rng_mut = np.random.default_rng(args.seed)
    rng_enc = np.random.default_rng(args.seed + 1)

    # One-hot helper (imported lazily to keep dependency on train_shakespeare optional)
    if encoding == "one-hot":
        from spike_esn.train_shakespeare import build_one_hot_spike_matrix

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    round_num = 0
    print("Starting online evolutionary training. Press Ctrl-C to stop.\n")

    try:
        while True:
            round_num += 1

            # ---- Wrap-around if we hit the end of the corpus ----
            window_need = args.washout + args.max_eval_len + 1  # +1 for targets
            if data_offset + window_need > len(data_int):
                print("[!] Reached end of corpus — wrapping to offset 0.")
                data_offset = 0

            # ---- Reconstruct current weight matrices ----
            W_out_cur = apply_delta(W_out_base, u_out, v_out)
            W_res_cur = apply_delta(W_res_base, u_res, v_res)

            # ---- Encode input window once (shared by all candidates) ----
            window_end = data_offset + args.washout + args.max_eval_len
            input_int  = data_int[data_offset : window_end]
            # targets_int[t] = true next char at step t (used after washout)
            targets_int = data_int[data_offset + 1 : window_end + 1]

            if encoding == "one-hot":
                spike_window = build_one_hot_spike_matrix(
                    input_int, vocab_size, encoder, rng_enc, deterministic=False
                )
            else:
                input_norm = data_norm[data_offset : window_end]
                spike_window = encoder.encode_series(input_norm, rng=rng_enc)

            # Precompute W_in projection (fixed — W_in is NOT evolved)
            win_proj = precompute_win_proj(
                spike_window, reservoir.W_in, reservoir._spike_kernel, N_sam
            )  # (N_res, T_window)

            # ---- Baseline fitness ----
            if use_logprob:
                baseline = evaluate_nll(
                    W_out_cur, W_res_cur, win_proj, targets_int,
                    args.washout, args.max_eval_len,
                )
            else:
                baseline = evaluate_streak(
                    W_out_cur, W_res_cur, win_proj, targets_int,
                    args.washout, vocab_size, encoding, args.max_eval_len,
                )

            seed_char = int_to_char.get(int(input_int[0]), "?")
            fmt_baseline = f"{baseline:.4f}" if use_logprob else str(int(baseline))
            print(f"[Round {round_num:4d}] offset={data_offset:7d} | "
                  f"baseline {fitness_label}={fmt_baseline} | "
                  f"seed='{seed_char}'")

            # ---- Evolution: collect winners ----
            winners: list[dict] = []
            n_evaluated = 0
            t_round_start = time.perf_counter()

            while len(winners) < args.min_winners:
                # Sample a fresh rank-1 delta for each candidate
                u_out_c, v_out_c = sample_delta(
                    W_out_base.shape[0], W_out_base.shape[1], args.sigma, rng_mut
                )
                u_res_c, v_res_c = sample_delta(
                    W_res_base.shape[0], W_res_base.shape[1], args.sigma, rng_mut
                )
                W_out_c = apply_delta(W_out_base, u_out + u_out_c, v_out + v_out_c)
                W_res_c = apply_delta(W_res_base, u_res + u_res_c, v_res + v_res_c)

                if use_logprob:
                    fitness_c = evaluate_nll(
                        W_out_c, W_res_c, win_proj, targets_int,
                        args.washout, args.max_eval_len,
                    )
                else:
                    fitness_c = evaluate_streak(
                        W_out_c, W_res_c, win_proj, targets_int,
                        args.washout, vocab_size, encoding, args.max_eval_len,
                    )
                n_evaluated += 1

                if fitness_c > baseline:
                    winners.append({
                        "fitness": fitness_c,
                        "u_out":   u_out + u_out_c,
                        "v_out":   v_out + v_out_c,
                        "u_res":   u_res + u_res_c,
                        "v_res":   v_res + v_res_c,
                    })
                    fmt_fit = f"{fitness_c:.4f}" if use_logprob else str(int(fitness_c))
                    print(f"  Winner {len(winners):2d}/{args.min_winners}: "
                          f"{fitness_label}={fmt_fit} (eval #{n_evaluated})", flush=True)

                elif args.progress_every > 0 and n_evaluated % args.progress_every == 0:
                    elapsed = time.perf_counter() - t_round_start
                    print(f"  ... {n_evaluated:5d} candidates | "
                          f"{len(winners)}/{args.min_winners} winners | "
                          f"{elapsed:.0f}s", flush=True)

            t_elapsed = time.perf_counter() - t_round_start
            print(f"  → {len(winners)} winners from {n_evaluated} candidates "
                  f"in {t_elapsed:.1f}s")

            # ---- Weighted-average update ----
            u_out, v_out, u_res, v_res = weighted_update(winners)

            # ---- Write updated weights back to model ----
            model.W_out     = apply_delta(W_out_base, u_out, v_out)
            reservoir.W_res = apply_delta(W_res_base, u_res, v_res)
            # Roll the winning delta into the base so next round continues from here
            W_out_base = model.W_out.copy()
            W_res_base = reservoir.W_res.copy()
            u_out[:] = 0.0; v_out[:] = 0.0
            u_res[:] = 0.0; v_res[:] = 0.0

            # ---- Slide window forward ----
            # Advance past the warmup and the evaluated streak so the next
            # round sees fresh text.
            data_offset += args.washout + max(baseline, 1)
            model._online_data_offset = data_offset

            # ---- Save checkpoint ----
            with open(model_path, "wb") as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  Saved → '{model_path}' | next offset={data_offset}\n")

            # ---- Check stop condition ----
            if args.max_rounds > 0 and round_num >= args.max_rounds:
                print(f"Reached --max-rounds={args.max_rounds}. Done.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user. Final model has been saved.")


if __name__ == "__main__":
    main()
