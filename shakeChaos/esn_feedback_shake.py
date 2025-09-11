# esn_char_predict.py
# Echo State Network for next-character prediction on tinyshakespear.txt
# Implements "ESN with feedback" (u -> u + V @ x) and batch gradient descent
# for V following Appendix A of "Improving the Performance of Echo State Networks Through State Feedback".
#
# Usage (example):
# python esn_char_predict.py --path tinyshakespeare.txt --n_res 200 --feedback --fb_steps 20 --fb_lr 1e-3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import math

def softmax(logits):
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / np.sum(exp)

def evaluate_free_running(
    indices,
    vocab_size,
    Win, W, Wout, V=None,
    alpha=0.3,
    include_input_in_readout=True,
    seed_len=200,
    gen_len=1000,
):
    """
    Free-running evaluation: generate text by feeding back own predictions.
    Compute char-level perplexity against the true test sequence.
    
    Args:
        indices: np.array of test indices (int)
        vocab_size: number of chars
        Win, W, Wout, V: ESN parameters
        alpha: leaking rate
        seed_len: length of seed prefix from test set
        gen_len: number of characters to generate/evaluate
    Returns:
        perplexity (float), generated string
    """
    n_res = W.shape[0]
    I = np.eye(vocab_size, dtype=np.float32)
    x = np.zeros(n_res, dtype=np.float32)
    bias = np.array([1.0], dtype=np.float32)

    # seed with ground-truth
    seed_idx = indices[:seed_len]
    u = I[seed_idx[0]]

    # run seed into reservoir
    for t in range(seed_len-1):
        v_idx = seed_idx[t+1]
        if V is None:
            fb = np.zeros(vocab_size, dtype=np.float32)
        else:
            fb = (V @ x).astype(np.float32)
        in_vec = np.concatenate([bias, (u + fb).astype(np.float32)], dtype=np.float32)
        preact = Win @ in_vec + W @ x
        x = (1.0 - alpha) * x + alpha * np.tanh(preact)
        u = I[v_idx]  # teacher forcing only for the seed phase

    # now free-run
    generated = []
    log_probs = []
    u_idx = seed_idx[-1]
    u = I[u_idx]
    for t in range(gen_len):
        if V is None:
            fb = np.zeros(vocab_size, dtype=np.float32)
        else:
            fb = (V @ x).astype(np.float32)
        in_vec = np.concatenate([bias, (u + fb).astype(np.float32)], dtype=np.float32)
        preact = Win @ in_vec + W @ x
        x = (1.0 - alpha) * x + alpha * np.tanh(preact)

        # features for readout
        if include_input_in_readout:
            z = np.concatenate([bias, u, x], dtype=np.float32)
        else:
            z = np.concatenate([bias, x], dtype=np.float32)

        logits = Wout @ z
        probs = softmax(logits)

        # predicted char
        pred_idx = int(np.argmax(probs))
        generated.append(pred_idx)

        # true next char (from test sequence)
        true_idx = indices[seed_len + t] if seed_len + t < len(indices) else None
        if true_idx is not None:
            log_probs.append(math.log(probs[true_idx] + 1e-12))  # avoid log(0)

        # update input u for next step (feed prediction back)
        u = I[pred_idx]

    # perplexity over available true targets
    if len(log_probs) > 0:
        avg_logp = sum(log_probs) / len(log_probs)
        perplexity = math.exp(-avg_logp)
    else:
        perplexity = float("inf")

    return perplexity, generated


def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return chars, stoi, itos

def text_to_indices(text, stoi):
    return np.array([stoi[ch] for ch in text], dtype=np.int32)

def init_reservoir(n_in, n_res, density, spectral_radius, input_scale, seed):
    rng = np.random.default_rng(seed)
    # Input weights include bias; shape: n_res x (1 + n_in)
    Win = (rng.standard_normal((n_res, 1 + n_in)).astype(np.float32)) * input_scale
    # Reservoir weights sparse-ish dense matrix
    W = np.zeros((n_res, n_res), dtype=np.float32)
    nnz = int(density * n_res * n_res)
    idx_i = rng.integers(0, n_res, size=nnz)
    idx_j = rng.integers(0, n_res, size=nnz)
    vals = rng.standard_normal(nnz).astype(np.float32)
    W[idx_i, idx_j] = vals
    # Scale to desired spectral radius
    try:
        eigvals = np.linalg.eigvals(W.astype(np.float64))
        sr = np.max(np.abs(eigvals)).real
        if sr > 0:
            W *= (spectral_radius / sr)
    except Exception:
        # fallback: power iteration approx if eig fails
        W = W * (spectral_radius / max(1e-12, np.sqrt((W**2).sum())))
    return Win, W

def ridge_readout(Z, Y, reg):
    # Solve Wout = Y Z^T (Z Z^T + reg I)^{-1}
    # Z: n_feat x N, Y: n_out x N
    n_feat = Z.shape[0]
    ZZt = Z @ Z.T
    A = ZZt + reg * np.eye(n_feat, dtype=np.float32)
    # Solve for inv(A) @ (Z @ Y.T)? We want Wout shape (n_out, n_feat)
    # More numerically stable: solve linear system A X = Z @ Y.T.T? We'll compute using solve
    # Compute YZt = Y @ Z.T
    YZt = Y @ Z.T
    # Solve A.T x = YZt.T then transpose
    X = np.linalg.solve(A.T, YZt.T).T.astype(np.float32)
    return X  # shape (n_out, n_feat)

def run_esn_forward_collect(
    indices,
    vocab_size,
    Win, W,
    V=None,
    alpha=0.3,
    washout=100,
    include_input_in_readout=True,
):
    """
    Runs ESN forward on indices sequence and collects features & states.
    Returns:
      Z: n_feat x T_eff (features for regression)
      Xs: n_res x T_eff (reservoir states after update at each step)
      Y: n_out x T_eff (one-hot targets for next char)
      preacts_list: list of preact vectors (for activation derivatives)
    """
    n_res = W.shape[0]
    n_in = vocab_size
    n_out = vocab_size
    I = np.eye(vocab_size, dtype=np.float32)
    x = np.zeros(n_res, dtype=np.float32)
    bias = np.array([1.0], dtype=np.float32)

    # Feature size
    if include_input_in_readout:
        n_feat = 1 + n_in + n_res
    else:
        n_feat = 1 + n_res

    # Iterate and collect after washout
    T = len(indices) - 1
    Z_cols = []
    Xs = []
    Ys = []
    preacts = []

    Win_u = Win[:, 1:]  # n_res x n_in

    for t in range(T):
        u_idx = indices[t]
        v_idx = indices[t + 1]
        u = I[u_idx]

        # Feedback: if V provided, compute feedback = V @ x, else zero
        if V is None:
            fb = np.zeros(n_in, dtype=np.float32)
        else:
            # V shape (n_in, n_res); x shape (n_res,)
            fb = (V @ x).astype(np.float32)

        in_vec = np.concatenate([bias, (u + fb).astype(np.float32)], dtype=np.float32)
        preact = Win @ in_vec + W @ x
        x = (1.0 - alpha) * x + alpha * np.tanh(preact)

        if t >= washout:
            if include_input_in_readout:
                z = np.concatenate([bias, u, x], dtype=np.float32)
            else:
                z = np.concatenate([bias, x], dtype=np.float32)
            Z_cols.append(z)
            Xs.append(x.copy())
            Ys.append(I[v_idx])
            preacts.append(preact.copy())

    if len(Z_cols) == 0:
        return np.zeros((n_feat, 0), dtype=np.float32), np.zeros((n_res, 0), dtype=np.float32), np.zeros((n_out, 0), dtype=np.float32), []
    Z = np.stack(Z_cols, axis=1).astype(np.float32)  # n_feat x N_eff
    Xs = np.stack(Xs, axis=1).astype(np.float32)    # n_res x N_eff
    Y = np.stack(Ys, axis=1).astype(np.float32)     # n_out x N_eff
    return Z, Xs, Y, preacts

def compute_gradient_V_batch(
    Win, W, V, Xs, preacts, Z, Y, Wout, alpha=0.3, include_input_in_readout=True
):
    """
    Compute gradient dS/dV for MSE loss S = (1/(2N)) sum ||y - y_pred||^2
    Following Appendix A recursion (generalized for vector input).
    - Win: n_res x (1 + n_in)
    - W: n_res x n_res
    - V: n_in x n_res
    - Xs: n_res x N (states after update)
    - preacts: list length N of preactivation vectors (Win @ [1; u+Vx] + W@x)
    - Z: feature matrix n_feat x N
    - Y: targets n_out x N
    - Wout: n_out x n_feat
    Returns gradient same shape as V (n_in x n_res)
    """
    n_res = W.shape[0]
    n_out = Wout.shape[0]
    n_in = V.shape[0]
    N = Xs.shape[1]
    # indices where x features start in Wout
    if include_input_in_readout:
        x_start = 1 + n_in
    else:
        x_start = 1
    Wout_x = Wout[:, x_start : x_start + n_res]  # n_out x n_res

    Win_u = Win[:, 1:]  # n_res x n_in

    # Prepare D_prev (∂x_{k}/∂V) tensor: shape n_res x n_in x n_res
    D_prev = np.zeros((n_res, n_in, n_res), dtype=np.float32)

    grad = np.zeros_like(V, dtype=np.float32)  # n_in x n_res

    # iterate through collected time steps sequentially (these correspond to times after washout)
    # note: preacts[k] correspond to the preact used to compute x_k (after update)
    for k in range(N):
        x_prev = Xs[:, k].astype(np.float32)  # this is x_k (state after update)
        preact = preacts[k].astype(np.float32)
        # derivative of tanh preact: g' = 1 - tanh^2(preact). but x is (1-alpha)*x_prev + alpha*tanh(preact).
        tanh_pre = np.tanh(preact)
        gprime = (1.0 - tanh_pre * tanh_pre).astype(np.float32)  # n_res

        # term1: direct derivative from V acting directly on V @ x_prev:
        # T_dir[:, p, q] = Win_u[:, p] * x_prev[q]
        # Build Win_u[:,:,None] * x_prev[None,None,:]
        T_dir = Win_u[:, :, None] * x_prev[None, None, :]  # shape n_res x n_in x n_res

        # term2: Win_u @ (V @ D_prev)
        # First compute M = V @ D_prev  -> shape (n_in, n_in, n_res)
        # V: n_in x n_res ; D_prev: n_res x n_in x n_res
        M = np.tensordot(V, D_prev, axes=(1, 0))  # result n_in x n_in x n_res
        # Then Win_u @ M over axis 0 of M -> tensordot over Win_u axis 1 and M axis 0
        term2 = np.tensordot(Win_u, M, axes=(1, 0))  # shape n_res x n_in x n_res

        # term3: W @ D_prev (matrix multiply across the leading axis)
        term3 = np.tensordot(W, D_prev, axes=(1, 0))  # shape n_res x n_in x n_res

        Dz = T_dir + term2 + term3  # shape n_res x n_in x n_res

        # Multiply by gprime (across leading axis) and apply leaking factor
        D_k = (1.0 - alpha) * D_prev + alpha * (gprime[:, None, None] * Dz)  # n_res x n_in x n_res

        # compute y_pred at this k: using Z column and Wout (we already have predictions earlier, but recompute)
        z_k = Z[:, k]  # n_feat
        y_pred = Wout @ z_k  # n_out
        y_true = Y[:, k]
        err = (y_pred - y_true).astype(np.float32)  # shape n_out: (y_pred - y)

        # temp = Wout_x @ D_k -> shape: (n_out, n_in, n_res)
        temp = np.tensordot(Wout_x, D_k, axes=(1, 0))  # n_out x n_in x n_res

        # contribution to gradient: contraction over output axis between err and temp:
        # contrib[p, q] = sum_out err[out] * temp[out, p, q]
        contrib = np.tensordot(err, temp, axes=(0, 0))  # shape n_in x n_res

        grad += contrib  # accumulate

        D_prev = D_k

    # Normalize: paper uses average (1/N). We will match S = (1/(2N)) sum ||err||^2, derivative consistent with factor 1/N
    grad = grad / float(N)
    return grad  # shape n_in x n_res

def train_with_feedback(
    train_idx,
    vocab_size,
    Win, W,
    alpha=0.3,
    washout=100,
    fb_steps=10,
    fb_lr=1e-3,
    readout_reg=1e-6,
    include_input_in_readout=True,
    fb_stability_a=0.99,
    seed=42,
):
    """
    Trains ESN with feedback V optimized by batch gradient descent (Appendix A).
    Returns final Wout, V, and training-loss history (MSE).
    """
    rng = np.random.default_rng(seed)
    n_res = W.shape[0]
    n_in = vocab_size
    n_out = vocab_size

    # Initialize V (n_in x n_res)
    V = np.zeros((n_in, n_res), dtype=np.float32)

    # Precompute inputs for training (one-hot)
    I = np.eye(vocab_size, dtype=np.float32)
    T_total = len(train_idx) - 1
    # We'll run repeated gradient-descent iterations; at each iteration we:
    # 1) run forward (collect Z, Xs, Y, preacts) using current V
    # 2) compute ridge readout Wout from Z,Y
    # 3) compute MSE and gradient dS/dV via recursion and update V (with stability enforcement)
    losses = []

    for it in range(fb_steps):
        Z, Xs, Y, preacts = run_esn_forward_collect(
            train_idx, vocab_size, Win, W, V=V,
            alpha=alpha, washout=washout, include_input_in_readout=include_input_in_readout
        )
        if Z.shape[1] == 0:
            raise RuntimeError("No training samples after washout; reduce washout or provide more data.")

        # learn Wout via ridge
        Wout = ridge_readout(Z, Y, reg=readout_reg)

        # compute predictions and MSE
        Ypred = Wout @ Z
        errs = Ypred - Y
        mse = float(np.mean(errs * errs))
        losses.append(mse)
        print(f"[FB iter {it+1}/{fb_steps}] training MSE: {mse:.6e}")

        # compute gradient w.r.t V
        grad = compute_gradient_V_batch(Win, W, V, Xs, preacts, Z, Y, Wout, alpha=alpha, include_input_in_readout=include_input_in_readout)

        # gradient descent step
        V_new = V - fb_lr * grad

        # Stability enforcement:
        # Effective A' = W + Win[:,1:] @ V_new
        Win_u = Win[:, 1:]  # n_res x n_in
        Aprime = W + Win_u @ V_new  # n_res x n_res
        # compute largest singular value
        try:
            svals = np.linalg.svd(Aprime, compute_uv=False)
            max_s = float(np.max(np.abs(svals)))
        except Exception:
            # fallback to spectral radius approx using eigenvalues
            try:
                ev = np.linalg.eigvals(Aprime)
                max_s = float(np.max(np.abs(ev)).real)
            except Exception:
                max_s = 1.0

        # enforce that max_s <= fb_stability_a (a parameter slightly below theoretical bound)
        if max_s >= fb_stability_a:
            # scale V_new to reduce the added term Win_u @ V_new
            # simple heuristic: scale factor = fb_stability_a / max_s
            scale = (fb_stability_a / max_s) * 0.999
            V_new *= scale
            print(f"  [FB iter {it+1}] stability scaling applied (scale={scale:.4e}), max_s before={max_s:.4e}")

        V = V_new.astype(np.float32)

    # final Wout on last V
    Z, Xs, Y, preacts = run_esn_forward_collect(
        train_idx, vocab_size, Win, W, V=V,
        alpha=alpha, washout=washout, include_input_in_readout=include_input_in_readout
    )
    Wout = ridge_readout(Z, Y, reg=readout_reg)
    return Wout, V, losses

def evaluate_accuracy(indices, vocab_size, Win, W, Wout, V=None, alpha=0.3, include_input_in_readout=True, washout=0):
    # Teacher forcing evaluation: predict next char from current char and current state
    n_res = W.shape[0]
    I = np.eye(vocab_size, dtype=np.float32)
    x = np.zeros(n_res, dtype=np.float32)
    bias = np.array([1.0], dtype=np.float32)
    correct = 0
    total = 0
    T = len(indices) - 1
    Win_u = Win[:, 1:]
    for t in range(T):
        u_idx = indices[t]
        v_idx = indices[t + 1]
        u = I[u_idx]
        if V is None:
            fb = np.zeros(vocab_size, dtype=np.float32)
        else:
            fb = (V @ x).astype(np.float32)
        in_vec = np.concatenate([bias, (u + fb).astype(np.float32)], dtype=np.float32)
        preact = Win @ in_vec + W @ x
        x = (1.0 - alpha) * x + alpha * np.tanh(preact)

        if t >= washout:
            if include_input_in_readout:
                z = np.concatenate([bias, u, x], dtype=np.float32)
            else:
                z = np.concatenate([bias, x], dtype=np.float32)
            y_pred = Wout @ z
            pred_idx = int(np.argmax(y_pred))
            correct += (pred_idx == v_idx)
            total += 1
    acc = correct / total if total > 0 else 0.0
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="tinyshakespeare.txt", help="Path to text file")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Chronological train split ratio")
    parser.add_argument("--n_res", type=int, default=300, help="Reservoir size")
    parser.add_argument("--density", type=float, default=0.05, help="Reservoir connection density in [0,1]")
    parser.add_argument("--spectral_radius", type=float, default=0.9, help="Reservoir spectral radius")
    parser.add_argument("--alpha", type=float, default=0.3, help="Leaking rate")
    parser.add_argument("--input_scale", type=float, default=1.0, help="Scale for input weights")
    parser.add_argument("--reg", type=float, default=1.0, help="(unused) RLS regularization")
    parser.add_argument("--washout", type=int, default=100, help="Washout steps during training")
    parser.add_argument("--loss_chunk", type=int, default=2000, help="Steps per averaged loss point (unused)")
    parser.add_argument("--include_input", action="store_true", help="Include direct input-to-readout in features")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # feedback options
    parser.add_argument("--feedback", action="store_true", help="Enable V feedback optimization")
    parser.add_argument("--fb_steps", type=int, default=10, help="Number of gradient descent iterations for V")
    parser.add_argument("--fb_lr", type=float, default=1e-3, help="Learning rate for V gradient descent")
    parser.add_argument("--readout_reg", type=float, default=1e-6, help="Ridge regularization for readout when training V")
    parser.add_argument("--fb_stability_a", type=float, default=0.99, help="Max singular value bound for A' (slightly < theoretical a=1 for tanh)")

    args = parser.parse_args()

    # Load text
    with open(args.path, "r", encoding="utf-8") as f:
        text = f.read()

    # Build vocab and indices
    chars, stoi, itos = build_vocab(text)
    data = text_to_indices(text, stoi)
    vocab_size = len(chars)
    N = len(data)

    # Chronological split
    split = int(args.train_ratio * N)
    train_idx = data[:split]
    test_idx  = data[split:]

    print(f"Text length: {N}, Vocab size: {vocab_size}")
    print(f"Train steps: {len(train_idx)-1}, Test steps: {len(test_idx)-1}")

    # Initialize reservoir
    Win, W = init_reservoir(
        n_in=vocab_size,
        n_res=args.n_res,
        density=args.density,
        spectral_radius=args.spectral_radius,
        input_scale=args.input_scale,
        seed=args.seed,
    )

    if args.feedback:
        print("Training with feedback (optimizing V)...")
        Wout, V, losses = train_with_feedback(
            train_idx,
            vocab_size,
            Win, W,
            alpha=args.alpha,
            washout=args.washout,
            fb_steps=args.fb_steps,
            fb_lr=args.fb_lr,
            readout_reg=args.readout_reg,
            include_input_in_readout=args.include_input,
            fb_stability_a=args.fb_stability_a,
            seed=args.seed,
        )
    else:
        # No feedback: collect states and do ridge readout once
        Z, Xs, Y, preacts = run_esn_forward_collect(
            train_idx, vocab_size, Win, W, V=None,
            alpha=args.alpha, washout=args.washout, include_input_in_readout=args.include_input
        )
        Wout = ridge_readout(Z, Y, reg=args.readout_reg if hasattr(args, "readout_reg") else 1e-6)
        V = None
        losses = []

    # Evaluate accuracy on test set (teacher forcing)
    test_acc = evaluate_accuracy(
        test_idx,
        vocab_size,
        Win, W, Wout, V=V,
        alpha=args.alpha,
        include_input_in_readout=args.include_input,
        washout=0,
    )

    print(f"Test accuracy (next-char, teacher forcing): {test_acc:.4f}")

    # Plot training loss (if available)
    if len(losses) > 0:
        plt.figure(figsize=(8,4))
        xs = np.arange(1, len(losses)+1)
        plt.plot(xs, losses, label="Training MSE (per FB iter)")
        plt.xlabel("Feedback iteration")
        plt.ylabel("MSE")
        plt.title("ESN Readout + Feedback Training Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("training_loss_fb.png", dpi=150)
        try:
            plt.show()
        except Exception:
            pass

    ppl, gen_indices = evaluate_free_running(
        test_idx,
        vocab_size,
        Win, W, Wout, V=V,
        alpha=args.alpha,
        include_input_in_readout=args.include_input,
        seed_len=200,
        gen_len=1000,
    )

    gen_text = "".join([itos[i] for i in gen_indices])
    print(f"Free-running perplexity on test set: {ppl:.4f}")
    print("Sample generated text:")
    print(gen_text[:500], "...")


if __name__ == "__main__":
    main()
