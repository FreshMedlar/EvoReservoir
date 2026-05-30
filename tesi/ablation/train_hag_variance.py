import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add the parent directory (tesi) to path so we can import the models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.hag_variance_reservoir import HAGVarianceReservoir

def main():
    # 1. Load dataset (pointing to the shared dataset folder)
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "tinyshakespeare.txt")
    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"Dataset loaded. Total characters: {len(text)}")
    
    # 2. Vocabulary setup
    vocab = sorted(list(set(text)))
    vocab_size = len(vocab)
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    print(f"Vocabulary size: {vocab_size}")
    
    # 3. Parameters
    res_size = 1000
    train_len = 80000
    val_len = 10000
    warmup_steps = 100
    beta = 1e-4  # Ridge regularization coefficient
    input_scaling = 1.0
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize HAG Variance Reservoir
    reservoir_hag_var = HAGVarianceReservoir(
        res_size, vocab_size, input_scaling=input_scaling, density=0.01,
        target_std=0.38, std_spread=0.05, weight_increment=0.01
    )
    
    # Prepare inputs & targets
    total_len = warmup_steps + train_len + val_len
    if len(text) < total_len + 1:
        raise ValueError(f"Text dataset is too short for the configured lengths.")
        
    sub_text = text[:total_len + 1]
    indices = [char2idx[c] for c in sub_text]
    x_indices = indices[:-1]
    y_indices = indices[1:]
    
    # 4. Unsupervised Pre-training Phase (Author Style)
    print("\n--- Running Unsupervised Pre-training Phase (HAG Variance DESP) ---")
    pretrain_len = 20000
    # Prepare one-hot pre-training inputs
    pretrain_onehot = torch.zeros(pretrain_len, vocab_size)
    for i in range(pretrain_len):
        pretrain_onehot[i, x_indices[i]] = 1.0
        
    # Clone the initial weights for verification/comparison
    W_initial = reservoir_hag_var.W.clone()
    
    # Run the offline HAG adaptation
    reservoir_hag_var.pretrain(pretrain_onehot, T_current=500)
    print("Pre-training complete.")
    print(f"   └─ Total Genesis Ops: {reservoir_hag_var.genesis_ops_total}")
    print(f"   └─ Total Pruning Ops: {reservoir_hag_var.pruning_ops_total}")
    
    # Confront initial W and final W (before spectral radius re-scaling)
    with torch.no_grad():
        diff_mask = W_initial != reservoir_hag_var.W
        num_changed = torch.sum(diff_mask).item()
        mean_abs_diff = torch.mean(torch.abs(W_initial - reservoir_hag_var.W)).item()
        max_abs_diff = torch.max(torch.abs(W_initial - reservoir_hag_var.W)).item()
        
        initial_zeros = torch.sum(W_initial == 0.0).item()
        final_zeros = torch.sum(reservoir_hag_var.W == 0.0).item()
        
        print("\n=== Weight Matrix (W) Confrontation ===")
        print(f"Number of weights changed by HAG: {num_changed} / {res_size**2} ({num_changed / (res_size**2) * 100:.4f}%)")
        print(f"Mean absolute change in W: {mean_abs_diff:.6f}")
        print(f"Max absolute change in W: {max_abs_diff:.6f}")
        print(f"Initial zero-weight connections: {initial_zeros}")
        print(f"Post-HAG zero-weight connections: {final_zeros} (change: {final_zeros - initial_zeros})")
        print("=======================================\n")
    
    # Re-scale spectral radius of W to 0.95 to restore signal gain
    print("Re-scaling spectral radius of W to 0.95 post pre-training...")
    radius = torch.max(torch.abs(torch.linalg.eigvals(reservoir_hag_var.W)))
    with torch.no_grad():
        reservoir_hag_var.W.copy_(reservoir_hag_var.W * (0.95 / radius))
    
    # 5. Training Phase (Frozen W)
    print("\n--- Running Training Phase (Frozen Weights) ---")
    reservoir_hag_var.reset_state()
    
    # We collect reservoir states during training phase
    X = []
    Y = []
    
    for i in range(warmup_steps + train_len):
        char_idx = x_indices[i]
        # One-hot input vector
        x_onehot = torch.zeros(vocab_size)
        x_onehot[char_idx] = 1.0
        
        with torch.no_grad():
            reservoir_hag_var.step(x_onehot)
            
        # Collect state and target only after warmup
        if i >= warmup_steps:
            X.append(reservoir_hag_var.state.clone())
            
            # Target is a one-hot representation of the next character
            target_idx = y_indices[i]
            y_onehot = torch.zeros(vocab_size)
            y_onehot[target_idx] = 1.0
            Y.append(y_onehot)
            
    # Convert lists to tensors
    X = torch.stack(X)  # shape: (train_len, res_size)
    Y = torch.stack(Y)  # shape: (train_len, vocab_size)
    
    # Fit Readout weights using Ridge Regression
    print("Solving Ridge Regression for Readout weights...")
    A = torch.matmul(X.T, X) + beta * torch.eye(res_size)
    B = torch.matmul(X.T, Y)
    
    W_out_T = torch.linalg.solve(A, B)
    W_out = W_out_T.T
    
    # Assign the solved weights to reservoir readout
    reservoir_hag_var.readout.weight.data = W_out
    
    # 6. Evaluation Phase
    print("Evaluating on validation sequence...")
    val_loss = 0.0
    val_correct = 0
    loss_fn = nn.CrossEntropyLoss()
    
    for i in range(warmup_steps + train_len, total_len):
        char_idx = x_indices[i]
        x_onehot = torch.zeros(vocab_size)
        x_onehot[char_idx] = 1.0
        
        with torch.no_grad():
            out = reservoir_hag_var.step(x_onehot)
            
        target_idx = y_indices[i]
        target_tensor = torch.tensor([target_idx])
        
        loss = loss_fn(out.unsqueeze(0), target_tensor)
        val_loss += loss.item()
        
        pred_idx = torch.argmax(out).item()
        if pred_idx == target_idx:
            val_correct += 1
            
    val_loss /= val_len
    val_acc = val_correct / val_len
    
    # Print comparison summary
    print("\n================ EXPERIMENT SUMMARY ================")
    print(f"HAG Variance Reservoir (Pre-trained): Validation Loss = {val_loss:.4f}, Accuracy = {val_acc * 100:.2f}%")
    print("====================================================")

if __name__ == "__main__":
    main()
