import os
import sys
import copy
import torch
import torch.nn as nn
import numpy as np

# Insert the parent directory (tesi) at index 0 to override local ablation model imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.reservoir import Reservoir

def run_experiment(reservoir, step_fn_name, char2idx, idx2char, x_indices, y_indices, 
                   vocab_size, res_size, warmup_steps, train_len, val_len, beta):
    print(f"\n--- Running Experiment with {step_fn_name} ---")
    reservoir.reset_state()
    
    # We collect reservoir states during training phase
    X = []
    Y = []
    
    # Warmup and training loop
    step_fn = getattr(reservoir, step_fn_name)
    for i in range(warmup_steps + train_len):
        char_idx = x_indices[i]
        # One-hot input vector
        x_onehot = torch.zeros(vocab_size)
        x_onehot[char_idx] = 1.0
        
        with torch.no_grad():
            step_fn(x_onehot)
            
        # Collect state and target only after warmup
        if i >= warmup_steps:
            X.append(reservoir.state.clone())
            
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
    reservoir.readout.weight.data = W_out
    
    # Evaluate on validation set
    print("Evaluating on validation sequence...")
    val_loss = 0.0
    val_correct = 0
    loss_fn = nn.CrossEntropyLoss()
    total_len = warmup_steps + train_len + val_len
    
    for i in range(warmup_steps + train_len, total_len):
        char_idx = x_indices[i]
        x_onehot = torch.zeros(vocab_size)
        x_onehot[char_idx] = 1.0
        
        with torch.no_grad():
            out = step_fn(x_onehot)
            
        target_idx = y_indices[i]
        target_tensor = torch.tensor([target_idx])
        
        loss = loss_fn(out.unsqueeze(0), target_tensor)
        val_loss += loss.item()
        
        pred_idx = torch.argmax(out).item()
        if pred_idx == target_idx:
            val_correct += 1
            
    val_loss /= val_len
    val_acc = val_correct / val_len
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    
    # Text Generation Sample
    print("Generating Text Sample (Temp=0.1) ---")
    prompt = "ROMEO:\n"
    print(f"Prompt: {prompt}", end="")
    
    # Reset and warmup reservoir with the prompt
    reservoir.reset_state()
    for char in prompt[:-1]:
        x_onehot = torch.zeros(vocab_size)
        x_onehot[char2idx[char]] = 1.0
        with torch.no_grad():
            step_fn(x_onehot)
            
    # Generate text (Sampling)
    current_char = prompt[-1]
    generated_text = ""
    for _ in range(150):
        x_onehot = torch.zeros(vocab_size)
        x_onehot[char2idx[current_char]] = 1.0
        with torch.no_grad():
            out = step_fn(x_onehot)
        
        probs = torch.softmax(out / 0.1, dim=0).numpy()
        pred_idx = np.random.choice(vocab_size, p=probs)
        next_char = idx2char[pred_idx]
        generated_text += next_char
        current_char = next_char
    print(generated_text)
    print("------------------------------------------")
    return val_loss, val_acc

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
    
    # Initialize reservoir (Evolutionary Style)
    # Using spontaneous_rate=0.001 to enable evolutionary dynamics for comparison
    reservoir_evo = Reservoir(res_size, vocab_size, input_scaling=input_scaling, density=0.01, spontaneous_rate=0.001)
    
    # Create an identical copy of the reservoir for the Simplified Style
    reservoir_simple = copy.deepcopy(reservoir_evo)
    
    # Prepare inputs & targets
    total_len = warmup_steps + train_len + val_len
    if len(text) < total_len + 1:
        raise ValueError(f"Text dataset is too short for the configured lengths.")
        
    sub_text = text[:total_len + 1]
    indices = [char2idx[c] for c in sub_text]
    x_indices = indices[:-1]
    y_indices = indices[1:]
    
    # Run evolutionary style
    evo_loss, evo_acc = run_experiment(
        reservoir_evo, "step", char2idx, idx2char, x_indices, y_indices,
        vocab_size, res_size, warmup_steps, train_len, val_len, beta
    )
    
    # Run simplified style
    simple_loss, simple_acc = run_experiment(
        reservoir_simple, "step_no_evolution", char2idx, idx2char, x_indices, y_indices,
        vocab_size, res_size, warmup_steps, train_len, val_len, beta
    )
    
    # Print comparison summary
    print("\n================ COMPARISON SUMMARY ================")
    print(f"Evolutionary Reservoir: Validation Loss = {evo_loss:.4f}, Accuracy = {evo_acc * 100:.2f}%")
    print(f"  └─ Total Genesis Ops: {reservoir_evo.genesis_ops_total}, Total Pruning Ops: {reservoir_evo.pruning_ops_total}")
    print(f"Simplified Reservoir:   Validation Loss = {simple_loss:.4f}, Accuracy = {simple_acc * 100:.2f}%")
    print(f"Difference (Evo - Simple Acc): {(evo_acc - simple_acc) * 100:+.2f}%")
    print("====================================================")

if __name__ == "__main__":
    # Let's also support running with spontaneous rate from command line if needed
    main()
