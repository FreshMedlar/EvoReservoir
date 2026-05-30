import os
import torch
import torch.nn as nn
import numpy as np
from model.reservoir import Reservoir

def main():
    # 1. Load dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset", "tinyshakespeare.txt")
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
    res_size = 1500
    train_len = 120000
    val_len = 10000
    warmup_steps = 200
    beta = 1e-4  # Ridge regularization coefficient
    input_scaling = 1.0
    
    # Initialize reservoir
    reservoir = Reservoir(res_size, vocab_size, input_scaling=input_scaling, density=0.01)
    
    # 4. Prepare inputs & targets
    # We will use one-hot encoding for inputs
    total_len = warmup_steps + train_len + val_len
    if len(text) < total_len + 1:
        raise ValueError(f"Text dataset is too short for the configured lengths.")
        
    sub_text = text[:total_len + 1]
    indices = [char2idx[c] for c in sub_text]
    
    # Inputs: x_t, Targets: y_t = x_{t+1}
    x_indices = indices[:-1]
    y_indices = indices[1:]
    
    # 5. Run the reservoir on the warmup + training sequence
    print("Running reservoir on training sequence...")
    reservoir.reset_state()
    
    # We collect reservoir states during training phase
    X = []
    Y = []
    
    # Warmup and training loop
    for i in range(warmup_steps + train_len):
        char_idx = x_indices[i]
        # One-hot input vector
        x_onehot = torch.zeros(vocab_size)
        x_onehot[char_idx] = 1.0
        
        # Step reservoir (which also updates fast_trail, slow_trail, scaling)
        with torch.no_grad():
            reservoir.step(x_onehot)
            
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
    
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"X mean: {X.mean().item():.4f}, X std: {X.std().item():.4f}, X min: {X.min().item():.4f}, X max: {X.max().item():.4f}")
    print(f"Scaling mean: {reservoir.scaling.mean().item():.4f}, min: {reservoir.scaling.min().item():.4f}, max: {reservoir.scaling.max().item():.4f}")
    print(f"Slow trail mean: {reservoir.slow_trail.mean().item():.4f}, min: {reservoir.slow_trail.min().item():.4f}, max: {reservoir.slow_trail.max().item():.4f}")
    
    # 6. Fit Readout weights using Ridge Regression on the last 100k steps
    print("Solving Ridge Regression for Readout weights (on last 100k steps)...")
    X_train = X[-100000:]
    Y_train = Y[-100000:]
    
    # W_out = Y_train^T X_train (X_train^T X_train + beta * I)^-1
    # We solve: (X_train^T X_train + beta * I) W_out^T = X_train^T Y_train
    A = torch.matmul(X_train.T, X_train) + beta * torch.eye(res_size)
    B = torch.matmul(X_train.T, Y_train)
    
    W_out_T = torch.linalg.solve(A, B)
    W_out = W_out_T.T
    
    # Assign the solved weights to reservoir readout
    reservoir.readout.weight.data = W_out
    print("Readout weights updated.")
    
    # 7. Evaluate on validation set
    print("Evaluating on validation sequence...")
    val_loss = 0.0
    val_correct = 0
    loss_fn = nn.CrossEntropyLoss()
    
    # Evaluate sequence
    for i in range(warmup_steps + train_len, total_len):
        char_idx = x_indices[i]
        x_onehot = torch.zeros(vocab_size)
        x_onehot[char_idx] = 1.0
        
        with torch.no_grad():
            out = reservoir.step(x_onehot)
            
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
    
    # 8. Text Generation Sample
    print("\n--- Generating Text Sample (Greedy) ---")
    prompt = "ROMEO:\n"
    print(f"Prompt: {prompt}", end="")
    
    # Warmup reservoir with the prompt
    reservoir.reset_state()
    for char in prompt[:-1]:
        x_onehot = torch.zeros(vocab_size)
        x_onehot[char2idx[char]] = 1.0
        with torch.no_grad():
            reservoir.step(x_onehot)
            
    # Generate text (Greedy)
    current_char = prompt[-1]
    generated_text = ""
    for _ in range(150):
        x_onehot = torch.zeros(vocab_size)
        x_onehot[char2idx[current_char]] = 1.0
        with torch.no_grad():
            out = reservoir.step(x_onehot)
        pred_idx = torch.argmax(out).item()
        next_char = idx2char[pred_idx]
        generated_text += next_char
        current_char = next_char
    print(generated_text)
    print("---------------------------------------")

    print("\n--- Generating Text Sample (Temp=0.1) ---")
    print(f"Prompt: {prompt}", end="")
    
    # Warmup reservoir with the prompt again
    reservoir.reset_state()
    for char in prompt[:-1]:
        x_onehot = torch.zeros(vocab_size)
        x_onehot[char2idx[char]] = 1.0
        with torch.no_grad():
            reservoir.step(x_onehot)
            
    # Generate text (Sampling)
    current_char = prompt[-1]
    generated_text = ""
    for _ in range(150):
        x_onehot = torch.zeros(vocab_size)
        x_onehot[char2idx[current_char]] = 1.0
        with torch.no_grad():
            out = reservoir.step(x_onehot)
        
        probs = torch.softmax(out / 0.1, dim=0).numpy()
        pred_idx = np.random.choice(vocab_size, p=probs)
        next_char = idx2char[pred_idx]
        generated_text += next_char
        current_char = next_char
    print(generated_text)
    print("------------------------------------------")

if __name__ == "__main__":
    main()
