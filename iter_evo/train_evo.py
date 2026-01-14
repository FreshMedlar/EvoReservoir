
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import math
from typing import Tuple, List, Dict
from esn import SparseESN
import os

# --- Configuration ---
BATCH_SIZE = 32
SEQ_LEN = 64
HIDDEN_SIZE = 2048
DENSITY = 0.1
POPULATION_SIZE = 32 
SIGMA = 0.02
LEARNING_RATE = 0.001
STEPS = 10000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Data Loading ---
def load_data(path):
    with open(path, 'r') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, vocab_size, stoi, itos

def get_batch(data, batch_size, seq_len, device):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)

# --- Evolution Components ---

class Rank1NoiseGenerator:
    def __init__(self, output_size, input_size, device="cpu"):
        self.output_size = output_size
        self.input_size = input_size
        self.device = device
        
    def sample(self, pop_size) -> Tuple[torch.Tensor, torch.Tensor]:
        M = pop_size // 2
        A = torch.randn(M, self.output_size, device=self.device)
        B = torch.randn(M, self.input_size, device=self.device)
        return A, B

class ScalarNoiseGenerator:
    def __init__(self, device="cpu"):
        self.device = device
        
    def sample(self, pop_size) -> torch.Tensor:
        M = pop_size // 2
        return torch.randn(M, device=self.device)

class AdamOptimizer:
    def __init__(self, params_shape: List[int], lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = torch.zeros(params_shape, device=DEVICE)
        self.v = torch.zeros(params_shape, device=DEVICE)
        self.step = 0
        
    def update(self, param, grad):
        self.step += 1
        
        self.m = self.betas[0] * self.m + (1 - self.betas[0]) * grad
        self.v = self.betas[1] * self.v + (1 - self.betas[1]) * (grad ** 2)
        
        m_hat = self.m / (1 - self.betas[0] ** self.step)
        v_hat = self.v / (1 - self.betas[1] ** self.step)
        
        param_new = param + self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        return param_new

def compute_fitness_vectorized(model: SparseESN, 
                               W_rec_base, W_in_base, alpha_base,
                               noise_rec, noise_in, noise_alpha, 
                               sigma, data_x, data_y):
    
    A_rec, B_rec = noise_rec
    A_in, B_in = noise_in
    epsilon_alpha = noise_alpha
    
    M = A_rec.shape[0]
    pop_size = M * 2 # M positive, M negative
    
    # 1. Construct Population Weights
    # Reshape base to [1, H, H]
    W_rec_base_expanded = W_rec_base.unsqueeze(0)
    W_in_base_expanded = W_in_base.unsqueeze(0)
    
    # Rank-1 Perturbations
    # P_rec = A[i] @ B[i].T -> [M, H, H]
    P_rec = torch.bmm(A_rec.unsqueeze(2), B_rec.unsqueeze(1))
    
    # Positive and Negative Populations
    W_rec_pos = W_rec_base_expanded + sigma * P_rec
    W_rec_neg = W_rec_base_expanded - sigma * P_rec
    
    # Concatenate -> [2M, H, H]
    W_rec_all = torch.cat([W_rec_pos, W_rec_neg], dim=0)
    
    # Pruning (Vectorized)
    k_rec = int(model.hidden_size * model.hidden_size * DENSITY)
    # Flatten last two dims: [Pop, H*H]
    flat = W_rec_all.flatten(1)
    # Find thresholds per member
    # [Pop, 1]
    thresholds = torch.topk(torch.abs(flat), k_rec, dim=1).values.min(dim=1, keepdim=True).values
    thresholds = thresholds.unsqueeze(2) # [Pop, 1, 1] for broadcasting
    
    mask = torch.abs(W_rec_all) >= thresholds
    W_rec_all = W_rec_all * mask.float()
    
    # Input Weights
    P_in = torch.bmm(A_in.unsqueeze(2), B_in.unsqueeze(1)) # [M, H, In]
    W_in_pos = W_in_base_expanded + sigma * P_in
    W_in_neg = W_in_base_expanded - sigma * P_in
    W_in_all = torch.cat([W_in_pos, W_in_neg], dim=0) # [Pop, H, In]
    
    # Alphas
    # E_alpha: [M]
    E_alpha_exp = epsilon_alpha.view(M, 1, 1)
    alpha_pos = alpha_base + sigma * E_alpha_exp
    alpha_neg = alpha_base - sigma * E_alpha_exp
    alpha_all = torch.cat([alpha_pos, alpha_neg], dim=0) # [Pop, 1, 1]
    
    # 2. Forward Pass (Vectorized)
    # Input x: [Batch, Seq]
    # x_onehot: [Batch, Seq, In]
    x_oh = F.one_hot(data_x, model.input_size).float()
    
    # Expand input for Population: [Pop, Batch, Seq, In]
    # This might be huge if we replicate?
    # Better: [Batch, Seq, In] shared.
    # At each step t, x_t is [Batch, In].
    # We want to multiply W_in_all [Pop, H, In] with x_t [Batch, In].
    # Result -> [Pop, Batch, H].
    # x_t.T -> [In, Batch].
    # W_in_all @ x_t.T -> [Pop, H, Batch] -> transpose -> [Pop, Batch, H].
    
    h = torch.zeros(pop_size, BATCH_SIZE, model.hidden_size, device=DEVICE)
    loss_all = torch.zeros(pop_size, device=DEVICE)
    
    # Transpose input for easier multiplication
    # x_oh: [Batch, Seq, In] -> [Seq, In, Batch] (Wait, easier to keep batch last for matmul)
    x_seq = x_oh.permute(1, 2, 0) # [Seq, In, Batch]
    data_y_seq = data_y.permute(1, 0) # [Seq, Batch]
    
    for t in range(SEQ_LEN):
        xt = x_seq[t] # [In, Batch]
        
        # W_in_all: [Pop, H, In]
        # We need [Pop, H, Batch]
        # Can use bmm broadcasting? No, bmm needs matching dim 0.
        # We can reshape inputs to repeat.
        # OR: W_in_all @ xt: 
        # But xt is shared. 
        # Einsum is cleaner: 'phi,ib->phb'
        in_act = torch.einsum('phi,ib->phb', W_in_all, xt) # [Pop, H, Batch]
        # Transpose to [Pop, Batch, H]
        in_act = in_act.transpose(1, 2)
        
        # Recurrent part
        # W_rec_all: [Pop, H, H]
        # h: [Pop, Batch, H]
        # We want [Pop, Batch, H]
        # h permute -> [Pop, H, Batch] (wait, h is vector row usually in torch linear)
        # F.linear(h, W) is h @ W.T
        # h: [Pop, Batch, H]
        # W.T: [Pop, H, H] (transposed last 2 dims)
        # bmm(h, W.transpose(1,2)) -> [Pop, Batch, H]
        rec_act = torch.bmm(h, W_rec_all.transpose(1, 2))
        
        pre_act = rec_act + in_act
        
        # Leaky ReLU
        # alpha broadcasting: [Pop, 1, 1] ok
        # "apply leaky relu to forward": LeakyReLU(pre_act)
        # With alpha from parameter? 
        # User defined 'self.leaky_alpha'. 
        # Using that as the 'alpha' in (1-a)h + a*act ?
        # Or as activation param?
        # User said "apply a leaky relu function to the forward". 
        # In last step I implemented "h_new = F.leaky_relu(pre_act)".
        # And I added 'alpha' as a learnable parameter.
        # Wait, F.leaky_relu(x, negative_slope).
        # Ah, 'leaky_alpha' in my code was passed to constructor as 'leaky_alpha'.
        # SparseESN used it as `self.leaky_alpha` but `forward` used hardcoded `0.01` negative slope!
        # "h_new = F.leaky_relu(pre_activation, negative_slope=0.01)"
        # I did not perform `(1-alpha)*h + ...` in the code I wrote in esn.py.
        # But I added `leaky_alpha` as a parameter in the previous turn.
        # Let's assume the user wants `leaky_alpha` to be the mixing rate if they call it 'alpha'?
        # OR the negative slope?
        # Creating a learnable negative slope is weird (unbounded?).
        # Creating a learnable mixing rate for ESN is standard.
        # Let's interpret 'alpha' not as neg_slope (which I fixed at 0.01) but as mixing rate.
        # Current ESN implementation was: `h_new = LeakyReLU(Rec + In)`. State was overwritten.
        # Standard ESN is: `h_new = (1-a)h + a * Activation(...)`. 
        # Given "apply a leaky relu function", this implies activation is LeakyReLU.
        # I will upgrade the model to Leaky Integrator ESN with LeakyReLU activation.
        # h_t = (1 - alpha) * h_{t-1} + alpha * LeakyReLU(...)
        # This makes 'alpha' meaningful.
        
        h_update = F.leaky_relu(pre_act, negative_slope=0.01)
        h = (1 - alpha_all) * h + alpha_all * h_update
        
        # Output
        # Using first output_size neurons
        # h: [Pop, Batch, H] -> slice -> [Pop, Batch, Out]
        logits = h[:, :, :model.output_size]
        
        # Loss
        # Targets: [Batch] -> [Pop, Batch]
        yt = data_y_seq[t]
        # Flatten for cross_entropy: [Pop*Batch, Out] vs [Pop*Batch]
        loss_t = F.cross_entropy(logits.reshape(-1, model.output_size), 
                                 yt.repeat(pop_size), 
                                 reduction='none')
        # Unwrap to [Pop, Batch]
        loss_t = loss_t.view(pop_size, BATCH_SIZE)
        loss_all += loss_t.mean(dim=1)
        
    return -(loss_all / SEQ_LEN) # Fitness

def rank_transformation(fitness):
    N = len(fitness)
    indices = torch.argsort(fitness)
    ranks = torch.zeros_like(fitness)
    ranks[indices] = torch.arange(N, device=fitness.device, dtype=torch.float32)
    return (ranks / (N - 1)) - 0.5

def main():
    wandb.init(project="evo-reservoir", name="vectorized-evo")
    
    # Checkpoint dir
    os.makedirs("checkpoints", exist_ok=True)
    
    data_path = "tinyshakespeare.txt"
    if not os.path.exists(data_path):
        data_path = "tinyshakespeare.txt" 
    
    raw_data, vocab_size, stoi, itos = load_data(data_path)
    n_train = int(len(raw_data) * 0.9)
    train_data = raw_data[:n_train]
    test_data = raw_data[n_train:]
    
    print(f"Vocab: {vocab_size}, Train: {len(train_data)}, Test: {len(test_data)}")
    
    model = SparseESN(vocab_size, HIDDEN_SIZE, vocab_size, density=DENSITY).to(DEVICE)
    
    # Init optimizers
    noise_rec = Rank1NoiseGenerator(HIDDEN_SIZE, HIDDEN_SIZE, device=DEVICE)
    noise_in = Rank1NoiseGenerator(HIDDEN_SIZE, vocab_size, device=DEVICE)
    noise_alpha = ScalarNoiseGenerator(device=DEVICE)
    
    opt_rec = AdamOptimizer(model.W_rec.shape, lr=LEARNING_RATE)
    opt_in = AdamOptimizer(model.W_in.shape, lr=LEARNING_RATE)
    opt_alpha = AdamOptimizer(model.leaky_alpha.shape, lr=LEARNING_RATE)
    
    print("Starting Vectorized Evolution...")
    
    for step in range(STEPS):
        x, y = get_batch(train_data, BATCH_SIZE, SEQ_LEN, DEVICE)
        
        # 1. Sample Noise
        A_rec, B_rec = noise_rec.sample(POPULATION_SIZE)
        A_in, B_in = noise_in.sample(POPULATION_SIZE)
        E_alpha = noise_alpha.sample(POPULATION_SIZE)
        
        # 2. Evaluate (Vectorized)
        fitness = compute_fitness_vectorized(
            model, 
            model.W_rec.data, model.W_in.data, model.leaky_alpha.data,
            (A_rec, B_rec), (A_in, B_in), E_alpha,
            SIGMA, x, y
        )
        
        # 3. Shape Fitness
        shaped_fit = rank_transformation(fitness)
        M = POPULATION_SIZE // 2
        diff_fit = (shaped_fit[:M] - shaped_fit[M:]) / 2.0
        
        # 4. Updates
        # ... (Same as before)
        
        # --- W_rec ---
        grad_rec = torch.zeros_like(model.W_rec.data)
        for i in range(M):
            grad_rec += diff_fit[i] * torch.outer(A_rec[i], B_rec[i])
        grad_rec /= (M * SIGMA)
        
        W_rec_new = opt_rec.update(model.W_rec.data, grad_rec)
        
        # Prune
        k_rec = int(HIDDEN_SIZE * HIDDEN_SIZE * DENSITY)
        flat = W_rec_new.view(-1)
        thr = torch.topk(torch.abs(flat), k_rec).values.min()
        mask = torch.abs(W_rec_new) >= thr
        W_rec_new = W_rec_new * mask.float()
        
        model.W_rec.data = W_rec_new
        if step % 50 == 0:
             rho = model.enforce_spectral_radius(0.95)
        else:
             rho = 0.0 # Skip for speed
        
        # --- W_in ---
        grad_in = torch.zeros_like(model.W_in.data)
        for i in range(M):
            grad_in += diff_fit[i] * torch.outer(A_in[i], B_in[i])
        grad_in /= (M * SIGMA)
        model.W_in.data = opt_in.update(model.W_in.data, grad_in)
        
        # --- Alpha ---
        grad_alpha = torch.zeros_like(model.leaky_alpha.data)
        for i in range(M):
            grad_alpha += diff_fit[i] * E_alpha[i]
        grad_alpha /= (M * SIGMA)
        
        alpha_new = opt_alpha.update(model.leaky_alpha.data, grad_alpha)
        model.leaky_alpha.data = torch.clamp(alpha_new, 0.01, 0.99)
        
        # Logging
        loss = -fitness.max().item()
        
        if step % 10 == 0:
            print(f"Step {step}: Loss {loss:.4f} | Alpha {model.leaky_alpha.item():.3f}")
            wandb.log({
                "train/loss": loss, 
                "train/alpha": model.leaky_alpha.item()
            })
            
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                xt, yt = get_batch(test_data, BATCH_SIZE, SEQ_LEN, DEVICE)
                h = torch.zeros(BATCH_SIZE, HIDDEN_SIZE, device=DEVICE)
                xt_oh = F.one_hot(xt, vocab_size).float()
                loss_test = 0
                
                # Non-vectorized forward for testing using shared model weights
                # We need to manually match the logic if we changed it to leaky integrator
                # BUT model.forward in esn.py is NOT leaky integrator yet!
                # I need to update esn.py forward to match the new logic (Leaky Integrator)
                # Or update compute_fitness to match esn.py.
                # User asked "how to make it better", I suggested learning alpha.
                # Implicitly, learning alpha implies leaky integrator.
                # I should update esn.py to be consistent.
                
                for t in range(SEQ_LEN):
                    # Using model() calls esn.py forward
                    h = model(xt_oh[:, t, :], h)
                    logits = model.get_output(h)
                    loss_test += F.cross_entropy(logits, yt[:, t]).item()
                loss_test /= SEQ_LEN
                print(f"Test Loss: {loss_test:.4f}")
                wandb.log({"test/loss": loss_test})

        if step > 0 and step % 1000 == 0:
            ckpt_path = f"checkpoints/model_step_{step}.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_rec_state_dict': opt_rec.m, # Saving opt state as well
                'optimizer_in_state_dict': opt_in.m,
                'optimizer_alpha_state_dict': opt_alpha.m,
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    wandb.finish()

if __name__ == "__main__":
    main()
