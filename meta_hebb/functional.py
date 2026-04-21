import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# 1. FUNCTIONAL HEBBIAN MODEL
# ==============================================================================

def hebbian_update(x, y, current_weights, coeffs):
    """
    Performs the Hebbian update step y = x @ W, then W_new = W + delta.
    """
    # Dimensions
    B, In, Out = current_weights.shape
    
    # Reshape for broadcasting
    x_col = x.view(B, In, 1)
    y_row = y.view(B, 1, Out)
    
    # Coefficients extraction
    c_hebb  = coeffs[..., 0]  # A
    c_in    = coeffs[..., 1]  # B
    c_out   = coeffs[..., 2]  # C
    c_bias  = coeffs[..., 3]  # D
    c_eta   = coeffs[..., 4]  # Learning Rate
    
    # Hebbian Terms
    term1 = c_hebb * torch.bmm(x_col, y_row) 
    term2 = c_in * x_col 
    term3 = c_out * y_row 
    term4 = c_bias
    
    # Update
    delta = c_eta * (term1 + term2 + term3 + term4)
    return current_weights + delta

def layer_forward(x, weights):
    return torch.bmm(x.unsqueeze(1), weights).squeeze(1)

def model_forward_step(x, fast_weights_l1, fast_weights_l2, coeffs_l1, coeffs_l2):
    # --- Layer 1 ---
    h = layer_forward(x, fast_weights_l1)
    h_act = torch.tanh(h)
    new_fast_weights_l1 = hebbian_update(x, h, fast_weights_l1, coeffs_l1)
    
    # --- Layer 2 ---
    y = layer_forward(h_act, fast_weights_l2)
    new_fast_weights_l2 = hebbian_update(h_act, y, fast_weights_l2, coeffs_l2)
    
    return y, new_fast_weights_l1, new_fast_weights_l2

def evaluate_sequence(inputs, targets, w1_init, w2_init, coeffs_l1, coeffs_l2):
    """
    Runs the model over a sequence.
    """
    B, Seq, _ = inputs.shape
    
    # Use the passed initialization
    w1 = w1_init
    w2 = w2_init
    
    losses = []
    
    for t in range(Seq):
        x_t = inputs[:, t, :] 
        y_gt = targets[:, t, :] 
        
        y_pred, w1, w2 = model_forward_step(x_t, w1, w2, coeffs_l1, coeffs_l2)
        
        step_loss = F.mse_loss(y_pred, y_gt)
        losses.append(step_loss)
        
    total_loss = torch.stack(losses).mean()
    return -total_loss 

# ==============================================================================
# 2. DATA GENERATOR
# ==============================================================================

def get_batch(batch_size, seq_len, x_dim, device):
    # Weights for the task (Batch, In, Out=10)
    w_target = torch.randn(batch_size, x_dim, 10, device=device)
    # Inputs
    x = torch.randn(batch_size, seq_len, x_dim, device=device)
    # Targets
    y = torch.bmm(x, w_target)
    return x, y

# ==============================================================================
# 3. EGGROLL OPTIMIZER
# ==============================================================================

class EggrollOptimizer:
    def __init__(self, params_dict, config):
        self.params = params_dict
        self.pop_size = config['pop_size']
        self.sigma = config['sigma']
        self.lr = config['lr']
        self.rank = config.get('rank', 1) 
        self.device = list(params_dict.values())[0].device
        
    def step(self, inputs, targets):
        """
        Performs one EGGROLL update step.
        """
        half_pop = self.pop_size // 2
        
        # 1. Generate Perturbations 
        perturbations = {}
        for name, p in self.params.items():
            orig_shape = p.shape
            flat_dim_in = orig_shape[0]
            flat_dim_out = p.numel() // flat_dim_in
            
            A = torch.randn(half_pop, flat_dim_in, self.rank, device=self.device)
            B = torch.randn(half_pop, flat_dim_out, self.rank, device=self.device)
            perturbations[name] = (A, B, orig_shape)

        # 2. Construct Population
        def get_perturbed_param(name, idx, sign):
            A, B, shape = perturbations[name]
            scale = (1.0 / math.sqrt(self.rank)) * self.sigma * sign
            noise = torch.mm(A[idx], B[idx].T)
            return self.params[name] + scale * noise.view(shape)

        pop_params_l1 = []
        pop_params_l2 = []
        
        # Positive and Negative perturbations
        for i in range(half_pop):
            pop_params_l1.append(get_perturbed_param('coeffs_l1', i, 1.0))
            pop_params_l2.append(get_perturbed_param('coeffs_l2', i, 1.0))
        for i in range(half_pop):
            pop_params_l1.append(get_perturbed_param('coeffs_l1', i, -1.0))
            pop_params_l2.append(get_perturbed_param('coeffs_l2', i, -1.0))
            
        stack_l1 = torch.stack(pop_params_l1)
        stack_l2 = torch.stack(pop_params_l2)
        
        # Init weights for evaluation
        B_size = inputs.shape[0]
        w1_init = torch.empty(B_size, 10, 20, device=self.device).uniform_(-0.1, 0.1)
        w2_init = torch.empty(B_size, 20, 10, device=self.device).uniform_(-0.1, 0.1)

        # 3. Vectorized Evaluation
        # We wrap evaluation in no_grad to save memory/compute as we don't need autograd
        with torch.no_grad():
            fitness_fn = torch.vmap(evaluate_sequence, in_dims=(None, None, None, None, 0, 0))
            fitnesses = fitness_fn(inputs, targets, w1_init, w2_init, stack_l1, stack_l2)
        
        # 4. Fitness Shaping
        ranks = torch.argsort(torch.argsort(fitnesses))
        shaped_fitness = (ranks.float() / (self.pop_size - 1)) - 0.5
        
        # 5. Update Mean Parameters
        # [FIX]: Wrap update in no_grad to allow in-place modification of leaf tensors
        with torch.no_grad():
            for name in self.params:
                A, B, shape = perturbations[name] 
                
                f_pos = shaped_fitness[:half_pop]
                f_neg = shaped_fitness[half_pop:]
                combined_fitness = (f_pos - f_neg).view(half_pop, 1, 1) 
                
                A_weighted = A * combined_fitness
                grad_flat = torch.einsum('mir,mor->io', A_weighted, B)
                
                update_step = (self.lr / (self.sigma * self.pop_size)) * grad_flat
                self.params[name] += update_step.view(shape)
            
        return fitnesses.mean().item()

# ==============================================================================
# 4. TRAINING LOOP
# ==============================================================================

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {DEVICE}")
    
    # Task Config
    X_DIM = 10
    SEQ_LEN = 50
    BATCH_SIZE = 32
    
    # Model Init
    coeffs_l1 = torch.nn.Parameter(torch.zeros(10, 20, 5, device=DEVICE))
    coeffs_l2 = torch.nn.Parameter(torch.zeros(20, 10, 5, device=DEVICE))
    # nn.init.uniform_(coeffs_l1, -0.01, 0.01)
    # nn.init.uniform_(coeffs_l2, -0.01, 0.01)
    
    params = {'coeffs_l1': coeffs_l1, 'coeffs_l2': coeffs_l2}
    
    egg_config = {
        'pop_size': 256,    
        'sigma': 0.2,       
        'lr': 0.001,         
        'rank': 1           
    }
    
    optimizer = EggrollOptimizer(params, egg_config)
    
    print("Starting EGGROLL Meta-Training...")
    
    for step in range(1001):
        x, y = get_batch(BATCH_SIZE, SEQ_LEN, X_DIM, DEVICE)
        mean_fitness = optimizer.step(x, y)
        
        if step % 50 == 0:
            mse = -mean_fitness
            print(f"Step {step:04d} | MSE: {mse:.6f} | Fitness: {mean_fitness:.4f}")

    print("\nValidation Run (Zero Noise):")
    x_val, y_val = get_batch(BATCH_SIZE, SEQ_LEN, X_DIM, DEVICE)
    
    w1_val = torch.empty(BATCH_SIZE, 10, 20, device=DEVICE).uniform_(-0.1, 0.1)
    w2_val = torch.empty(BATCH_SIZE, 20, 10, device=DEVICE).uniform_(-0.1, 0.1)
    
    final_score = evaluate_sequence(x_val, y_val, w1_val, w2_val, params['coeffs_l1'], params['coeffs_l2'])
    print(f"Final Validation MSE: {-final_score.item():.6f}")
    
    print("\nLearned Hebbian Learning Rates (Avg per layer):")
    print(f"Layer 1 Eta: {params['coeffs_l1'][..., 4].mean().item():.4f}")
    print(f"Layer 2 Eta: {params['coeffs_l2'][..., 4].mean().item():.4f}")

if __name__ == "__main__":
    main()
