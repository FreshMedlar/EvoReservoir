import torch
import torch.nn as nn
import numpy as np

class HAGVarianceReservoir():
    def __init__(self, res_size, output_dim, 
            input_scaling=1.0, 
            e_ratio=0.8, 
            density=0.01, 
            target_std=0.25,
            std_spread=0.05,
            weight_increment=0.1):
        self.res_size = res_size
        self.e_size = int(res_size * e_ratio)
        
        # HAG Variance (DESP) Parameters
        self.target_std = target_std
        self.std_spread = std_spread
        self.weight_increment = weight_increment
        
        ### Weights initialization 
        W = torch.randn(res_size, res_size)
        
        # Apply sparsity (most values set to 0)
        mask = torch.rand(res_size, res_size) < density
        W = W * mask
        
        # 2. Apply E-I constraints
        # Excitatory columns: set all values to be positive
        W[:, :self.e_size] = torch.abs(W[:, :self.e_size])
        
        # Inhibitory columns: set all values to be negative
        W[:, self.e_size:] = -torch.abs(W[:, self.e_size:])
        
        # 3. Spectral radius scaling (maintains stability)
        radius = torch.max(torch.abs(torch.linalg.eigvals(W)))
        self.W = nn.Parameter(W * (0.95 / radius), requires_grad=False)
        self.W_in = nn.Parameter(torch.randn(res_size, output_dim) * input_scaling, requires_grad=False)

        self.state = torch.zeros(res_size, device=self.W.device)
        self.readout = nn.Linear(res_size, output_dim, bias=False)
        
        # track operations
        self.genesis_ops_total = 0
        self.pruning_ops_total = 0

    def pretrain(self, pretrain_inputs, T_current=500):
        # Reset state at beginning of pretrain
        self.state = torch.zeros(self.res_size, device=self.W.device)
        self.genesis_ops_total = 0
        self.pruning_ops_total = 0
        
        num_steps = pretrain_inputs.shape[0]
        
        # We process inputs in blocks of length T_current
        for start_idx in range(0, num_steps - T_current + 1, T_current):
            block_inputs = pretrain_inputs[start_idx : start_idx + T_current]
            
            # Record state history for this block
            states_history = []
            for t in range(T_current):
                x = block_inputs[t]
                x_in = torch.matmul(self.W_in, x)
                x_1 = torch.matmul(self.state, self.W.T)
                self.state = torch.tanh(x_1 + x_in)
                states_history.append(self.state.clone())
                
            # Convert to tensor: shape (T_current, res_size)
            states_history = torch.stack(states_history)
            
            # Compute standard deviation and Pearson correlation matrix for this block
            std = torch.std(states_history, dim=0) # shape: (res_size,)
            
            # compute Pearson correlation matrix R
            states_mean = torch.mean(states_history, dim=0, keepdim=True)
            states_centered = states_history - states_mean
            cov = torch.matmul(states_centered.T, states_centered) / (T_current - 1)
            
            std_eps = std + 1e-8
            R = cov / torch.outer(std_eps, std_eps)
            R = torch.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Compute delta_z
            delta_z = (std - self.target_std) / self.std_spread
            
            need_pruning = torch.where(delta_z >= 1.0)[0]
            need_genesis = torch.where(delta_z <= -1.0)[0]
            
            # Genesis (low variance)
            if len(need_genesis) > 0:
                for i in need_genesis:
                    candidates = torch.ones(self.res_size, dtype=torch.bool, device=self.W.device)
                    candidates[i] = False
                    zero_mask = self.W[i] == 0.0
                    candidates = candidates & zero_mask
                    
                    candidate_indices = torch.where(candidates)[0]
                    if len(candidate_indices) > 0:
                        corrs = R[i, candidate_indices]
                        best_idx = candidate_indices[torch.argmax(corrs)]
                        
                        delta = self.weight_increment if best_idx < self.e_size else -self.weight_increment
                        with torch.no_grad():
                            self.W[i, best_idx] += delta
                        self.genesis_ops_total += 1
                        
            # Pruning (high variance)
            if len(need_pruning) > 0:
                for i in need_pruning:
                    candidates = self.W[i] != 0.0
                    
                    candidate_indices = torch.where(candidates)[0]
                    if len(candidate_indices) > 0:
                        corrs = R[i, candidate_indices]
                        worst_idx = candidate_indices[torch.argmin(corrs)]
                        
                        delta = -self.weight_increment if worst_idx < self.e_size else self.weight_increment
                        with torch.no_grad():
                            self.W[i, worst_idx] += delta
                        self.pruning_ops_total += 1
                        
                with torch.no_grad():
                    self.W[:, :self.e_size] = torch.clamp(self.W[:, :self.e_size], min=0.0)
                    self.W[:, self.e_size:] = torch.clamp(self.W[:, self.e_size:], max=0.0)

    def step(self, x):
        # Standard frozen ESN update
        x_in = torch.matmul(self.W_in, x)
        x_1 = torch.matmul(self.state, self.W.T)
        self.state = torch.tanh(x_1 + x_in)
        return self.readout(self.state)

    def reset_state(self):
        self.state = torch.zeros(self.state.shape, device=self.W.device)
