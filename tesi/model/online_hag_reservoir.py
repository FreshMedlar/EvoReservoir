import torch
import torch.nn as nn
import numpy as np

class HAGReservoir():
    def __init__(self, res_size, output_dim, 
            input_scaling=1.0, 
            e_ratio=0.8, 
            density=0.01, 
            target_rate=0.5,
            rate_spread=0.1,
            weight_increment=0.1,
            rewiring_interval=100,
            alpha=0.01):
        self.res_size = res_size
        self.e_size = int(res_size * e_ratio)
        
        # HAG Parameters
        self.target_rate = target_rate
        self.rate_spread = rate_spread
        self.weight_increment = weight_increment
        self.rewiring_interval = rewiring_interval
        self.alpha = alpha
        
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
        
        # running statistics buffers (for Pearson correlation & HADSP dynamics)
        self.running_mean = torch.ones(res_size, device=self.W.device) * target_rate
        self.running_signed_mean = torch.zeros(res_size, device=self.W.device)
        self.running_var = torch.ones(res_size, device=self.W.device) * (rate_spread ** 2)
        self.running_cov = torch.eye(res_size, device=self.W.device) * (rate_spread ** 2)
        
        # track operations
        self.genesis_ops_total = 0
        self.pruning_ops_total = 0
        self.latest_genesis_ops = 0
        self.latest_pruning_ops = 0
        self.step_count = 0

    def step(self, x):
        # 1. Update state using standard ESN update (no scaling parameter to keep HAG clean)
        x_in = torch.matmul(self.W_in, x)
        x_1 = torch.matmul(self.state, self.W.T)
        self.state = torch.tanh(x_1 + x_in)

        # 2. Update running statistics
        # We track running_mean on absolute state values to reflect firing rate/amplitude correctly
        # under a symmetric activation function (tanh) centered at 0.
        self.running_mean = (1.0 - self.alpha) * self.running_mean + self.alpha * torch.abs(self.state)
        self.running_signed_mean = (1.0 - self.alpha) * self.running_signed_mean + self.alpha * self.state
        diff = self.state - self.running_signed_mean
        self.running_var = (1.0 - self.alpha) * self.running_var + self.alpha * (diff ** 2)
        self.running_cov = (1.0 - self.alpha) * self.running_cov + self.alpha * torch.outer(diff, diff)
        
        self.step_count += 1
        self.latest_genesis_ops = 0
        self.latest_pruning_ops = 0

        # 3. Periodic rewiring using online HAG
        if self.step_count % self.rewiring_interval == 0:
            # HADSP delta_z: (mean - target_rate) / rate_spread
            delta_z = (self.running_mean - self.target_rate) / self.rate_spread
            
            # Identify target neurons needing adaptation
            need_pruning = torch.where(delta_z >= 1.0)[0]
            need_genesis = torch.where(delta_z <= -1.0)[0]
            
            # Compute Pearson correlation matrix R
            std = torch.sqrt(self.running_var) + 1e-8
            R = self.running_cov / torch.outer(std, std)
            R = torch.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Genesis for low-activity neurons
            if len(need_genesis) > 0:
                for i in need_genesis:
                    # Candidates are all other neurons where connection is zero
                    candidates = torch.ones(self.res_size, dtype=torch.bool, device=self.W.device)
                    candidates[i] = False
                    zero_mask = self.W[i] == 0.0
                    candidates = candidates & zero_mask
                    
                    candidate_indices = torch.where(candidates)[0]
                    if len(candidate_indices) > 0:
                        # Select source j with max Pearson correlation
                        corrs = R[i, candidate_indices]
                        best_idx = candidate_indices[torch.argmax(corrs)]
                        
                        # Add connection respecting E-I constraints
                        delta = self.weight_increment if best_idx < self.e_size else -self.weight_increment
                        with torch.no_grad():
                            self.W[i, best_idx] += delta
                        self.latest_genesis_ops += 1
                        self.genesis_ops_total += 1

            # Pruning for high-activity neurons
            if len(need_pruning) > 0:
                for i in need_pruning:
                    # Candidates are existing active connections
                    candidates = self.W[i] != 0.0
                    
                    candidate_indices = torch.where(candidates)[0]
                    if len(candidate_indices) > 0:
                        # Select source j with min Pearson correlation
                        corrs = R[i, candidate_indices]
                        worst_idx = candidate_indices[torch.argmin(corrs)]
                        
                        # Weaken connection respecting E-I constraints
                        delta = -self.weight_increment if worst_idx < self.e_size else self.weight_increment
                        with torch.no_grad():
                            self.W[i, worst_idx] += delta
                        self.latest_pruning_ops += 1
                        self.pruning_ops_total += 1
                
                # Enforce E-I sign constraints on modified weights
                with torch.no_grad():
                    self.W[:, :self.e_size] = torch.clamp(self.W[:, :self.e_size], min=0.0)
                    self.W[:, self.e_size:] = torch.clamp(self.W[:, self.e_size:], max=0.0)

        return self.readout(self.state)

    def reset_state(self):
        self.state = torch.zeros(self.state.shape, device=self.W.device)
        self.genesis_ops_total = 0
        self.pruning_ops_total = 0
        self.latest_genesis_ops = 0
        self.latest_pruning_ops = 0
        self.step_count = 0
        
        # Reset running statistics
        self.running_mean.fill_(self.target_rate)
        self.running_signed_mean.zero_()
        self.running_var.fill_(self.rate_spread ** 2)
        self.running_cov.copy_(torch.eye(self.res_size, device=self.W.device) * (self.rate_spread ** 2))
