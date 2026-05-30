import numpy as np
import torch
import torch.nn as nn

class Reservoir():
    def __init__(self, res_size, output_dim, 
            input_scaling=1.0, 
            e_ratio=0.8, 
            density=0.01, 
            spontaneous_rate=0):
        self.res_size = res_size
        self.e_size = int(res_size * e_ratio)
        self.spontaneous_rate = spontaneous_rate
        
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
        # per neuron trailing activation
        self.fast_trail = torch.ones(res_size) * 0.5
        self.slow_trail = torch.ones(res_size) * 0.5

        self.state = torch.zeros(res_size)
        self.scaling = torch.ones(res_size)

        self.readout = nn.Linear(res_size, output_dim, bias=False)
        
        # track operations
        self.genesis_ops_total = 0
        self.pruning_ops_total = 0
        self.latest_genesis_ops = 0
        self.latest_pruning_ops = 0
        
    # perform one step in the reservoir
    def step(self, x):
        x_in = torch.matmul(self.W_in, x)
        x_1 = torch.matmul(self.state, self.W.T)
        # Standard non-leaky ESN update (states bounded to [-1, 1] by tanh)
        self.state = torch.tanh(self.scaling * (x_1 + x_in))

        # trails update
        self.slow_trail = 0.99 * self.slow_trail + 0.01 * torch.abs(self.state)
        self.fast_trail = 0.8 * self.fast_trail + 0.2 * torch.abs(self.state)

        ### HOMEOSTASIS STEP
        abs_slow = self.slow_trail
        low_mask = abs_slow < 0.15
        high_mask = abs_slow > 0.85
        normal_mask = ~(low_mask | high_mask)  # inclusive
        
        self.scaling[low_mask] += 0.01
        self.scaling[high_mask] -= 0.01
        self.scaling[normal_mask] = 1.0
        self.scaling = torch.clamp(self.scaling, min=0.1, max=5.0)

        ### genesis / pruning
        weak_mask = abs_slow < 0.10
        strong_mask = abs_slow > 0.90
        
        # Incorporate spontaneous exploratory genesis
        if self.spontaneous_rate > 0:
            exploratory_mask = torch.rand(self.res_size, device=self.slow_trail.device) < self.spontaneous_rate
            weak_mask = weak_mask | exploratory_mask
            
        weak_indices = torch.where(weak_mask)[0]
        strong_indices = torch.where(strong_mask)[0]
        
        self.latest_genesis_ops = len(weak_indices)
        self.latest_pruning_ops = len(strong_indices)
        self.genesis_ops_total += len(weak_indices)
        self.pruning_ops_total += len(strong_indices)

        if len(weak_indices) > 0:
            # 1. Choose a random source neuron for each weak target neuron
            src_indices = torch.randint(0, self.res_size, (len(weak_indices),), device=self.slow_trail.device)
            
            # 2. Strengthen existing connection or create a new one by adding 0.1
            # (Add +0.1 for excitatory sources, -0.1 for inhibitory sources to respect E-I constraints)
            is_excitatory = src_indices < self.e_size
            delta = torch.where(is_excitatory, 0.1, -0.1)
            
            # 3. Apply the delta to the selected connections in place
            with torch.no_grad():
                self.W[weak_indices, src_indices] += delta

        if len(strong_indices) > 0:
            # 1. Choose a random source neuron for each strong target neuron
            src_indices = torch.randint(0, self.res_size, (len(strong_indices),), device=self.slow_trail.device)
            
            # 2. Weaken connection by subtracting 0.1 from absolute value
            is_excitatory = src_indices < self.e_size
            delta = torch.where(is_excitatory, -0.1, 0.1)
            
            with torch.no_grad():
                self.W[strong_indices, src_indices] += delta
                # Enforce E-I constraints (excitatory columns positive, inhibitory negative)
                self.W[:, :self.e_size] = torch.clamp(self.W[:, :self.e_size], min=0.0)
                self.W[:, self.e_size:] = torch.clamp(self.W[:, self.e_size:], max=0.0)

        return self.readout(self.state)

    def step_no_evolution(self, x):
        x_in = torch.matmul(self.W_in, x)
        x_1 = torch.matmul(self.state, self.W.T)
        # Standard non-leaky ESN update (states bounded to [-1, 1] by tanh)
        self.state = torch.tanh(self.scaling * (x_1 + x_in))
        return self.readout(self.state)

    def reset_state(self):
        self.state = torch.zeros(self.state.shape)
        self.genesis_ops_total = 0
        self.pruning_ops_total = 0
        self.latest_genesis_ops = 0
        self.latest_pruning_ops = 0




