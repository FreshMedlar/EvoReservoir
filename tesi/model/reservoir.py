import numpy as np
import torch
import torch.nn as nn

class Reservoir():
    def __init__(self, res_size, output_dim, input_scaling=0.5, e_ratio=0.8, density=0.05):
        self.res_size = res_size
        self.e_size = int(res_size * e_ratio)
        
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
        self.W = nn.Parameter(W * (0.9 / radius), requires_grad=False)
        self.W_in = nn.Parameter(torch.randn(res_size, output_dim) * input_scaling, requires_grad=False)
        # per neuron trailing activation
        self.fast_trail = torch.zeros(res_size)
        self.slow_trail = torch.zeros(res_size)

        self.state = torch.zeros(res_size)
        self.scaling = torch.ones(res_size)

        self.readout = nn.Linear(res_size, output_dim, bias=False)
        
        
    # perform one step in the reservoir
    def step(self, x):
        x_in = torch.matmul(self.W_in, x)
        x_1 = torch.matmul(self.state, self.W.T)
        # Standard non-leaky ESN update (states bounded to [-1, 1] by tanh)
        self.state = torch.tanh(self.scaling * (x_1 + x_in))

        # trails update
        self.slow_trail = 0.99 * self.slow_trail + 0.01 * self.state
        self.fast_trail = 0.8 * self.fast_trail + 0.2 * self.state

        ### HOMEOSTASIS STEP
        abs_slow = torch.abs(self.slow_trail)
        low_mask = abs_slow < 0.1
        high_mask = abs_slow > 0.9
        normal_mask = ~(low_mask | high_mask)  # inclusive
        
        self.scaling[low_mask] += 0.01
        self.scaling[high_mask] -= 0.01
        self.scaling[normal_mask] = 1.0
        self.scaling = torch.clamp(self.scaling, min=0.1, max=5.0)

        ### genesis
        weak_mask = abs_slow < 0.1
        strong_mask = abs_slow > 0.9
        weak_indices = torch.where(weak_mask)[0]
        strong_indices = torch.where(strong_mask)[0]
        
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
        

        return self.readout(self.state)

    def reset_state(self):
        self.state = torch.zeros(self.state.shape)




