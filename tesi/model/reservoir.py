import numpy as np
import torch
import torch.nn as nn

class Reservoir():
    def __init__(self, res_size, output_dim, input_scaling=0.5):
        self.res_size = res_size
        
        spectral_radius = 0.95
        ### Weights initialization 
        W = torch.randn(res_size, res_size)
        eigvals = torch.linalg.eigvals(W)
        rad = torch.max(torch.abs(eigvals))
        self.W = nn.Parameter(W * (spectral_radius / rad), requires_grad=False)
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
        low_mask = self.slow_trail < 0.1
        high_mask = self.slow_trail > 0.9
        normal_mask = ~(low_mask | high_mask)  # inclusive
        
        self.scaling[low_mask] += 0.01
        self.scaling[high_mask] -= 0.01
        self.scaling[normal_mask] = 1.0

        return self.readout(self.state)

    def reset_state(self):
        self.state = torch.zeros(self.state.shape)




