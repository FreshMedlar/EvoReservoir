import torch
import torch.nn as nn

class Reservoir():
    def __init__(self, res_size, output_dim, input_scaling=0.5, spectral_radius=0.9):
        self.res_size = res_size
        
        ### Weights initialization
        W = torch.randn(res_size, res_size)
        radius = torch.max(torch.abs(torch.linalg.eigvals(W)))
        self.W = nn.Parameter(W * (spectral_radius / radius), requires_grad=False)
        self.W_in = nn.Parameter(torch.randn(res_size, output_dim) * input_scaling, requires_grad=False)
        
        self.state = torch.zeros(res_size)
        self.readout = nn.Linear(res_size, output_dim, bias=False)
        
    # perform one step in the reservoir
    def step(self, x):
        x_in = torch.matmul(self.W_in, x)
        x_1 = torch.matmul(self.state, self.W.T)
        # Standard non-leaky ESN update
        self.state = torch.tanh(x_1 + x_in)
        return self.readout(self.state)

    def reset_state(self):
        self.state = torch.zeros(self.state.shape)
